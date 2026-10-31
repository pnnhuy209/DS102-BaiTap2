import csv
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageDraw


API_URL = "https://commons.wikimedia.org/w/api.php"
THU_MUC = Path("du_lieu_crawl")
THU_MUC_ANH = THU_MUC / "anh_goc"
SO_ANH_MOI_LOP = 12
SEED = 42

HEADERS = {
    "User-Agent": "MNIST-student-assignment/1.0 (educational project)"
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TEN_SO = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


def goi_api(tham_so):
    url = API_URL + "?" + urllib.parse.urlencode(tham_so)
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def bo_html(text):
    if not text:
        return ""
    ket_qua = []
    dang_o_trong_the = False
    for ky_tu in text:
        if ky_tu == "<":
            dang_o_trong_the = True
        elif ky_tu == ">":
            dang_o_trong_the = False
        elif not dang_o_trong_the:
            ket_qua.append(ky_tu)
    return "".join(ket_qua).strip()


def tim_anh_cua_so(nhan, so_luong):
    ket_qua = []
    page_da_xem = set()

    def them_pages(pages):
        for page in pages.values():
            page_id = page.get("pageid")
            if page_id in page_da_xem:
                continue
            page_da_xem.add(page_id)

            imageinfo = page.get("imageinfo", [])
            if not imageinfo:
                continue

            info = imageinfo[0]
            mime = info.get("mime", "")
            if not mime.startswith("image/"):
                continue

            metadata = info.get("extmetadata", {})
            ket_qua.append({
                "nhan": nhan,
                "title": page.get("title", ""),
                "page_url": info.get("descriptionurl", ""),
                "image_url": info.get("thumburl") or info.get("url", ""),
                "author": bo_html(metadata.get("Artist", {}).get("value", "")),
                "license": metadata.get("LicenseShortName", {}).get("value", ""),
            })

    cac_category = [
        f"Category:{nhan} (numeral)",
        f"Category:{TEN_SO[nhan]} (numeral)",
    ]
    for category in cac_category:
        tham_so = {
            "action": "query",
            "format": "json",
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": "file",
            "gcmlimit": 50,
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata",
            "iiurlwidth": 256,
        }
        du_lieu = goi_api(tham_so)
        them_pages(du_lieu.get("query", {}).get("pages", {}))
        if len(ket_qua) >= so_luong:
            return ket_qua[:so_luong]

    cac_truy_van = [
        f'handwritten digit {nhan} {TEN_SO[nhan]}',
        f'handwritten numeral {nhan}',
        f'"digit {nhan}"',
        f'"numeral {nhan}"',
        f'"number {nhan}" symbol',
    ]

    for tu_khoa in cac_truy_van:
        tham_so = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrnamespace": 6,
            "gsrsearch": tu_khoa,
            "gsrlimit": 50,
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata",
            "iiurlwidth": 256,
        }
        du_lieu = goi_api(tham_so)
        pages = du_lieu.get("query", {}).get("pages", {})
        them_pages(pages)

        if len(ket_qua) >= so_luong:
            break

    return ket_qua[:so_luong]


def tai_va_xu_ly_anh(muc_anh, vi_tri):
    request = urllib.request.Request(muc_anh["image_url"], headers=HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        noi_dung = response.read()

    thu_muc_nhan = THU_MUC_ANH / str(muc_anh["nhan"])
    thu_muc_nhan.mkdir(parents=True, exist_ok=True)
    ten_anh = f"{muc_anh['nhan']}_{vi_tri:03d}.png"
    duong_dan = thu_muc_nhan / ten_anh

    with Image.open(io.BytesIO(noi_dung)) as anh:
        anh = ImageOps.exif_transpose(anh).convert("RGB")
        anh.thumbnail((256, 256))
        anh.save(duong_dan, format="PNG")

        anh_xam = ImageOps.grayscale(anh)
        anh_xam.thumbnail((24, 24))
        nen = Image.new("L", (28, 28), color=255)
        x = (28 - anh_xam.width) // 2
        y = (28 - anh_xam.height) // 2
        nen.paste(anh_xam, (x, y))
        mang_anh = np.asarray(nen, dtype=np.float32) / 255.0

    muc_anh["local_file"] = str(duong_dan)
    return mang_anh


def tao_anh_xem_truoc(anh, nhan):
    so_cot = 10
    so_hang = int(np.ceil(len(anh) / so_cot))
    kich_thuoc_o = 56
    preview = Image.new("RGB", (so_cot * kich_thuoc_o, so_hang * kich_thuoc_o), "white")
    draw = ImageDraw.Draw(preview)

    for i, (mang_anh, label) in enumerate(zip(anh, nhan)):
        hinh = Image.fromarray((mang_anh * 255).astype(np.uint8)).resize((40, 40))
        hinh = hinh.convert("RGB")
        x = (i % so_cot) * kich_thuoc_o + 8
        y = (i // so_cot) * kich_thuoc_o + 12
        preview.paste(hinh, (x, y))
        draw.text((x, 1 + (i // so_cot) * kich_thuoc_o), str(label), fill="black")

    preview.save(THU_MUC / "preview.png")


def main():
    THU_MUC.mkdir(exist_ok=True)
    THU_MUC_ANH.mkdir(exist_ok=True)

    tat_ca_anh = []
    tat_ca_nhan = []
    metadata = []
    url_da_tai = set()

    for nhan in range(10):
        print(f"Đang tìm ảnh cho số {nhan} ...")
        danh_sach = tim_anh_cua_so(nhan, SO_ANH_MOI_LOP * 2)
        da_lay = 0

        for muc_anh in danh_sach:
            if da_lay >= SO_ANH_MOI_LOP:
                break
            if not muc_anh["image_url"] or muc_anh["image_url"] in url_da_tai:
                continue

            try:
                mang_anh = tai_va_xu_ly_anh(muc_anh, da_lay + 1)
            except Exception as loi:
                print("  Bỏ qua ảnh lỗi:", loi)
                continue

            tat_ca_anh.append(mang_anh)
            tat_ca_nhan.append(nhan)
            metadata.append(muc_anh)
            url_da_tai.add(muc_anh["image_url"])
            da_lay += 1
            time.sleep(0.4)

        print(f"  Đã tải {da_lay} ảnh")

    X = np.asarray(tat_ca_anh, dtype=np.float32).reshape(-1, 784)
    y = np.asarray(tat_ca_nhan, dtype=np.uint8)

    rng = np.random.default_rng(SEED)
    train_index = []
    test_index = []
    for nhan in range(10):
        vi_tri = np.where(y == nhan)[0]
        rng.shuffle(vi_tri)
        moc = max(1, int(len(vi_tri) * 0.8))
        train_index.extend(vi_tri[:moc])
        test_index.extend(vi_tri[moc:])

    train_index = np.asarray(train_index)
    test_index = np.asarray(test_index)
    np.savez_compressed(
        THU_MUC / "dataset_crawl.npz",
        X_train=X[train_index],
        y_train=y[train_index],
        X_test=X[test_index],
        y_test=y[test_index],
    )

    with open(THU_MUC / "metadata.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["nhan", "title", "page_url", "image_url", "author", "license", "local_file"],
        )
        writer.writeheader()
        writer.writerows(metadata)

    tao_anh_xem_truoc(X.reshape(-1, 28, 28), y)

    print("\nHoàn thành.")
    print("Tổng số ảnh:", len(X))
    print("Tập train:", len(train_index))
    print("Tập test :", len(test_index))
    print("Hãy mở du_lieu_crawl/preview.png để kiểm tra và loại ảnh sai nhãn.")


if __name__ == "__main__":
    main()

