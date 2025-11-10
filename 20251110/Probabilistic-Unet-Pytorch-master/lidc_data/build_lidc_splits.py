# -*- coding: utf-8 -*-
"""
构建 LIDC-IDRI 数据集的 train/val/test(5:1:1) 划分，并分别导出到
  images_root/split/患者/SeriesUID/SOPUID/image.dcm
  labels_root/split/患者/SeriesUID/SOPUID/{mask_r*.png, poly_r*.json, meta.json}
说明：
- 仅将 XML 中 点数>=3 的 ROI 视为有效多边形并栅格化为掩膜；len=1 的视为非结节/种子点，忽略
- 划分在“患者级”完成，避免信息泄漏
"""

import os, glob, json, csv, shutil, random
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
import pydicom
from tqdm import tqdm

# ==================== 你需要改的配置 ====================
RAW_ROOT      = r"E:\workspace\LIDC-IDRI\processingCT_manifest_1760698817279\LIDC-IDRI"  # 原始LIDC根目录
IMAGES_ROOT   = r"E:\workspace\LIDC-IDRI\per\dataset_images"   # 输出：图像大根目录
LABELS_ROOT   = r"E:\workspace\LIDC-IDRI\per\dataset_labels"   # 输出：标注大根目录
COPY_DICOM    = True    # True=复制DICOM到images_root；False=只在meta里记录原始相对路径
INCLUDE_EMPTY = False   # 是否包含“无任何医生多边形”的切片作为负样本
SPLIT_SEED    = 2025    # 划分随机种子
SPLIT_RATIO   = (5,1,1) # train:val:test
WINDOW_CENTER = -600.0  # 如果你也想导出窗宽窗位图像，可在meta中记录（本脚本不导出npy以节省空间）
WINDOW_WIDTH  = 1500.0
# =====================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_lidc_xml(xml_path):
    """
    读取一个 series 目录下的 XML，返回:
      readers[reader_id][sop_uid] = [polygon, ...]
    注意:
    - 去掉 XML 命名空间，保证不同版本的 LIDC XML 都能解析
    - 只把 点数>=3 的 ROI 当作可栅格化的多边形；len=1 的“种子点”忽略
    """
    import xml.etree.ElementTree as ET
    from collections import defaultdict

    def strip_ns(tag):
        # '{namespace}tag' -> 'tag'
        return tag.split('}', 1)[-1] if '}' in tag else tag

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 先把所有元素的 tag 替换为去命名空间的裸名，后续统一用裸名匹配
    for el in root.iter():
        el.tag = strip_ns(el.tag)

    readers = defaultdict(lambda: defaultdict(list))

    # readingSession（最多 4 个医生）
    for r_idx, reading in enumerate(root.findall(".//readingSession"), start=1):
        # 结节（有边界）
        for nodule in reading.findall("./unblindedReadNodule"):
            for roi in nodule.findall("./roi"):
                sop_el = roi.find("./imageSOP_UID")
                if sop_el is None or (sop_el.text is None):
                    continue
                sop = sop_el.text.strip()

                # 收集该 ROI 的边界点
                poly = []
                for em in roi.findall("./edgeMap"):
                    x = em.find("./xCoord"); y = em.find("./yCoord")
                    if x is None or y is None or x.text is None or y.text is None:
                        continue
                    # LIDC 坐标是 1-based 像素坐标，转为 0-based；(x,y) 即 (列,行)
                    try:
                        xx = int(round(float(x.text))) - 1
                        yy = int(round(float(y.text))) - 1
                        poly.append((xx, yy))
                    except Exception:
                        continue

                # 只保留能构成多边形的 ROI
                if len(poly) >= 3:
                    readers[r_idx][sop].append(poly)

        # 非结节（nonNodule）不参与监督，这里不加入

    return dict(readers)


def rasterize_polygons(polys, hw):
    """将若干polygon并集栅格化为二值mask，hw=(H,W)"""
    H, W = hw
    img = Image.new("L", (W, H), 0)
    drw = ImageDraw.Draw(img)
    for p in polys:
        if len(p) >= 3:
            drw.polygon(p, outline=1, fill=1)
    return np.array(img, dtype=np.uint8)

def scan_all_series(raw_root):
    """递归扫描，找到同时含有 *.xml 与 *.dcm 的最内层目录"""
    series = []
    for patient in sorted(glob.glob(os.path.join(raw_root, "LIDC-*"))):
        for root, dirs, files in os.walk(patient):
            has_xml = any(f.lower().endswith(".xml") for f in files)
            has_dcm = any(f.lower().endswith(".dcm") for f in files)
            if has_xml and has_dcm:
                # 假设一个series目录仅有一个xml（若有多个，可自行选择规则）
                xmls = [os.path.join(root, f) for f in files if f.lower().endswith(".xml")]
                series.append((patient, root, xmls[0]))
    return series

def collect_slices(series_tuple):
    """从一个series目录中收集所有切片，连同4位医师的多边形"""
    patient_dir, series_dir, xml_path = series_tuple
    readers = parse_lidc_xml(xml_path)
    # 读一个dcm拿SeriesUID
    d0 = pydicom.dcmread(sorted(glob.glob(os.path.join(series_dir, "*.dcm")))[0], stop_before_pixels=True)
    series_uid = str(getattr(d0, "SeriesInstanceUID", Path(series_dir).name))

    slices = []
    for dcm_path in sorted(glob.glob(os.path.join(series_dir, "*.dcm"))):
        try:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        except Exception as e:
            # 有些压缩传输语法可能读不了：略过
            continue
        sop = str(ds.SOPInstanceUID)
        pxsp = getattr(ds, "PixelSpacing", [1.0,1.0])
        thk = float(getattr(ds, "SliceThickness", 1.0))
        rows = int(getattr(ds, "Rows", 0)); cols = int(getattr(ds, "Columns", 0))

        # 收集四位医师的polygon
        polys_per_reader = []
        has_any_polygon = False
        for rid in range(1,5):
            polys = readers.get(rid, {}).get(sop, [])
            polys_per_reader.append(polys)
            if len(polys) > 0:
                has_any_polygon = True

        if (not INCLUDE_EMPTY) and (not has_any_polygon):
            # 跳过完全无多边形的切片
            continue

        slices.append({
            "patient_id": Path(patient_dir).name,
            "series_uid": series_uid,
            "sop_uid": sop,
            "series_dir": series_dir,
            "dcm_path": dcm_path,
            "H": rows, "W": cols,
            "pixel_spacing": [float(pxsp[0]), float(pxsp[1])],
            "slice_thickness": thk,
            "polys_per_reader": polys_per_reader
        })
    return slices

def split_patients(all_patient_ids, ratio=(5,1,1), seed=2025):
    """按患者ID划分 5:1:1"""
    random.seed(seed)
    ids = list(sorted(set(all_patient_ids)))
    random.shuffle(ids)
    total = sum(ratio)
    n = len(ids)
    n_train = int(round(n * ratio[0] / total))
    n_val   = int(round(n * ratio[1] / total))
    train_ids = ids[:n_train]
    val_ids   = ids[n_train:n_train+n_val]
    test_ids  = ids[n_train+n_val:]
    return set(train_ids), set(val_ids), set(test_ids)

def export_split(slices, split_name, images_root, labels_root):
    """将一个split的切片导出到目标目录；返回写入的行数"""
    img_root = Path(images_root) / split_name
    lab_root = Path(labels_root) / split_name
    ensure_dir(img_root); ensure_dir(lab_root)

    manifest_path = Path(labels_root) / f"split_manifest_{split_name}.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow([
            "patient_id","series_uid","sop_uid",
            "rel_image_dir","rel_label_dir",
            "H","W","pixel_spacing_row","pixel_spacing_col","slice_thickness",
            "has_r1","has_r2","has_r3","has_r4",
            "wl","ww"
        ])

        writen = 0
        for item in tqdm(slices, desc=f"Export {split_name}"):
            pid = item["patient_id"]; suid = item["series_uid"]; sop = item["sop_uid"]
            # 目录：split/患者/SeriesUID/SOPUID/
            img_dir = img_root / pid / suid / sop
            lab_dir = lab_root / pid / suid / sop
            ensure_dir(img_dir); ensure_dir(lab_dir)

            # 1) 图像侧：复制或记录DICOM
            if COPY_DICOM:
                dst = img_dir / "image.dcm"
                try:
                    shutil.copy2(item["dcm_path"], dst)
                except Exception as e:
                    # 复制失败可跳过这个切片
                    continue
            # 2) 标注侧：栅格化四位医师的掩膜 + 保存多边形与meta
            H, W = item["H"], item["W"]
            has = []
            for rid in range(1,5):
                polys = item["polys_per_reader"][rid-1]
                has.append(1 if len(polys) > 0 else 0)
                # 掩膜
                mask = rasterize_polygons(polys, (H, W)) if polys else np.zeros((H,W), np.uint8)
                Image.fromarray(mask*255, mode="L").save(lab_dir / f"mask_r{rid}.png")
                # 原始多边形
                with open(lab_dir / f"poly_r{rid}.json", "w", encoding="utf-8") as fj:
                    json.dump(polys, fj, ensure_ascii=False)

            meta = {
                "patient_id": pid,
                "series_uid": suid,
                "sop_uid": sop,
                "image_relpath": str((Path(split_name)/pid/suid/sop/"image.dcm").as_posix()) if COPY_DICOM else None,
                "pixel_spacing": item["pixel_spacing"],
                "slice_thickness": item["slice_thickness"],
                "shape_hw": [H, W],
                "wl": WINDOW_CENTER, "ww": WINDOW_WIDTH
            }
            with open(lab_dir / "meta.json", "w", encoding="utf-8") as fm:
                json.dump(meta, fm, ensure_ascii=False, indent=2)

            w.writerow([
                pid, suid, sop,
                str((Path(split_name)/pid/suid/sop).as_posix()),
                str((Path(split_name)/pid/suid/sop).as_posix()),
                H, W, item["pixel_spacing"][0], item["pixel_spacing"][1], item["slice_thickness"],
                has[0], has[1], has[2], has[3],
                WINDOW_CENTER, WINDOW_WIDTH
            ])
            writen += 1
    return writen

def main():
    print(">>> 扫描 series ...")
    series_list = scan_all_series(RAW_ROOT)
    if not series_list:
        print("未找到同时含 DICOM 与 XML 的目录，请检查 RAW_ROOT 是否正确。")
        return

    print(f"共发现 series 目录：{len(series_list)}")
    print(f"共发现 series 目录：{len(series_list)}")

    # === 诊断：看看前几个 series 的多边形匹配情况 ===
    for idx, s in enumerate(series_list[:3], start=1):
        patient_dir, series_dir, xml_path = s
        readers = parse_lidc_xml(xml_path)
        dcm_list = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
        good = 0
        for dcm_path in dcm_list:
            try:
                ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                sop = str(ds.SOPInstanceUID)
            except Exception:
                continue
            polys = 0
            for rid in range(1,5):
                polys += len(readers.get(rid, {}).get(sop, []))
            if polys > 0:
                good += 1
        print(f"[DEBUG] series[{idx}] {Path(series_dir).name}: 有多边形的切片 {good} / {len(dcm_list)}")

    all_slices = []
    for s in tqdm(series_list, desc="解析XML并匹配切片"):
        slc = collect_slices(s)
        all_slices.extend(slc)

    if not all_slices:
        print("没有可导出的切片（可能因为 INCLUDE_EMPTY=False 且无结节被过滤）。")
        return

    all_patients = [x["patient_id"] for x in all_slices]
    tr_ids, va_ids, te_ids = split_patients(all_patients, SPLIT_RATIO, SPLIT_SEED)
    print(f"患者级划分：train={len(tr_ids)}, val={len(va_ids)}, test={len(te_ids)}")

    train_slices = [x for x in all_slices if x["patient_id"] in tr_ids]
    val_slices   = [x for x in all_slices if x["patient_id"] in va_ids]
    test_slices  = [x for x in all_slices if x["patient_id"] in te_ids]
    print(f"切片计数：train={len(train_slices)}, val={len(val_slices)}, test={len(test_slices)}")

    ensure_dir(Path(IMAGES_ROOT)); ensure_dir(Path(LABELS_ROOT))

    n_tr = export_split(train_slices, "train", IMAGES_ROOT, LABELS_ROOT)
    n_va = export_split(val_slices,   "val",   IMAGES_ROOT, LABELS_ROOT)
    n_te = export_split(test_slices,  "test",  IMAGES_ROOT, LABELS_ROOT)
    print(f"完成：train={n_tr}, val={n_va}, test={n_te}")
    print(f"清单文件：\n  {Path(LABELS_ROOT)/'split_manifest_train.csv'}\n  {Path(LABELS_ROOT)/'split_manifest_val.csv'}\n  {Path(LABELS_ROOT)/'split_manifest_test.csv'}")

if __name__ == "__main__":
    main()
