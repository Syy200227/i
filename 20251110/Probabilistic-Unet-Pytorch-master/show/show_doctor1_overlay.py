# doctor1_overlay_auto_by_sop.py
# 作用：从 XML 里拿到医生1的 ROI → 用 SOP 匹配到正确 DICOM 切片 → 叠加显示（不保存）
import os, math, glob
import numpy as np
import pydicom
import xml.etree.ElementTree as ET
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pydicom.multival import MultiValue

# —— 放在画图前 ——
import matplotlib
from matplotlib import font_manager

def pick_cjk_font():
    candidates = [
        "SimHei", "Microsoft YaHei", "Microsoft JhengHei",
        "PingFang SC", "Hiragino Sans GB",
        "Noto Sans CJK SC", "Source Han Sans SC",
        "WenQuanYi Zen Hei", "Arial Unicode MS"
    ]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in have:
            return name
    return None

cjk = pick_cjk_font()
if cjk:
    matplotlib.rcParams["font.sans-serif"] = [cjk, "DejaVu Sans"]
# 让负号也正常显示
matplotlib.rcParams["axes.unicode_minus"] = False

# ========== 基础工具 ==========
def _first_number(v):
    if v is None: return None
    if isinstance(v, (list, MultiValue)):
        try: return float(v[0])
        except: return None
    try: return float(v)
    except: return None

def dcm_to_hu(ds):
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept

def window_image(hu, center=None, width=None):
    if center is None or width is None or math.isclose(width or 0.0, 0.0):
        lo, hi = np.percentile(hu, [1, 99])
    else:
        lo, hi = center - width/2.0, center + width/2.0
    hu = np.clip(hu, lo, hi)
    return (hu - lo) / max(hi - lo, 1e-6)

def maybe_invert(gray01, ds):
    photometric = getattr(ds, "PhotometricInterpretation", "").upper()
    return (1.0 - gray01) if "MONOCHROME1" in photometric else gray01

def polygon_to_mask(contour_xy_1based, rows, cols):
    pts0 = [(x-1, y-1) for (x, y) in contour_xy_1based]  # 1-based → 0-based
    img = Image.new("L", (cols, rows), 0)
    ImageDraw.Draw(img).polygon(pts0, outline=1, fill=1)
    return np.array(img, dtype=np.uint8)

# ========== XML：拿医生1的 ROI（SOP、Z、点数、坐标） ==========
def get_doctor1_rois(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"lidc": root.tag.split('}')[0].strip('{')}

    rs_list = root.findall(".//lidc:readingSession", ns)
    if not rs_list:
        return []

    rs1 = rs_list[2]#切换医生0123
    rois = []
    for nodule in rs1.findall("lidc:unblindedReadNodule", ns):
        for roi in nodule.findall("lidc:roi", ns):
            sop = (roi.findtext("lidc:imageSOP_UID", default="", namespaces=ns) or "").strip()
            ztxt = roi.findtext("lidc:imageZposition", default="", namespaces=ns)
            try: z = float(ztxt)
            except: z = None
            pts = []
            for em in roi.findall("lidc:edgeMap", ns):
                x = em.findtext("lidc:xCoord", default="", namespaces=ns)
                y = em.findtext("lidc:yCoord", default="", namespaces=ns)
                if x and y:
                    pts.append((int(x), int(y)))
            rois.append({"sop": sop, "z": z, "pts": pts})
    return rois

# ========== 在 dicom_root 里递归查找与 SOP 匹配的 DICOM ==========
def build_sop_to_path_index(dicom_root):
    idx = {}
    for path in glob.iglob(os.path.join(dicom_root, "**", "*.dcm"), recursive=True):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            sop = str(ds.SOPInstanceUID)
            idx[sop] = path
        except Exception:
            continue
    return idx

# ========== 主流程：自动找对的切片并叠加 ==========
def show_overlay_by_xml_sop(xml_path, dicom_root, prefer_center_slice=True, outline=False):
    rois = get_doctor1_rois(xml_path)
    if not rois:
        print("❌ XML 中未找到医生1的 ROI")
        return

    # 只保留可封闭的轮廓（点数>=3）
    rois = [r for r in rois if len(r["pts"]) >= 3 and r["sop"]]
    if not rois:
        print("❌ 医生1的 ROI 都是单点/线段或缺 SOP，无法叠加")
        return

    # 选一层：默认选“中间层”（也可选点数最多的一层）
    rois_sorted = sorted(rois, key=lambda r: (r["z"] is None, r["z"]))
    if prefer_center_slice and any(r["z"] is not None for r in rois_sorted):
        zs = [r["z"] for r in rois_sorted if r["z"] is not None]
        z_mid = zs[len(zs)//2]
        cand = min([r for r in rois_sorted if r["z"] is not None], key=lambda r: abs(r["z"] - z_mid))
        target = cand
    else:
        # 回退：选点数最多的
        target = max(rois_sorted, key=lambda r: len(r["pts"]))

    print(f"🩺 目标层：Z={target['z']} , SOP={target['sop'][:40]}... , 点数={len(target['pts'])}")

    # 建 SOP→路径 索引并命中
    sop2path = build_sop_to_path_index(dicom_root)
    if target["sop"] not in sop2path:
        print("❌ 在 dicom_root 中找不到与该 SOP 匹配的 .dcm 文件。请确认 dicom_root 指向正确病例/序列的根目录。")
        return

    dcm_path = sop2path[target["sop"]]
    print(f"✅ 命中 DICOM：{dcm_path}")

    # 读取 DICOM 并渲染
    ds = pydicom.dcmread(dcm_path)
    hu = dcm_to_hu(ds)
    wc = _first_number(getattr(ds, "WindowCenter", None))
    ww = _first_number(getattr(ds, "WindowWidth", None))
    gray01 = window_image(hu, wc, ww)
    gray01 = maybe_invert(gray01, ds)

    rows, cols = int(ds.Rows), int(ds.Columns)

    # 生成总 mask（如果这一层有多个 ROI，就都叠加）
    mask = np.zeros((rows, cols), dtype=np.uint8)
    for r in rois:
        if r["sop"] == target["sop"]:
            mask += polygon_to_mask(r["pts"], rows, cols)
    mask = np.clip(mask, 0, 1)

    # 显示（不保存）
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow((gray01 * 255).astype(np.uint8), cmap="gray")
    if mask.any():
        if outline:
            from scipy.ndimage import binary_erosion
            edge = mask.astype(bool) ^ binary_erosion(mask.astype(bool))
            edge_rgb = np.zeros((rows, cols, 3), dtype=np.float32)
            edge_rgb[..., 0] = edge * 1.0
            ax.imshow(edge_rgb, alpha=0.9)
        else:
            ax.imshow(mask, cmap="Reds", alpha=0.35)
    else:
        print("ℹ️ 该 SOP 层没有可填充的闭合轮廓（不太可能，除非都被判为单点/线段）")
    ax.set_title("医生1 标注叠加（SOP 精确匹配 / 未落盘）", pad=10)
    ax.axis("off")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 只改这两行：
    xml_path   = r"E:\workspace\LIDC-IDRI\CT-manifest-1760698817279\LIDC-IDRI\LIDC-IDRI-0005\01-01-2000-NA-NA-42125\3000548.000000-NA-86225\076.xml"#指向具体的xml文件
    dicom_root = r"E:\workspace\LIDC-IDRI\CT-manifest-1760698817279\LIDC-IDRI\LIDC-IDRI-0005\01-01-2000-NA-NA-42125\3000548.000000-NA-86225"  # 指向病例根目录；脚本会递归搜索 *.dcm

    show_overlay_by_xml_sop(xml_path, dicom_root, prefer_center_slice=True, outline=False)
