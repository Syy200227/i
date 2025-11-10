import os
import math
import numpy as np
import pydicom
from pydicom.multival import MultiValue
import matplotlib.pyplot as plt

# ---------- 工具函数 ----------
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

def _first_number(v):
    if v is None:
        return None
    if isinstance(v, (list, MultiValue)):
        try:
            return float(v[0])
        except Exception:
            return None
    try:
        return float(v)
    except Exception:
        return None

def print_brief_meta(ds):
    """打印常见 DICOM 头信息（可按需增减）"""
    def val(name, default=""):
        return getattr(ds, name, default)
    wc = _first_number(getattr(ds, "WindowCenter", None))
    ww = _first_number(getattr(ds, "WindowWidth", None))
    print("-" * 72)
    print("DICOM 基本信息：")
    print(f"SOP Class UID        : {val('SOPClassUID', '')}")
    print(f"SOP Instance UID     : {val('SOPInstanceUID', '')}")
    print(f"Modality             : {val('Modality', '')}")
    print(f"Manufacturer         : {val('Manufacturer', '')}")
    print(f"Model Name           : {val('ManufacturerModelName', '')}")
    print(f"Study Date           : {val('StudyDate', '')}")
    print(f"Series Date          : {val('SeriesDate', '')}")
    print(f"Patient ID           : {val('PatientID', '')}")
    print(f"Patient Sex          : {val('PatientSex', '')}")
    print(f"Body Part Examined   : {val('BodyPartExamined', '')}")
    print(f"Slice Thickness (mm) : {val('SliceThickness', '')}")
    print(f"KVP (kV)             : {val('KVP', '')}")
    print(f"Rows x Cols          : {val('Rows', '')} x {val('Columns', '')}")
    print(f"Window C / W         : {wc} / {ww}")
    print(f"RescaleSlope/Intercept: {val('RescaleSlope',1)} / {val('RescaleIntercept',0)}")
    print(f"PhotometricInterp.   : {val('PhotometricInterpretation','')}")
    print("-" * 72)
    # 需要完整标签表可解注释：
    print(ds)

def dcm_to_hu(ds):
    """像素 -> HU（自动处理压缩 Transfer Syntax）"""
    # 若是压缩传输语法，pixel_array 可能需要解码插件
    try:
        arr = ds.pixel_array.astype(np.float32)
    except Exception as e:
        # 尝试显式解压；若仍失败，提示安装解码插件
        try:
            ds.decompress()
            arr = ds.pixel_array.astype(np.float32)
        except Exception as e2:
            raise RuntimeError(
                "无法读取像素数据。若是 JPEG2000/JPEG 压缩 DICOM，请先安装：\n"
                "  pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg\n"
                "或 gdcm：\n"
                "  pip install gdcm\n原始错误：" + str(e2)
            ) from e2

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept
    return hu

def auto_window_center_width(hu, center=None, width=None, fallback="percentile"):
    """应用窗宽窗位，返回[0,1]灰度。优先用DICOM里的WC/WW，缺失则回退。"""
    if center is None or width is None or math.isclose(width or 0.0, 0.0):
        if fallback == "percentile":
            lo, hi = np.percentile(hu, [1, 99])
        else:
            lo, hi = float(np.min(hu)), float(np.max(hu))
    else:
        lo = center - width / 2.0
        hi = center + width / 2.0
    hu = np.clip(hu, lo, hi)
    gray01 = (hu - lo) / max(hi - lo, 1e-6)
    return gray01

def maybe_invert(gray01, ds):
    """MONOCHROME1 需要反色"""
    photometric = getattr(ds, "PhotometricInterpretation", "").upper()
    if "MONOCHROME1" in photometric:
        return 1.0 - gray01
    return gray01

def show_dicom(dcm_path, use_percentile_if_missing=True):
    """读取一个 DICOM，打印信息并在编译器内显示图像（不落盘）"""
    ds = pydicom.dcmread(dcm_path)
    print_brief_meta(ds)

    # 1) HU
    hu = dcm_to_hu(ds)

    # 2) WC/WW
    wc = _first_number(getattr(ds, "WindowCenter", None))
    ww = _first_number(getattr(ds, "WindowWidth", None))
    fallback = "percentile" if use_percentile_if_missing else "minmax"
    gray01 = auto_window_center_width(hu, wc, ww, fallback=fallback)

    # 3) 反色（如 MONOCHROME1）
    gray01 = maybe_invert(gray01, ds)

    # 4) 显示（仅内存，不保存）
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.title("DICOM → PNG 预览（未落盘）")
    plt.imshow((gray01 * 255).astype(np.uint8), cmap="gray")
    plt.show()

# ---------- 运行 ----------
if __name__ == "__main__":
    dcm_path = r"E:\workspace\LIDC-IDRI\per\dataset_images\train\LIDC-IDRI-0005\1.3.6.1.4.1.14519.5.2.1.6279.6001.129007566048223160327836686225\1.3.6.1.4.1.14519.5.2.1.6279.6001.158628732394963585751959079142\image.dcm"
    show_dicom(dcm_path)
