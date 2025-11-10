# -*- coding: utf-8 -*-
# 读取你“划分后”的目录结构：dataset_images & dataset_labels
import os, json, glob, random
from pathlib import Path
import numpy as np
from PIL import Image
import pydicom
import torch
from torch.utils.data import Dataset

def _dcm_to_hu(ds):
    arr = ds.pixel_array.astype(np.int16)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return (arr * slope + intercept).astype(np.float32)

def _apply_window_hu(x_hu, wl=-600.0, ww=1500.0):
    lo, hi = wl - ww/2.0, wl + ww/2.0
    x = np.clip((x_hu - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return x.astype(np.float32)

def _resize_hw(arr, size):
    if size is None: return arr
    mode = "F" if arr.dtype in (np.float32, np.float64) else "L"
    im = Image.fromarray(arr, mode=mode)
    im = im.resize((size, size), Image.BILINEAR)
    return np.array(im, dtype=arr.dtype)

class LIDCFromSplits(Dataset):
    """
    从 dataset_images/ & dataset_labels/ 读取一个 split（train/val/test）
    - images_root / labels_root: 两个大根目录
    - split: "train"|"val"|"test"
    - img_size: 统一缩放到正方形输入（ProbUNet默认下采样4次，建议16的倍数）
    - return_all_masks: True 返回4张掩膜 [4,H,W]；False 随机选一张 [H,W]
    - seed: 控制“随机挑医师”的可复现性
    """
    def __init__(self, images_root, labels_root, split="train",
                 img_size=192, return_all_masks=False, seed=2025):
        super().__init__()
        self.images_root = Path(images_root)
        self.labels_root = Path(labels_root)
        self.split = split
        self.img_size = img_size
        self.return_all_masks = return_all_masks
        self.rng = random.Random(seed)

        # 遍历 labels/{split}/**/**/**/meta.json 作为索引（它记录了 image_relpath）
        self.items = []
        label_split_root = self.labels_root / split
        for meta_path in label_split_root.rglob("meta.json"):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # 定位对应的 DICOM
            if meta.get("image_relpath"):
                dcm_path = (self.images_root / meta["image_relpath"]).resolve()
            else:
                # 回退：与 labels 同层级的 images 结构一致
                rel = meta_path.relative_to(self.labels_root / split).parent
                dcm_path = (self.images_root / split / rel / "image.dcm").resolve()
            # 四位医师掩膜的路径
            base = meta_path.parent
            mask_paths = [base / f"mask_r{k}.png" for k in range(1, 5)]
            if not dcm_path.exists():
                # 有些切片可能拷贝失败；跳过
                continue
            self.items.append({
                "dcm": str(dcm_path),
                "masks": [str(p) for p in mask_paths],
                "meta": meta,
                "label_dir": str(base),
            })

        if len(self.items) == 0:
            raise RuntimeError(f"[LIDCFromSplits] 找不到任何样本：{label_split_root}")

        # 让 DataLoader 每次复现性随机
        self.rand_state = None

    def __len__(self):
        return len(self.items)

    def _pick_one_mask(self, mask_paths):
        # 优先从“非空掩膜”的医师里等概率挑一个；都空则取 r1
        nonempty = []
        masks = []
        for p in mask_paths:
            a = np.array(Image.open(p).convert("L"))  # 0~255
            m = (a > 0).astype(np.float32)
            masks.append(m)
            if m.sum() > 0: nonempty.append(len(masks)-1)
        if self.return_all_masks:
            return np.stack(masks, 0)  # [4,H,W]
        idx = self.rng.choice(nonempty) if nonempty else 0
        return masks[idx]  # [H,W]

    def __getitem__(self, i):
        item = self.items[i]
        ds = pydicom.dcmread(item["dcm"])
        hu = _dcm_to_hu(ds)
        img01 = _apply_window_hu(hu, wl=-600.0, ww=1500.0)
        if self.img_size is not None:
            img01 = _resize_hw(img01, self.img_size)

        mask = self._pick_one_mask(item["masks"])
        if self.img_size is not None and mask.ndim == 2:
            mask = _resize_hw(mask, self.img_size)
        elif self.img_size is not None and mask.ndim == 3:
            mask = np.stack([_resize_hw(m, self.img_size) for m in mask], 0)

        # to torch
        img_t = torch.from_numpy(img01).float().unsqueeze(0)  # [1,H,W]
        if isinstance(mask, np.ndarray) and mask.ndim == 2:
            m_t = torch.from_numpy(mask).float()               # [H,W]
        else:
            m_t = torch.from_numpy(mask).float()               # [4,H,W]

        meta = {
            "series_uid": ds.SeriesInstanceUID,
            "sop_uid": ds.SOPInstanceUID,
            "pixel_spacing": tuple(map(float, getattr(ds, "PixelSpacing", [1.0,1.0]))),
            "slice_thickness": float(getattr(ds, "SliceThickness", 1.0)),
            "shape_hw": img_t.shape[-2:]
        }
        return img_t, m_t, meta
