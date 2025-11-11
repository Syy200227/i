# dataset.py
# -*- coding: utf-8 -*-
"""
LIDC-IDRI 三折数据加载 & DataLoader 工具
--------------------------------------
· 兼容 create_data_split.py 生成的 train/val/test_data.pkl
· 兼容旧版 lidcSeg.pt (tuple: images, masks)
· 支持随机/指定/全部医生掩膜
· 自带 RandomFlip 数据增强 & 一键构造 train/val/test DataLoader
"""
import os, random, joblib, torch
from typing import Callable, Optional, Tuple
import numpy as np
from torch.utils.data import Dataset, DataLoader


# ---------- 可选同步增强 ----------
class RandomFlip:
    """50% 概率水平翻转，保持 image 与 mask 对齐"""
    def __call__(self, img: np.ndarray, msk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:            # img:(C,H,W)  msk:(H,W) or (4,H,W)
            img = img[:, :, ::-1]
            msk = msk[..., ::-1]
        return img.copy(), msk.copy()


# ---------- 数据集 ----------
class LIDC_Split(Dataset):
    def __init__(
        self,
        file_path: str,
        rater: str = "random",          # "random"/"all"/"0"-"3" 或 int 0~3
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.rater = str(rater)
        self.transform = transform

        if file_path.endswith(".pkl"):
            data = joblib.load(file_path)
            self.images = [v["image"].astype(np.float32) for v in data.values()]
            self.masks  = [np.stack(v["masks"]).astype(np.float32)
                           for v in data.values()]          # (4,H,W)
        elif file_path.endswith(".pt"):
            img_list, msk_list = torch.load(file_path)
            self.images = [x.numpy().astype(np.float32) for x in img_list]
            self.masks  = [x.numpy().astype(np.float32) for x in msk_list]
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        # 裁剪到 [0,1]
        self.images = [np.clip(im, 0.0, 1.0) for im in self.images]
        self.masks  = [np.clip(msk, 0.0, 1.0) for msk in self.masks]

    # -------- private ---------
    def _select_mask(self, msk: np.ndarray) -> np.ndarray:
        if self.rater == "random":
            return msk[random.randint(0, 3)]        # (H,W)
        elif self.rater == "all":
            return msk                              # (4,H,W)
        else:                                       # 指定医生
            return msk[int(self.rater)]

    # -------- Dataset API -----
    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = np.expand_dims(self.images[idx], 0)   # (1,H,W)
        msk = self._select_mask(self.masks[idx])    # (H,W) or (4,H,W)

        if self.transform is not None:
            img, msk = self.transform(img, msk)     # 同步几何增强

        return torch.from_numpy(img).float(), torch.from_numpy(msk).float()


# ---------- 一键构造 DataLoader ----------
def build_loaders(
    data_dir: str,
    batch_train: int = 8,
    batch_val: int = 2,
    rater: str = "random",
    augment: bool = True,
    num_workers: int = 0,
):
    """
    返回 train_loader, val_loader, test_loader

    Parameters
    ----------
    data_dir     : 目录下需含 train_data.pkl / val_data.pkl / test_data.pkl
    batch_train  : 训练集 batch_size
    batch_val    : 验证 / 测试 batch_size
    rater        : 同 LIDC_Split.rater
    augment      : 训练集是否做 RandomFlip
    """
    train_ds = LIDC_Split(
        os.path.join(data_dir, "train_data.pkl"),
        rater=rater,
        transform=RandomFlip() if augment else None,
    )
    val_ds   = LIDC_Split(os.path.join(data_dir, "val_data.pkl"),   rater=rater)
    test_ds  = LIDC_Split(os.path.join(data_dir, "test_data.pkl"),  rater=rater)

    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_val, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_val, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# ---------- 简易自测 ----------
if __name__ == "__main__":
    # 把此路径改成你的实际目录
    DIR = r"E:\workspace\puent-25\20251110\data_pickle"

    tr, va, te = build_loaders(DIR, batch_train=4, batch_val=2)
    print("train batches:", len(tr), "val batches:", len(va), "test batches:", len(te))
    img, msk = next(iter(tr))
    print("sample image shape:", img.shape, "mask shape:", msk.shape)
