# visualize_4gt.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from data_pickle.dataset import LIDC_Split  # 和 eval_ged.py 一样用这个

# 路径 & 配置（根据你的实际情况改）
DATA_DIR   = r"E:\workspace\puent-25\20251110\data_pickle"
VAL_PKL    = os.path.join(DATA_DIR, "val_data.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 1) 读验证集，拿一张图 + 4 个医生掩膜
    ds = LIDC_Split(VAL_PKL, rater="all")
    idx = 0  # 想看第几张就改这里
    sample = ds[idx]

    img = sample[0]       # (1,H,W) 或 (H,W)
    msk_all = sample[1]   # (4,H,W)

    if isinstance(img, torch.Tensor):
        img_np = img.cpu().numpy()
    else:
        img_np = np.array(img)

    if img_np.ndim == 2:
        img_np = img_np[None, ...]    # (1,H,W)

    ct_np = img_np[0]                 # (H,W)

    # 4 位医生掩膜 -> numpy (4,H,W)
    if isinstance(msk_all, torch.Tensor):
        msk_all_np = msk_all.cpu().numpy()
    else:
        msk_all_np = np.array(msk_all, dtype=np.float32)    # (4,H,W)

    gt_np_list = [(msk_all_np[i] > 0.5).astype(np.float32)
                  for i in range(msk_all_np.shape[0])]      # 4 个医生

    # 2) 画图：一行 5 张图：CT + GT1~4
    plt.figure(figsize=(10, 2.5))

    # CT
    plt.subplot(1, 5, 1)
    plt.imshow(ct_np, cmap="gray")
    plt.title("CT")
    plt.axis("off")

    # 四个医生 GT
    for i, gt in enumerate(gt_np_list):
        plt.subplot(1, 5, 2 + i)
        plt.imshow(gt, cmap="gray", vmin=0, vmax=1)
        plt.title(f"GT{i+1}")
        plt.axis("off")

    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    save_path = os.path.join("images", f"vis_4gt_only_idx{idx}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print("Saved figure to:", os.path.abspath(save_path))


if __name__ == "__main__":
    main()
