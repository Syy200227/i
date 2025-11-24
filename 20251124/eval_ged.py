# eval_ged.py
# -*- coding: utf-8 -*-
"""
计算 Probabilistic U-Net + DEviS 的 GED^2 指标（Generalized Energy Distance）
距离采用 IoU 距离：d(A,B) = 1 - IoU(A,B)
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from probabilistic_unetv3 import ProbabilisticUnet
from data_pickle.dataset import LIDC_Split

# ===================================================
# 路径 & 参数
# ===================================================
DATA_DIR   = r"E:\workspace\puent-25\20251110\data_pickle"
CKPT_PATH  = r"./checkpoints/ckpt_epoch50.pth"
PLOTS_DIR  = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型结构参数（要与训练一致）
INPUT_CHANNELS = 1
NUM_CLASSES    = 2
NUM_FILTERS    = [32, 64, 128, 192]
LATENT_DIM     = 8
BETA           = 0.03
EDL_LAMBDA     = 1e-4
NUM_DOCTORS    = 4
THRESH         = 0.3
M_LIST         = [1, 2, 4, 8]

# ===================================================
# IoU 工具函数
# ===================================================
def iou_bin(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return inter / float(union)

def pairwise_mean_distance_iou_bin(masks: list) -> float:
    n = len(masks)
    if n < 2:
        return 0.0
    acc = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            acc += 1.0 - iou_bin(masks[i], masks[j])
    return (2.0 / (n * (n - 1))) * acc

# ===================================================
# 模型构建
# ===================================================
@torch.no_grad()
def build_model():
    net = ProbabilisticUnet(
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        num_filters=NUM_FILTERS,
        latent_dim=LATENT_DIM,
        no_convs_fcomb=4,
        beta=BETA,
        edl_lambda=EDL_LAMBDA,
        num_doctors=NUM_DOCTORS
    ).to(DEVICE)
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net

def build_val_loader():
    ds = LIDC_Split(os.path.join(DATA_DIR, "val_data.pkl"), rater="all")
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

# ===================================================
# GED² 计算
# ===================================================
@torch.no_grad()
def ged2_for_one_image(net, img_t, msk_all_np, M):
    """
    对单张图像计算 GED^2
    """
    img_t = img_t.to(DEVICE)
    B, _, H, W = img_t.shape
    R = msk_all_np.shape[0]

    # 随便选一个医生构建 posterior
    rid = 0
    ref_mask = torch.from_numpy(msk_all_np[rid:rid+1]).float().unsqueeze(0).to(DEVICE)
    doctor_id = torch.zeros((B, R, H, W), device=DEVICE)
    doctor_id[:, rid] = 1.0
    net.forward(img_t, ref_mask, doctor_id, training=True)

    # ---- 生成 M 个模型样本（二值）----
    samples_bin = []
    for _ in range(M):
        out = net.sample()
        probs = out["probs"][:, 1:2]
        pred_bin = (probs > THRESH).float().cpu().numpy()[0, 0]
        samples_bin.append(pred_bin)

    gt_masks = [(msk_all_np[i] > 0.5) for i in range(R)]

    Ey = np.mean([1 - iou_bin(s, y) for s in samples_bin for y in gt_masks])
    Ess = pairwise_mean_distance_iou_bin(samples_bin)
    Eyy = pairwise_mean_distance_iou_bin(gt_masks)

    ged2 = 2 * Ey - Ess - Eyy
    return max(0.0, ged2)

# ===================================================
# 主函数
# ===================================================
def main():
    print(f"==> Loading model from {CKPT_PATH}")
    net = build_model()
    loader = build_val_loader()

    ged_per_M = {M: [] for M in M_LIST}
    for n, (img, masks_all) in enumerate(loader):
        img = torch.clamp(img, 0, 1)
        msk_all_np = masks_all[0].numpy().astype(np.float32)
        for M in M_LIST:
            ged2 = ged2_for_one_image(net, img, msk_all_np, M)
            ged_per_M[M].append(ged2)
        if n % 20 == 0:
            print(f"processed {n} images...")

    means = []
    for M in M_LIST:
        arr = np.array(ged_per_M[M])
        mean_M = arr.mean()
        means.append(mean_M)
        print(f"M={M:>2d} GED^2(mean over images) = {mean_M:.4f}")
        plt.figure()
        plt.scatter(np.arange(len(arr)), arr, s=12)
        plt.title(f"GED^2 per image (M={M})")
        plt.xlabel("Image idx")
        plt.ylabel("GED^2")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"ged_scatter_M{M}.png"), dpi=150)
        plt.close()

    plt.figure()
    plt.plot(M_LIST, means, marker="o")
    plt.title("GED^2 vs #Samples")
    plt.xlabel("Number of model samples (M)")
    plt.ylabel("GED^2 (lower is better)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "ged_curve.png"), dpi=150)
    plt.close()
    print("Saved plots to:", os.path.abspath(PLOTS_DIR))


if __name__ == "__main__":
    main()
