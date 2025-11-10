# -*- coding: utf-8 -*-
# 评估：在 val 集上计算 GED^2（IoU 距离）并画 Fig.4 风格曲线
import os, re, math, numpy as np, torch
import matplotlib.pyplot as plt
from pathlib import Path
from probabilistic_unet import ProbabilisticUnet
from lidc_data.lidc_from_splits import LIDCFromSplits

# ========= 路径与模型配置（按你的训练保持一致） =========
IMAGES_ROOT = r"E:\workspace\LIDC-IDRI\per\dataset_images"
LABELS_ROOT = r"E:\workspace\LIDC-IDRI\per\dataset_labels"
CKPT_PATH   = r".\checkpoints\ckpt_crop128_e5.pth"   # 改成你要评估的权重
IMG_SIZE    = 192                                # 与训练一致
INPUT_CHANNELS = 1
NUM_CLASSES    = 1
NUM_FILTERS    = [32, 64, 128, 192]
LATENT_DIM     = 6
BETA           = 10.0
POS_WEIGHT     = 25.0
# 评估设置
MAX_VAL_IMGS   = None          # 只抽前 N 张；None=全部 val
M_LIST         = [1, 4, 8, 16] # 每张图的采样次数
THRESH         = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =======================================================

@torch.no_grad()
def iou_bin(a, b):
    """二值 IoU（空-空=1，仅一边空=0）"""
    if isinstance(a, torch.Tensor): a = a.cpu().numpy()
    if isinstance(b, torch.Tensor): b = b.cpu().numpy()
    A = (a > 0).astype(np.uint8); B = (b > 0).astype(np.uint8)
    inter = (A & B).sum(); union = (A | B).sum()
    return 1.0 if union == 0 else inter / float(union)

def dist_mask(a, b):
    return 1.0 - iou_bin(a, b)

def est_ged2(X_list, Y_list):
    """
    X_list: [x1..xm]  预测二值掩膜
    Y_list: [y1..yn]  医师二值掩膜
    返回：GED^2 的有限样本估计（基于 d=1-IoU）
    """
    m, n = len(X_list), len(Y_list)
    if m == 0 or n == 0: return np.nan
    # 2/(mn) * sum d(xi,yj)
    xy = 0.0
    for x in X_list:
        for y in Y_list:
            xy += dist_mask(x, y)
    term_xy = 2.0 * xy / (m * n)
    # 1/m^2 * sum d(xi,xj)
    xx = 0.0
    for i in range(m):
        for j in range(m):
            xx += dist_mask(X_list[i], X_list[j])
    term_xx = xx / (m * m)
    # 1/n^2 * sum d(yi,yj)
    yy = 0.0
    for i in range(n):
        for j in range(n):
            yy += dist_mask(Y_list[i], Y_list[j])
    term_yy = yy / (n * n)
    return term_xy - term_xx - term_yy

def main():
    # 1) 数据：val 集，返回四医师掩膜
    ds_val = LIDCFromSplits(IMAGES_ROOT, LABELS_ROOT, split="val",
                            img_size=IMG_SIZE, return_all_masks=True, seed=2026)
    indices = list(range(len(ds_val)))[: (MAX_VAL_IMGS or len(ds_val))]
    print(f"[Data] val samples used = {len(indices)}")

    # 2) 模型
    net = ProbabilisticUnet(
        input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES,
        num_filters=NUM_FILTERS, latent_dim=LATENT_DIM,
        no_convs_fcomb=4, beta=BETA, pos_weight=POS_WEIGHT
    ).to(DEVICE)
    print(f"[Model] load ckpt: {CKPT_PATH}")
    net.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE), strict=True)
    net.eval()

    # 3) 评估
    results = {m: [] for m in M_LIST}
    with torch.no_grad():
        for k, i in enumerate(indices, 1):
            img, mask4, _ = ds_val[i]            # img:[1,H,W], mask4:[4,H,W] float
            img = img.unsqueeze(0).to(DEVICE)    # [1,1,H,W]
            # 医师真值列表（保持0/1）
            Y_list = [(mask4[j].numpy() > 0.5).astype(np.uint8) for j in range(mask4.shape[0])]

            # 先走一遍 forward 准备特征；y 占位即可
            y_placeholder = torch.from_numpy((mask4.sum(0)>0).numpy()).unsqueeze(0).unsqueeze(0).to(DEVICE)
            net.forward(img, y_placeholder, training=True)

            for m in M_LIST:
                X_list = []
                for _ in range(m):
                    logits = net.sample(testing=True)              # [1,1,H,W]
                    prob   = torch.sigmoid(logits)[0,0].cpu().numpy()
                    X_list.append((prob > THRESH).astype(np.uint8))
                ged2 = est_ged2(X_list, Y_list)
                results[m].append(ged2)

            if k % 20 == 0:
                print(f"  processed {k}/{len(indices)}")

    # 4) 汇总 & 出图
    os.makedirs("plots", exist_ok=True)
    means, stds = [], []
    for m in M_LIST:
        arr = np.array(results[m], dtype=np.float32)
        means.append(np.nanmean(arr)); stds.append(np.nanstd(arr))

    plt.figure(figsize=(7,4))
    for i, m in enumerate(M_LIST):
        y = np.array(results[m], dtype=np.float32)
        x = np.full_like(y, fill_value=m, dtype=np.float32)
        plt.scatter(x, y, s=8, alpha=0.35)
        plt.plot([m], [means[i]], marker='D', markersize=8, color='red')
        plt.text(m+0.2, means[i], f"{means[i]:.3f}", va="center", fontsize=9, color="red")
    plt.title("GED$^2$ vs #samples on LIDC (val)")
    plt.xlabel("#samples m"); plt.ylabel(r"$\hat D^2_{\mathrm{GED}}$ (lower is better)")
    plt.grid(True, linestyle="--", alpha=0.4); plt.tight_layout()
    out_path = os.path.join("plots", "ged_from_splits.png")
    plt.savefig(out_path, dpi=150); plt.close()
    print("\nGED^2 summary (mean ± std):")
    for m, mu, sd in zip(M_LIST, means, stds):
        print(f"m={m:>2d} : {mu:.4f} ± {sd:.4f}")
    print("Saved plot to:", os.path.abspath(out_path))

if __name__ == "__main__":
    main()
