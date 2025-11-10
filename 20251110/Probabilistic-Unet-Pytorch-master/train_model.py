# -*- coding: utf-8 -*-
"""
Probabilistic U-Net 训练（裁剪到128，提升POS_WEIGHT，KL warm-up）
目录结构对接：
  - E:\workspace\LIDC-IDRI\dataset_images\{train|val|test}\Patient\SeriesUID\SOPUID\image.dcm
  - E:\workspace\LIDC-IDRI\dataset_labels\{train|val|test}\Patient\SeriesUID\SOPUID\mask_r1.png..mask_r4.png + meta.json
"""

import os, time, random, numpy as np, torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from lidc_data.lidc_from_splits import LIDCFromSplits
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation
import kl_fix  # 你的 KL(Independent || Independent) 补丁

# ================= 路径与超参（按需改） =================
IMAGES_ROOT = r"E:\workspace\LIDC-IDRI\per\dataset_images"
LABELS_ROOT = r"E:\workspace\LIDC-IDRI\per\dataset_labels"

CROP_SIZE   = 128        # 裁剪后缩放到 128×128
CROP_PAD    = 8          # bbox 外扩像素
BATCH_TRAIN = 8
BATCH_VAL   = 6
EPOCHS      = 40        # 训练更久
LR          = 1e-4
POS_WEIGHT  = 60.0       # 提升正类权重，缓解小目标不平衡
BETA_MAX    = 10.0       # KL warm-up 目标 beta
WARMUP_EPOCHS = 25       # 用 25 个 epoch 从 0 → 10 线性升温
LATENT_DIM  = 6          # 如要进一步抑制假阳性，可降到 4 或 2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 可视化
VIS_EVERY_EPOCH = 1
VIS_N_CASES     = 4
VIS_N_SAMPLES   = 6
VIS_THRESH      = 0.5
VIS_OUT_DIR     = r".\vis"
# =======================================================


# ------------------- 辅助函数 -------------------
def _edge(m):
    m = (m > 0).astype(np.uint8)
    pad = np.pad(m, 1, mode='edge')
    e = (pad[1:-1,1:-1] != pad[:-2,1:-1]) | (pad[1:-1,1:-1] != pad[2:,1:-1]) | \
        (pad[1:-1,1:-1] != pad[1:-1,:-2]) | (pad[1:-1,1:-1] != pad[1:-1,2:])
    return e.astype(np.uint8)

def dice_score(pred_bin, gt):
    eps = 1e-6
    inter = (pred_bin * gt).sum(dim=(1,2,3))
    denom = pred_bin.sum(dim=(1,2,3)) + gt.sum(dim=(1,2,3))
    return ((2*inter + eps)/(denom + eps)).mean().item()

def linear_beta(epoch, warmup_epochs=WARMUP_EPOCHS, beta_max=BETA_MAX):
    """KL warm-up：从 0 线性升到 beta_max"""
    if epoch <= 0: return 0.0
    if epoch >= warmup_epochs: return beta_max
    return beta_max * (epoch / float(warmup_epochs))

def _resize_hw(arr, size):
    from PIL import Image
    mode = "F" if arr.dtype in (np.float32, np.float64) else "L"
    im = Image.fromarray(arr, mode=mode)
    im = im.resize((size, size), Image.BILINEAR)
    return np.array(im, dtype=arr.dtype)

def _bbox_from_union(union):
    ys, xs = np.where(union > 0)
    if len(xs) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())

def _square_crop(y1,x1,y2,x2,H,W,pad=CROP_PAD):
    y1 = max(0, y1-pad); x1 = max(0, x1-pad)
    y2 = min(H-1, y2+pad); x2 = min(W-1, x2+pad)
    h = y2 - y1 + 1; w = x2 - x1 + 1
    side = max(h, w)
    cy = (y1 + y2) // 2; cx = (x1 + x2) // 2
    y1 = max(0, cy - side//2); y2 = min(H-1, y1 + side - 1)
    x1 = max(0, cx - side//2); x2 = min(W-1, x1 + side - 1)
    return y1,x1,y2,x2
# ------------------------------------------------


# -------------- 裁剪版 Dataset 包装器 --------------
class LIDCCropped(Dataset):
    """
    包装 LIDCFromSplits，使其基于四医师并集的 bbox 做方形裁剪 → 缩放到 CROP_SIZE
    训练：随机挑一位非空医师作为监督（若都空，返回全零）
    验证：返回四医师并集作为评估标签（在外层做）
    """
    def __init__(self, base_ds: LIDCFromSplits, split="train", img_size=CROP_SIZE, pad=CROP_PAD):
        self.base = base_ds
        self.split = split
        self.img_size = img_size
        self.pad = pad
        self.rng = random.Random(2025 if split=="train" else 2026)

    def __len__(self): return len(self.base)

    def __getitem__(self, i):
        # 强制 base_ds 返回四医师，以便做 bbox
        img, masks_or4, meta = self.base[i]  # 如果 base_ds 是 return_all_masks=True，则 [4,H,W]
        if masks_or4.ndim == 2:
            # 万一 base_ds 是“随机挑一位”的版本，退化处理：
            masks4 = masks_or4.unsqueeze(0).repeat(4,1,1)
        else:
            masks4 = masks_or4  # [4,H,W]

        img_np = img.squeeze(0).numpy()       # [H,W]
        H, W = img_np.shape
        union = (masks4.sum(0) > 0.5).numpy().astype(np.uint8)

        bbox = _bbox_from_union(union)
        if bbox is None:
            # 没有结节（负样本）：中心方形裁剪
            side = min(H, W)
            y1,x1,y2,x2 = _square_crop( (H-side)//2, (W-side)//2, (H+side)//2-1, (W+side)//2-1, H, W, pad=0 )
        else:
            y1,x1,y2,x2 = _square_crop(*bbox, H, W, pad=self.pad)

        # 裁剪 & 缩放
        img_c = img_np[y1:y2+1, x1:x2+1]
        img_c = _resize_hw(img_c, self.img_size)
        masks_c = []
        from PIL import Image
        for k in range(4):
            m = masks4[k].numpy().astype(np.uint8)
            m = m[y1:y2+1, x1:x2+1]
            m = Image.fromarray(m*255, mode="L").resize((self.img_size,self.img_size), Image.NEAREST)
            masks_c.append( (np.array(m)>127).astype(np.float32) )
        masks_c = np.stack(masks_c, 0)  # [4,S,S]

        # 训练集：随机挑一个非空医师；验证集：返回四医师（外层会并集）
        if self.split == "train":
            nonempty = [idx for idx in range(4) if masks_c[idx].sum() > 0]
            idx = self.rng.choice(nonempty) if nonempty else 0
            lab = masks_c[idx]  # [S,S]
            lab_t = torch.from_numpy(lab).float().unsqueeze(0)  # [1,S,S]
            return torch.from_numpy(img_c).float().unsqueeze(0), lab_t, meta
        else:
            lab4_t = torch.from_numpy(masks_c).float()          # [4,S,S]
            return torch.from_numpy(img_c).float().unsqueeze(0), lab4_t, meta
# -------------------------------------------------


# ---------------- 可视化（与前一致） ----------------
def _to_uint8_gray(x01):
    return (np.clip(x01, 0, 1) * 255.0).astype(np.uint8)

def visualize_val_samples(net, ds_val_all, epoch, n_cases=4, n_samples=6,
                          out_dir="./vis", thresh=0.5, device=DEVICE):
    os.makedirs(out_dir, exist_ok=True)
    idxs = list(range(len(ds_val_all)))
    random.Random(2025 + epoch).shuffle(idxs)
    idxs = idxs[:n_cases]

    net.eval()
    for k, i in enumerate(idxs, start=1):
        img, mask4, meta = ds_val_all[i]         # img:[1,S,S], mask4:[4,S,S]
        img_np = img.squeeze(0).numpy()
        gt_union = (mask4.sum(0) > 0.5).float().numpy()

        img_t = img.unsqueeze(0).to(device)
        y_placeholder = torch.from_numpy(gt_union).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            net.forward(img_t, y_placeholder, training=True)
            probs = []
            for _ in range(n_samples):
                logits = net.sample(testing=True)          # [1,1,S,S]
                p = torch.sigmoid(logits).cpu().numpy()[0,0]
                probs.append(p)
            probs = np.stack(probs, 0)
            mean_prob = probs.mean(0)
            bins = (probs > thresh).astype(np.uint8)

        cols = 2 + n_samples + 1
        fig = plt.figure(figsize=(2.4*cols, 2.4))

        ax = fig.add_subplot(1, cols, 1)
        ax.imshow(_to_uint8_gray(img_np), cmap='gray'); ax.set_title("input"); ax.axis('off')

        ax = fig.add_subplot(1, cols, 2)
        base = np.zeros((*gt_union.shape,3), np.uint8) + 30
        ge = _edge(gt_union)
        base[ge>0] = [0,255,0]
        ax.imshow(base); ax.set_title("GT (union)"); ax.axis('off')

        for s in range(n_samples):
            ax = fig.add_subplot(1, cols, 3+s)
            lay = base.copy()
            pe = _edge(bins[s])
            lay[pe>0] = [255,0,0]
            ax.imshow(lay); ax.set_title(f"sample {s+1}"); ax.axis('off')

        ax = fig.add_subplot(1, cols, cols)
        ax.imshow((mean_prob*255).astype(np.uint8), cmap='gray'); ax.set_title("mean prob"); ax.axis('off')

        out_path = Path(out_dir) / f"ep{epoch:02d}_val_{k:02d}_idx{i}.png"
        plt.tight_layout(); plt.savefig(out_path, dpi=110); plt.close(fig)
        print(f"[VIS] saved -> {out_path}")
# ----------------------------------------------------


# ===================== 主流程 =====================
def build_loaders():
    # 基础数据集：显式返回四医师，用于裁剪
    base_tr = LIDCFromSplits(IMAGES_ROOT, LABELS_ROOT, split="train",
                             img_size=None, return_all_masks=True, seed=2025)
    base_va = LIDCFromSplits(IMAGES_ROOT, LABELS_ROOT, split="val",
                             img_size=None, return_all_masks=True, seed=2026)

    # 裁剪包装（bbox→square→resize到128）
    ds_tr = LIDCCropped(base_tr, split="train", img_size=CROP_SIZE, pad=CROP_PAD)
    ds_va = LIDCCropped(base_va, split="val",   img_size=CROP_SIZE, pad=CROP_PAD)

    print(f"[STATS] train={len(ds_tr)}  val={len(ds_va)}  (crop={CROP_SIZE})")
    tr = DataLoader(ds_tr, batch_size=BATCH_TRAIN, shuffle=True,  num_workers=0, pin_memory=True)
    va = DataLoader(ds_va, batch_size=BATCH_VAL,   shuffle=False, num_workers=0, pin_memory=True)
    return tr, va, ds_va

def main():
    train_loader, val_loader_all, ds_val_all = build_loaders()

    net = ProbabilisticUnet(
        input_channels=1, num_classes=1,
        num_filters=[32,64,128,192],
        latent_dim=LATENT_DIM, no_convs_fcomb=4,
        beta=0.0,                 # KL warm-up 起始 beta
        pos_weight=POS_WEIGHT
    ).to(DEVICE)

    # === AdamW + CosineAnnealingLR ===
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=5e-5,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=0.0
    )
    # ================================

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, EPOCHS+1):
        # ---- KL warm-up：更新 beta ----
        net.beta = float(linear_beta(epoch, WARMUP_EPOCHS, BETA_MAX))

        net.train()
        run_loss = run_kl = run_rec = 0.0
        n_batches = 0
        t0 = time.time()

        for img, label, _ in train_loader:
            img = img.to(DEVICE)                         # [B,1,128,128]
            msk = (label > 0.5).float().to(DEVICE)      # [B,1,128,128]

            net.forward(img, msk, training=True)
            elbo = net.elbo(msk, analytic_kl=False)
            reg = (l2_regularisation(net.posterior) +
                   l2_regularisation(net.prior) +
                   l2_regularisation(net.fcomb.layers))
            loss = -elbo + 1e-5 * reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()

            with torch.no_grad():
                kl  = float(getattr(net, "kl", torch.tensor(0.)).detach().cpu().item())
                rec = float(getattr(net, "mean_reconstruction_loss", torch.tensor(0.)).detach().cpu().item())
                run_loss += float(loss.detach().cpu().item())
                run_kl   += kl
                run_rec  += rec
                n_batches += 1

        # ---- 验证（四医师并集 + 多采样均值）----
        net.eval()
        v_loss = v_dice = 0.0
        v_batches = 0
        with torch.no_grad():
            for img, label4, _ in val_loader_all:
                img = img.to(DEVICE)
                msk = (label4.sum(dim=1, keepdim=True) > 0.5).float().to(DEVICE)
                probs = 0
                for _ in range(4):
                    net.forward(img, msk, training=True)
                    probs += torch.sigmoid(net.sample(testing=True))
                probs /= 4.0
                elbo = net.elbo(msk, analytic_kl=False)
                loss = -elbo
                pred = (probs > 0.5).float()
                d = dice_score(pred, msk)
                v_loss += float(loss.detach().cpu().item())
                v_dice += d
                v_batches += 1

        # 学习率调度步进（每个 epoch 结束一次）
        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        print(f"[Epoch {epoch:02d}] beta={net.beta:>4.1f}  lr={cur_lr:.2e}  "
              f"train_loss={(run_loss/max(n_batches,1)):.4f}  "
              f"train_KL={(run_kl/max(n_batches,1)):.4f}  "
              f"train_recon={(run_rec/max(n_batches,1)):.2f}  "
              f"val_loss={(v_loss/max(v_batches,1)):.4f}  "
              f"val_dice={(v_dice/max(v_batches,1)):.4f}  "
              f"time={time.time()-t0:.1f}s")

        torch.save(net.state_dict(), f"checkpoints/ckpt_crop{CROP_SIZE}_e{epoch}.pth")

        if epoch % VIS_EVERY_EPOCH == 0:
            visualize_val_samples(
                net, ds_val_all, epoch,
                n_cases=VIS_N_CASES,
                n_samples=VIS_N_SAMPLES,
                out_dir=VIS_OUT_DIR,
                thresh=VIS_THRESH,
                device=DEVICE
            )


if __name__ == "__main__":
    print("Registered KL(Independent || Independent)")
    main()
