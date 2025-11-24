# evaluate.py —— 完全适配 probabilistic_unetv3，无 analytic_kl，无 mean_reconstruction_loss
# -*- coding: utf-8 -*-
import os, re, torch, numpy as np
import matplotlib.pyplot as plt
from data_pickle.dataset import build_loaders
from probabilistic_unetv3 import ProbabilisticUnet
from datetime import datetime, timezone, timedelta
# ======== 路径配置 ========
DATA_DIR  = r"E:\workspace\puent-25\20251110\data_pickle"
CKPT_DIR  = r"./checkpoints"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ======== 模型配置（与训练保持一致）========
INPUT_CHANNELS = 1
NUM_CLASSES    = 2
NUM_FILTERS    = [32, 64, 128, 192]
LATENT_DIM     = 8
BETA           = 0.03
EDL_LAMBDA     = 1e-4
NUM_DOCTORS    = 4

BATCH_SIZE     = 4
N_SAMPLES      = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dice_score(pred_bin, gt):
    eps = 1e-6
    inter = (pred_bin * gt).sum(dim=(1,2,3))
    denom = pred_bin.sum(dim=(1,2,3)) + gt.sum(dim=(1,2,3))
    return ((2*inter + eps) / (denom + eps)).cpu().numpy()   # shape = (batch,)


def build_val_loader():
    _, val_loader, _ = build_loaders(
        DATA_DIR, batch_train=BATCH_SIZE, batch_val=BATCH_SIZE,
        rater="random", augment=False, num_workers=0
    )
    return val_loader


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
    net.eval()
    return net


def list_checkpoints():
    cks = []
    for fn in os.listdir(CKPT_DIR):
        m = re.match(r"ckpt_epoch(\d+)\.pth$", fn)
        if m:
            cks.append((int(m.group(1)), os.path.join(CKPT_DIR, fn)))
    cks.sort(key=lambda x: x[0])
    return cks


@torch.no_grad()
def evaluate_ckpt(net, ckpt_path, val_loader):
    state = torch.load(ckpt_path, map_location=DEVICE)
    net.load_state_dict(state, strict=True)
    net.eval()

    KLs, RECONs, ELBOs = [], [], []
    D_MEANs, D_BESTs = [], []   # store scalars

    for img, msk in val_loader:
        img = torch.clamp(img, 0, 1).to(DEVICE)
        msk = (msk > 0.5).float().unsqueeze(1).to(DEVICE)

        # doctor_id 占位
        B, _, H, W = img.shape
        doctor_id = torch.zeros((B, NUM_DOCTORS, H, W), device=DEVICE)

        # --- forward posterior ---
        net.forward(img, msk, doctor_id, training=True)

        # --- ELBO ---
        elbo = net.elbo(msk)

        KLs.append(float(net.kl))
        RECONs.append(float(net.mean_recon))
        ELBOs.append(float(-elbo))

        # --- 多样化 Dice ---
        batch_dices = []
        for _ in range(N_SAMPLES):
            out = net.sample()
            prob_fg = out["probs"][:, 1:2]
            pred_bin = (prob_fg > 0.3).float()

            # 返回 shape=(batch,)
            d = dice_score(pred_bin, msk)
            batch_dices.append(d)

        batch_dices = np.stack(batch_dices, axis=0)
        # batch_dices.shape = (N_samples, batch)

        D_MEANs.append(float(batch_dices.mean()))   # ★ 转成标量
        D_BESTs.append(float(batch_dices.max()))    # ★ 转成标量

    return (
        float(np.mean(KLs)),
        float(np.mean(RECONs)),
        float(np.mean(ELBOs)),
        float(np.mean(D_MEANs)),
        float(np.mean(D_BESTs)),
    )


def plot_curve(xs, ys, title, ylabel, save_path):
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    print("Device:", DEVICE)
    val_loader = build_val_loader()
    net = build_model()

    ckpts = list_checkpoints()
    if not ckpts:
        print("No checkpoints found!")
        return

    epochs, KLs, RECONs, ELBOs, D_MEANs, D_BESTs = [], [], [], [], [], []

    for ep, path in ckpts:
        print(f"Evaluating epoch {ep}")
        kl, recon, elbo, dmean, dbest = evaluate_ckpt(net, path, val_loader)

        epochs.append(ep)
        KLs.append(kl)
        RECONs.append(recon)
        ELBOs.append(elbo)
        D_MEANs.append(dmean)
        D_BESTs.append(dbest)

    plot_curve(epochs, KLs,    "KL over Epochs",   "KL",   os.path.join(PLOTS_DIR,"kl_curve.png"))
    plot_curve(epochs, RECONs, "Reconstruction",   "Recon",os.path.join(PLOTS_DIR,"recon_curve.png"))
    plot_curve(epochs, ELBOs,  "ELBO",             "ELBO", os.path.join(PLOTS_DIR,"elbo_curve.png"))
    plot_curve(epochs, D_MEANs,"Dice (mean)",      "Dice", os.path.join(PLOTS_DIR,"dice_mean_curve.png"))
    plot_curve(epochs, D_BESTs,"Dice (best of N)", "Dice", os.path.join(PLOTS_DIR,"dice_best_curve.png"))

    print("Saved plots to:", os.path.abspath(PLOTS_DIR))


if __name__ == "__main__":
    main()

print("Training finished at", datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %Z%z"))