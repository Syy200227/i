import os, re, torch, numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
from probabilistic_unet import ProbabilisticUnet
from load_LIDC_data import LIDC_IDRI

# ======== 配置区（按你的训练配置修改）========
DATA_DIR = r"E:\workspace\LIDC-IDRI\pickle\\"   # 你的 LIDC pickle 路径
CKPT_DIR = r"E:\workspace\puent-25\stefanknegtProbabilistic-Unet-Pytorch\Probabilistic-Unet-Pytorch-master"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# 模型结构和损失权重（要与训练一致）
INPUT_CHANNELS = 1
NUM_CLASSES = 1
NUM_FILTERS = [32, 64, 128, 192]
LATENT_DIM = 6          # 如果你训练用的是2，这里改成2
BETA = 10.0
POS_WEIGHT = 25.0       # 如果在 probabilistic_unet.py 里启用了 pos_weight（建议保留）

# 评估参数
BATCH_SIZE = 4          # 验证时的 batch
VAL_SPLIT = 0.1
N_SAMPLES = 4           # 每幅图采样次数，用于 Dice-mean/best
MAX_VAL_IMAGES = None   # 限制验证图总数（None 表示全部，调小可加速）

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================================

def dice_score(pred_bin, gt):
    # pred_bin, gt: (B,1,H,W) 0/1
    eps = 1e-6
    inter = (pred_bin * gt).sum(dim=(1,2,3))
    denom = pred_bin.sum(dim=(1,2,3)) + gt.sum(dim=(1,2,3))
    return ((2*inter + eps) / (denom + eps)).cpu().numpy()

def build_val_loader():
    ds = LIDC_IDRI(dataset_location=DATA_DIR)
    indices = np.arange(len(ds))
    np.random.shuffle(indices)
    val_n = int(VAL_SPLIT * len(ds))
    val_idx = indices[:val_n] if val_n > 0 else indices
    if MAX_VAL_IMAGES is not None:
        val_idx = val_idx[:MAX_VAL_IMAGES]
    return DataLoader(ds, batch_size=BATCH_SIZE,
                      sampler=SubsetRandomSampler(val_idx),
                      num_workers=0, pin_memory=True), len(val_idx)

def build_model():
    # 需要你的 probabilistic_unet.py 里 __init__ 支持 pos_weight（如我之前给的）
    net = ProbabilisticUnet(
        input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES,
        num_filters=NUM_FILTERS, latent_dim=LATENT_DIM,
        no_convs_fcomb=4, beta=BETA, pos_weight=POS_WEIGHT
    ).to(DEVICE)
    net.eval()
    return net

def list_checkpoints():
    cks = []
    for fn in os.listdir(CKPT_DIR):
        m = re.match(r"ckpt_epoch(\d+)\.pth$", fn)
        if m: cks.append((int(m.group(1)), os.path.join(CKPT_DIR, fn)))
    cks.sort(key=lambda x: x[0])
    return cks

@torch.no_grad()
def evaluate_ckpt(net, ckpt_path, val_loader):
    state = torch.load(ckpt_path, map_location=DEVICE)
    net.load_state_dict(state, strict=True)
    net.eval()

    kl_vals, recon_means, elbos = [], [], []
    dice_means, dice_bests = [], []

    for img, msk, _ in val_loader:
        img = torch.clamp(img, 0.0, 1.0).to(DEVICE, non_blocking=True)          # (B,1,H,W)
        msk = (msk > 0.5).float().unsqueeze(1).to(DEVICE, non_blocking=True)    # (B,1,H,W)

        # 构建 posterior（验证也传 training=True）
        net.forward(img, msk, training=True)

        # KL + 重构BCE + ELBO（评估用采样KL，抖动更小）
        elbo = net.elbo(msk, analytic_kl=False)
        kl_vals.append(float(net.kl.detach().cpu().item()))
        recon_means.append(float(net.mean_reconstruction_loss.detach().cpu().item()))
        elbos.append(float((-elbo).detach().cpu().item()))

        # 采样 N 次 -> Dice mean & best
        dices = []
        for _ in range(N_SAMPLES):
            logits = net.sample(testing=True)                 # (B,1,H,W)
            prob = torch.sigmoid(logits)
            pred_bin = (prob > 0.5).float()
            dices.append(dice_score(pred_bin, msk))
        dices = np.stack(dices, axis=0)                       # (N,B)
        dice_means.append(dices.mean(axis=0))                 # (B,)
        dice_bests.append(dices.max(axis=0))                  # (B,)

    # 汇总到单个数
    kl_avg = float(np.mean(kl_vals)) if kl_vals else 0.0
    recon_avg = float(np.mean(recon_means)) if recon_means else 0.0
    elbo_avg = float(np.mean(elbos)) if elbos else 0.0
    dice_mean = float(np.mean(np.concatenate(dice_means))) if dice_means else 0.0
    dice_best = float(np.mean(np.concatenate(dice_bests))) if dice_bests else 0.0
    return kl_avg, recon_avg, elbo_avg, dice_mean, dice_best

def plot_curve(xs, ys, title, ylabel, save_path):
    plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    print("Device:", DEVICE)
    val_loader, nval = build_val_loader()
    print("Val images:", nval)
    net = build_model()

    ckpts = list_checkpoints()
    if not ckpts:
        print("No checkpoints found in", CKPT_DIR)
        return

    epochs, KLs, RECONs, ELBOs, D_MEANs, D_BESTs = [], [], [], [], [], []

    for ep, path in ckpts:
        print(f"Evaluating epoch {ep} -> {os.path.basename(path)}")
        kl, recon, elbo, dmean, dbest = evaluate_ckpt(net, path, val_loader)
        epochs.append(ep)
        KLs.append(kl); RECONs.append(recon); ELBOs.append(elbo)
        D_MEANs.append(dmean); D_BESTs.append(dbest)

    # 画图
    plot_curve(epochs, KLs, "KL over Epochs", "KL", os.path.join(PLOTS_DIR, "kl_curve.png"))
    plot_curve(epochs, RECONs, "Reconstruction (BCE mean) over Epochs", "BCE mean", os.path.join(PLOTS_DIR, "recon_curve.png"))
    plot_curve(epochs, ELBOs, "ELBO over Epochs", "ELBO (lower is better)", os.path.join(PLOTS_DIR, "elbo_curve.png"))
    plot_curve(epochs, D_MEANs, "Dice (mean sample) over Epochs", "Dice", os.path.join(PLOTS_DIR, "dice_mean_curve.png"))
    plot_curve(epochs, D_BESTs, "Dice (best of N samples) over Epochs", "Dice", os.path.join(PLOTS_DIR, "dice_best_curve.png"))

    print("Saved plots to:", os.path.abspath(PLOTS_DIR))

if __name__ == "__main__":
    main()
