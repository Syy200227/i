# train_model_doctor.py
# -*- coding: utf-8 -*-
import os, time, torch
import matplotlib.pyplot as plt
from data_pickle.dataset import build_loaders

from probabilistic_unetv3 import ProbabilisticUnet
from utils import l2_regularisation
import subprocess
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Data ----------------
DATA_DIR = r"E:\workspace\puent-25\20251110\data_pickle"

train_loader, val_loader, _ = build_loaders(
    DATA_DIR,
    batch_train=6,
    batch_val=2,
    rater="all",           # 显示传入
    augment=True,
    num_workers=0
)

# ---------------- Model ----------------
net = ProbabilisticUnet(
    input_channels=1,
    num_classes=2,
    latent_dim=8,
    beta=0.03,
    edl_lambda=1e-4,
    num_doctors=4,
    dice_weight=0.5
).to(DEVICE)

optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

# ---------------- Hyperparams ----------------
EPOCHS = 300
warmup_epochs = 30
iters_per_epoch = len(train_loader)
warmup_steps = warmup_epochs * iters_per_epoch

EDL_LAMBDA_MAX = 1e-4

THRESH = 0.3
os.makedirs("images", exist_ok=True)
VIS_INT = 200
os.makedirs("checkpoints", exist_ok=True)

# --------------------------------------------------------
def clean(img):
    return torch.clamp(torch.nan_to_num(img), 0, 1)

# --------------------------------------------------------
for ep in range(1, EPOCHS + 1):

    # --- EDL KL anneal ---
    net.edl_lambda = EDL_LAMBDA_MAX * min(1.0, ep / (EPOCHS * 0.25))

    for step, (img, masks_all) in enumerate(train_loader):

        step_global = (ep - 1) * iters_per_epoch + step

        # β warmup
        if step_global <= warmup_steps:
            net.beta_t = net.beta * (step_global / warmup_steps)
        else:
            net.beta_t = net.beta

        img = clean(img).to(DEVICE)
        B, R, H, W = masks_all.shape
        masks_all = (masks_all > 0.5).float().to(DEVICE)

        # randomly choose doctor
        rid = torch.randint(0, R, (1,)).item()
        mask = masks_all[:, rid:rid+1]

        # build doctor one-hot
        doctor_id = torch.zeros((B, R, H, W), device=DEVICE)
        doctor_id[:, rid] = 1.0

        # forward
        net.forward(img, mask, doctor_id, training=True)
        elbo = net.elbo(mask)

        reg = l2_regularisation(net.prior) + \
              l2_regularisation(net.posterior) + \
              l2_regularisation(net.fcomb.layers)

        loss = -elbo + 1e-5 * reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        # ====== 可视化：CT + 4 GT（第一行），Binary + ProbFG（第二行） ======
        if step_global % VIS_INT == 0:
            net.eval()
            with torch.no_grad():
                # 只拿 batch 中第 0 个样本做展示
                img_vis = img[0:1]                  # (1,1,H,W)
                masks_vis = masks_all[0].cpu().numpy()  # (R,H,W)，这里 R=4

                # 构造与当前训练相同医生 id（rid）对应的 one-hot，batch 大小=1
                doctor_id_vis = torch.zeros((1, masks_vis.shape[0],
                                             img_vis.shape[2], img_vis.shape[3]),
                                             device=DEVICE)
                doctor_id_vis[:, rid] = 1.0

                # forward 一次 + sample 一次
                net.forward(img_vis, mask[0:1], doctor_id_vis, training=True)
                out_vis = net.sample()
                prob_fg = out_vis["probs"][:, 1:2]          # (1,1,H,W)
                prob_fg_np = prob_fg.cpu().numpy()[0, 0]
                bin_fg_np = (prob_fg_np > THRESH).astype(float)

                ct_np = img_vis.cpu().numpy()[0, 0]
                gt_np_list = [(masks_vis[i] > 0.5).astype(float)
                              for i in range(masks_vis.shape[0])]  # 4 个医生

                # ---- 画图：第一行 CT+4GT，第二行 Binary+ProbFG ----
                fig = plt.figure(figsize=(10, 5))
                gs = fig.add_gridspec(2, 5)  # 2 行 5 列

                # 第一行：CT + GT1~4
                ax_ct  = fig.add_subplot(gs[0, 0])
                ax_gt1 = fig.add_subplot(gs[0, 1])
                ax_gt2 = fig.add_subplot(gs[0, 2])
                ax_gt3 = fig.add_subplot(gs[0, 3])
                ax_gt4 = fig.add_subplot(gs[0, 4])

                ax_ct.imshow(ct_np, cmap="gray")
                ax_ct.set_title("CT")
                ax_ct.axis("off")

                gt_axes = [ax_gt1, ax_gt2, ax_gt3, ax_gt4]
                for i, (ax, gt_img) in enumerate(zip(gt_axes, gt_np_list)):
                    ax.imshow(gt_img, cmap="gray", vmin=0, vmax=1)
                    ax.set_title(f"GT{i+1}")
                    ax.axis("off")

                # 第二行：Binary 和 ProbFG
                ax_bin  = fig.add_subplot(gs[1, 1:3])  # 占第2行 第2~3列
                ax_prob = fig.add_subplot(gs[1, 3:5])  # 占第2行 第4~5列

                ax_bin.imshow(bin_fg_np, cmap="gray", vmin=0, vmax=1)
                ax_bin.set_title(f"Binary (> {THRESH})")
                ax_bin.axis("off")

                ax_prob.imshow(prob_fg_np, cmap="gray")
                ax_prob.set_title("ProbFG")
                ax_prob.axis("off")

                plt.tight_layout()
                save_path = os.path.join(
                    "images", f"ep{ep}_step{step_global}_vis4gt.png"
                )
                plt.savefig(save_path, dpi=200)
                plt.close()
                print("Saved vis:", os.path.abspath(save_path))

            net.train()
        # ====== 结束可视化块 ======

        # logging
        if step % 50 == 0:
            print(f"[E{ep}|{step}] loss={loss.item():.3f} KL={net.kl:.3f} "
                  f"recon={net.mean_recon:.3f} beta_t={net.beta_t:.3f}")


    scheduler.step()

    torch.save(net.state_dict(), f"checkpoints/ckpt_epoch{ep}.pth")
    print(f"Saved checkpoint: epoch {ep}")

print("Training done.")
print("v4运行完成，现在自动执行 evaluate")

subprocess.run(["python", "evaluate.py"])