# import torch, numpy as np, time
# from torch.utils.data import DataLoader
# from torch.utils.data.sampler import SubsetRandomSampler
# from load_LIDC_data import LIDC_IDRI
# from probabilistic_unet import ProbabilisticUnet
# from utils import l2_regularisation
# import kl_fix  # KL修复，允许analytic_kl=True时使用
#
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.autograd.set_detect_anomaly(True)
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # 数据
# dataset = LIDC_IDRI(dataset_location=r'E:\workspace\LIDC-IDRI\pickle\\')
# dataset_size = len(dataset)
# indices = list(range(dataset_size))
# split = int(np.floor(0.1 * dataset_size))
# np.random.shuffle(indices)
# train_indices, test_indices = indices[split:], indices[:split]
# train_sampler = SubsetRandomSampler(train_indices)
# test_sampler  = SubsetRandomSampler(test_indices)
#
# # —— 更小的 batch 保守起步 ——
# train_loader = DataLoader(dataset, batch_size=2, sampler=train_sampler, num_workers=0)
# test_loader  = DataLoader(dataset, batch_size=1, sampler=test_sampler,  num_workers=0)
#
# print("Number of training/test patches:", (len(train_indices), len(test_indices)))
# print("Device:", device)
#
# # 模型
# net = ProbabilisticUnet(input_channels=1, num_classes=1,
#                         num_filters=[32,64,128,192], latent_dim=2,
#                         no_convs_fcomb=4, beta=10.0).to(device)
# net.train()
#
# optimizer = torch.optim.Adam(net.parameters(), lr=5e-5, weight_decay=0)
# epochs = 3
# log_interval = 50
#
# def clean_batch(patch, mask, step):
#     # 基本清洗
#     if not torch.isfinite(patch).all():
#         print(f"[bad patch] step={step}")
#         patch = torch.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)
#     if not torch.isfinite(mask).all():
#         print(f"[bad mask] step={step}")
#         mask = torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
#
#     patch = torch.clamp(patch, 0.0, 1.0)
#     mask  = (mask > 0.5).float()
#
#     # 再次确认
#     assert torch.isfinite(patch).all(), "patch has NaN/Inf after clean"
#     assert torch.isfinite(mask).all(),  "mask has NaN/Inf after clean"
#     return patch, mask
#
# def eval_once():
#     net.eval()
#     with torch.no_grad():
#         (p, m, _) = next(iter(test_loader))
#         p, m = clean_batch(p, m, step=-1)
#         p = p.to(device)
#         m = m.to(device).unsqueeze(1)
#
#         # 关键：要重建 posterior，必须 training=True
#         net.forward(p, m, training=True)
#
#         pred = torch.sigmoid(net.sample(testing=True))
#         elbo = net.elbo(m, analytic_kl=False)   # 这里就不会尺寸冲突
#         return float((-elbo).item()), float(pred.mean().item())
#     net.train()
#
#
# t0 = time.time()
# global_step = 0
# for epoch in range(1, epochs+1):
#     for step, (patch, mask, _) in enumerate(train_loader):
#         global_step += 1
#         patch, mask = clean_batch(patch, mask, step)
#         patch = patch.to(device)
#         mask  = mask.to(device).unsqueeze(1)
#
#         # 前向
#         net.forward(patch, mask, training=True)
#         # —— 训练时统一用 采样KL，更稳 ——
#         elbo = net.elbo(mask, analytic_kl=False)
#
#         # 正则
#         reg_loss = (l2_regularisation(net.posterior) +
#                     l2_regularisation(net.prior) +
#                     l2_regularisation(net.fcomb.layers))
#         loss = -elbo + 1e-5 * reg_loss
#
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
#         optimizer.step()
#
#         if global_step % log_interval == 0:
#             with torch.no_grad():
#                 kl_val  = float(net.kl.detach().cpu().item()) if hasattr(net, 'kl') else float('nan')
#                 recon   = float(getattr(net, 'mean_reconstruction_loss', torch.tensor(0.)).detach().cpu().item())
#                 train_l = float(loss.detach().cpu().item())
#             eval_l, pred_mean = eval_once()
#             dt = time.time() - t0
#             print(f"[{epoch}|{global_step}] loss={train_l:.4f} kl={kl_val:.4f} recon_mean={recon:.2f} "
#                   f"eval_loss={eval_l:.4f} pred_mean={pred_mean:.3f} time={dt:.1f}s")
#
# print("Done.")
##----------------------------------------------------给出重构jpg
import os, time, math, torch, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
from load_LIDC_data import LIDC_IDRI
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation
import kl_fix  # 解析式KL可用
from datetime import datetime, timezone, timedelta

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- 数据 ----------------
dataset = LIDC_IDRI(dataset_location=r'E:\workspace\LIDC-IDRI\pickle\\')
indices = np.random.permutation(len(dataset))
val_split = int(0.1 * len(dataset))
train_idx, val_idx = indices[val_split:], indices[:val_split]

# 建议 batch=4（若显存不够就用2），也支持梯度累积模拟大batch
batch_size = 8
accum_steps = 1  # ★梯度累积：2×4≈等效batch 8（若不需要，设为1）
train_loader = DataLoader(dataset, batch_size=batch_size,
                          sampler=SubsetRandomSampler(train_idx),
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(dataset, batch_size=1,
                          sampler=SubsetRandomSampler(val_idx),
                          num_workers=0, pin_memory=True)

print("train / val:", len(train_idx), len(val_idx), "device:", device)

# ---------------- 模型（更稳配置） ----------------
net = ProbabilisticUnet(
    input_channels=1, num_classes=1,
    num_filters=[32,64,128,192],
    latent_dim=6,               # ★论文常用，后验更有表达力
    no_convs_fcomb=4,
    beta=10.0,                  # KL 系数上限
    pos_weight=25.0             # ★BCE正例权重（可微调 10~50）
).to(device)

# AdamW 更稳；小学习率 + 有权重衰减
optimizer = torch.optim.AdamW(
    net.parameters(), lr=5e-5, weight_decay=1e-4,
    betas=(0.9, 0.98), eps=1e-8
)
# 可选：余弦退火或 Plateau
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

# ---------------- 工具函数 ----------------
def clean_batch(img, msk):
    img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    msk = torch.nan_to_num(msk, nan=0.0, posinf=1.0, neginf=0.0)
    img = torch.clamp(img, 0.0, 1.0)
    msk = (msk > 0.5).float()
    return img, msk

@torch.no_grad()
def evaluate_once():
    net.eval()
    img, msk, _ = next(iter(val_loader))
    img, msk = clean_batch(img, msk)
    img = img.to(device)
    msk = msk.to(device).unsqueeze(1)
    net.forward(img, msk, training=True)       # 构建 posterior
    pred = torch.sigmoid(net.sample(testing=True))
    # 评估阶段仍用采样KL，抖动更小
    elbo = net.elbo(msk, analytic_kl=False)
    return float(-elbo.item()), float(pred.mean().item()), img.detach().cpu(), msk.detach().cpu(), pred.detach().cpu()

# ---------------- 训练循环（KL预热 + 累积） ----------------
epochs = 30
log_int = 50
step_g = 0
os.makedirs('images', exist_ok=True)

# KL 预热：前 warmup_steps 线性从 0 → beta（建议覆盖 3~5 个 epoch）
warmup_epochs = 5
iters_per_epoch = math.ceil(len(train_idx) / batch_size)
warmup_steps = warmup_epochs * iters_per_epoch

net.train()
t0 = time.time()
optimizer.zero_grad(set_to_none=True)

for ep in range(1, epochs + 1):
    for step, (img, msk, _) in enumerate(train_loader, 1):
        step_g += 1

        # --- KL anneal ---
        if step_g <= warmup_steps:
            net.beta_t = net.beta * (step_g / float(warmup_steps))
        else:
            net.beta_t = net.beta

        img, msk = clean_batch(img, msk)
        img = img.to(device, non_blocking=True)
        msk = msk.to(device, non_blocking=True).unsqueeze(1)

        net.forward(img, msk, training=True)
        # 训练期：采样 KL（稳定）；收敛后可切成 analytic_kl=True
        elbo = net.elbo(msk, analytic_kl=False)
        reg = (l2_regularisation(net.posterior) +
               l2_regularisation(net.prior) +
               l2_regularisation(net.fcomb.layers))
        loss = (-elbo + 1e-5 * reg) / accum_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)  # ★更紧的裁剪

        if step_g % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step_g % log_int == 0:
            net.eval()
            val_loss, pred_mean, img_v, msk_v, pred_v = evaluate_once()
            kl_val = float(getattr(net, 'kl', torch.tensor(0.)).detach().cpu())
            recon_mean = float(getattr(net, 'mean_reconstruction_loss', torch.tensor(0.)).detach().cpu())
            dt = time.time() - t0
            print(f"[E{ep}|{step_g}] lr={optimizer.param_groups[0]['lr']:.2e} "
                  f"beta_t={net.beta_t:.2f} train_loss={(loss.item()*accum_steps):.3f} "
                  f"KL={kl_val:.3f} recon_mean={recon_mean:.3f} "
                  f"val_loss={val_loss:.3f} pred_mean={pred_mean:.3f} t={dt:.1f}s")

            # 保存 JPG 可视化
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(img_v[0, 0], cmap='gray'); axes[0].set_title('CT'); axes[0].axis('off')
            axes[1].imshow(msk_v[0, 0], cmap='gray'); axes[1].set_title('GT'); axes[1].axis('off')
            axes[2].imshow(pred_v[0, 0], cmap='gray'); axes[2].set_title('Pred'); axes[2].axis('off')
            plt.tight_layout()
            plt.savefig(f"images/step_{step_g}.jpg", dpi=150); plt.close()
            net.train()

    scheduler.step()  # 学习率调度

    # 每个 epoch 保存一次模型
    ckpt = f"ckpt_epoch{ep}.pth"
    torch.save(net.state_dict(), ckpt)
    print(f"Checkpoint saved: {ckpt}")

print("Training finished")

#查看耗时
print("Training finished at", datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %Z%z"))
