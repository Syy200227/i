# eval_ged.py —— Fig.4 风格的 GED^2 评估与绘图
import os, re, math
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler, DataLoader
from probabilistic_unet import ProbabilisticUnet
from load_LIDC_data import LIDC_IDRI

# ========= 配置（改成你的路径/结构） =========
DATA_DIR   = r"E:\workspace\LIDC-IDRI\pickle\\"
CKPT_DIR   = r"E:\workspace\puent-25\stefanknegtProbabilistic-Unet-Pytorch\Probabilistic-Unet-Pytorch-master"
CKPT_NAME  = "ckpt_epoch30.pth"   # 选定一个模型（例如最终的 epoch）
CKPT_PATH  = os.path.join(CKPT_DIR, CKPT_NAME)

# 模型结构需与训练一致
INPUT_CHANNELS = 1
NUM_CLASSES    = 1
NUM_FILTERS    = [32,64,128,192]
LATENT_DIM     = 6       # 若训练时是2，请改成2
BETA           = 10.0    # 不影响GED，但模型结构需要
POS_WEIGHT     = 25.0    # 若你的 probabilistic_unet 用到了 pos_weight 就留着

# 评估设置
VAL_SPLIT      = 0.1     # 从全数据划出验证集比例
MAX_VAL_IMGS   = None    # 限制用于评估的最大张数，加速用；None=全部
M_LIST         = [1,4,8,16]  # 采样次数
THRESH         = 0.5     # 概率阈值 -> 二值掩膜

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================================

def list_ckpts(ckpt_dir):
    out = []
    for fn in os.listdir(ckpt_dir):
        m = re.match(r"ckpt_epoch(\d+)\.pth$", fn)
        if m: out.append((int(m.group(1)), os.path.join(ckpt_dir, fn)))
    out.sort(key=lambda x: x[0])
    return out

@torch.no_grad()
def iou_bin(a, b):
    """
    a, b: np.ndarray / torch.Tensor of shape (H,W), values in {0,1}
    处理空掩膜：都空 -> IoU=1；仅一边空 -> 0
    """
    if isinstance(a, torch.Tensor): a = a.cpu().numpy()
    if isinstance(b, torch.Tensor): b = b.cpu().numpy()
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = (a & b).sum()
    union = (a | b).sum()
    if union == 0:
        return 1.0  # 两者皆空
    return inter / float(union)

def dist_mask(a, b):
    """ d(A,B) = 1 - IoU(A,B) """
    return 1.0 - iou_bin(a, b)

def est_ged2(X_list, Y_list):
    """
    X_list: 预测掩膜列表 [x1,...,xm]，每个(H,W) 0/1
    Y_list: 医生真值列表 [y1,...,yn]，每个(H,W) 0/1
    返回：GED^2 的有限样本估计
    """
    m = len(X_list)
    n = len(Y_list)
    if m == 0 or n == 0:
        return np.nan

    # 第一项：2/(mn) * sum d(x_i, y_j)
    xy = 0.0
    for x in X_list:
        for y in Y_list:
            xy += dist_mask(x, y)
    term_xy = 2.0 * xy / (m * n)

    # 第二项：1/m^2 * sum d(x_i, x_i')
    xx = 0.0
    for i in range(m):
        for j in range(m):
            xx += dist_mask(X_list[i], X_list[j])
    term_xx = xx / (m * m)

    # 第三项：1/n^2 * sum d(y_j, y_j')
    yy = 0.0
    for i in range(n):
        for j in range(n):
            yy += dist_mask(Y_list[i], Y_list[j])
    term_yy = yy / (n * n)

    return term_xy - term_xx - term_yy

def build_val_indices(ds_len, val_split, max_imgs=None):
    idx = np.random.permutation(ds_len)
    val_n = int(val_split * ds_len)
    val_idx = idx[:val_n] if val_n > 0 else idx
    if max_imgs is not None:
        val_idx = val_idx[:max_imgs]
    return val_idx

def main():
    print("Device:", DEVICE)
    # 载入数据集（我们直接访问 dataset.images / dataset.labels 来拿“4位医生的掩膜”）
    ds = LIDC_IDRI(dataset_location=DATA_DIR)
    val_idx = build_val_indices(len(ds), VAL_SPLIT, MAX_VAL_IMGS)
    print("Val images:", len(val_idx))

    # 构建模型并加载权重
    net = ProbabilisticUnet(
        input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES,
        num_filters=NUM_FILTERS, latent_dim=LATENT_DIM,
        no_convs_fcomb=4, beta=BETA, pos_weight=POS_WEIGHT
    ).to(DEVICE)
    print("Loading ckpt:", CKPT_PATH)
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    net.load_state_dict(state, strict=True)
    net.eval()

    # 为每个 m 计算所有验证图的 GED^2
    results = {m: [] for m in M_LIST}

    for k, idx in enumerate(val_idx, 1):
        # 取图像与“全部医生掩膜”
        img_np = ds.images[idx]            # (H,W), np.float32, [0,1]
        ys_np  = ds.labels[idx]            # (4,H,W), np.float32, [0,1]

        # 整理成 torch tensor
        img = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,H,W)
        # Y 列表（只保留有像素/即便全0也保留以用于“空掩膜规则”）
        Y_list = [ys_np[i] for i in range(ys_np.shape[0])]  # 每个(H,W)

        # 先把 UNet 特征+先验准备好（posterior 不需要）
        net.forward(img, segm=None, training=False)

        for m in M_LIST:
            X_list = []
            with torch.no_grad():
                for _ in range(m):
                    logits = net.sample(testing=True)        # (1,1,H,W)
                    prob   = torch.sigmoid(logits)[0,0].detach().cpu().numpy()  # (H,W)
                    X_list.append((prob > THRESH).astype(np.uint8))

            ged2 = est_ged2(X_list, Y_list)
            results[m].append(ged2)

        if k % 20 == 0:
            print(f"processed {k}/{len(val_idx)}")

    # 统计 & 绘图
    os.makedirs("plots", exist_ok=True)
    means = []
    stds  = []
    for m in M_LIST:
        arr = np.array(results[m], dtype=np.float32)
        means.append(np.nanmean(arr))
        stds.append(np.nanstd(arr))

    # Fig.4 风格：散点 + 均值标记
    plt.figure(figsize=(7,4))
    for i, m in enumerate(M_LIST):
        y = np.array(results[m], dtype=np.float32)
        x = np.full_like(y, fill_value=m, dtype=np.float32)
        plt.scatter(x, y, s=8, alpha=0.35, label=None)
        plt.plot([m], [means[i]], marker='D', markersize=8, color='red')  # 几何点=均值

    plt.title("Generalized Energy Distance (squared) vs #samples (LIDC)")
    plt.xlabel("#samples m")
    plt.ylabel(r"$\hat D^2_{\mathrm{GED}}$  (lower is better)")
    plt.grid(True, linestyle="--", alpha=0.4)
    # 标注均值
    for i, m in enumerate(M_LIST):
        plt.text(m+0.15, means[i], f"{means[i]:.3f}", va="center", fontsize=9, color="red")
    plt.tight_layout()
    out_path = os.path.join("plots", "ged_fig4_like.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    # 打印表格
    print("\nGED^2 summary (mean ± std):")
    for m, mu, sd in zip(M_LIST, means, stds):
        print(f"m={m:>2d} : {mu:.4f} ± {sd:.4f}")
    print("Saved plot to:", os.path.abspath(out_path))

if __name__ == "__main__":
    main()
