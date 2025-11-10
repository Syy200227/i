import torch, numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from load_LIDC_data import LIDC_IDRI
from probabilistic_unet import ProbabilisticUnet
import kl_fix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds = LIDC_IDRI(dataset_location=r'E:\workspace\LIDC-IDRI\pickle\\')
loader = DataLoader(ds, batch_size=2, shuffle=True)

net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192],
                        latent_dim=2, no_convs_fcomb=4, beta=10.0).to(device)
net.train()

patch, mask, _ = next(iter(loader))
print('patch:', patch.shape, patch.dtype, 'mask:', mask.shape, mask.dtype)

# 清洗&规整
patch = torch.clamp(patch, 0.0, 1.0)
mask = (mask > 0.5).float()
assert torch.isfinite(patch).all(), "patch 有 NaN/Inf"
assert torch.isfinite(mask).all(), "mask 有 NaN/Inf"

patch = patch.to(device)
mask  = mask.to(device).unsqueeze(1)

# 前向 + 一个 elbo
net.forward(patch, mask, training=True)
elbo = net.elbo(mask, analytic_kl=True)
print('ELBO:', elbo.item())

# 采样一张预测看看是否正常（不训练）
pred = torch.sigmoid(net.sample(testing=True))
print('pred:', pred.shape, 'finite?', torch.isfinite(pred).all().item())
