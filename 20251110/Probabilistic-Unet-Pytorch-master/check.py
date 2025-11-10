import torch
from torch.distributions import Normal, Independent, kl
import kl_fix  # 必须导入

p = Independent(Normal(torch.zeros(3), torch.ones(3)), 1)  # reinterpreted 1 维
q = Independent(Normal(torch.ones(3), torch.ones(3)), 1)
v = kl.kl_divergence(p, q)  # 应该返回一个标量张量
print(v, v.shape)           # tensor(1.5) 类似，shape: torch.Size([])
