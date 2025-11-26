# -*- coding: utf-8 -*-
"""
DEviS / Evidential 分支：
- 用 Softplus 产生每类非负“证据” e>=0
- Dirichlet 参数 alpha = e + 1
- 概率使用 E[p] = alpha / sum(alpha)
- 损失：Dirichlet 的期望交叉熵 + 相对均匀先验的 KL（校准正则）
"""
import torch, torch.nn as nn
import torch.nn.functional as F
class DEviSHead(nn.Module):
    """
    把 backbone/UNet+Fcomb 的线性输出映射到每类证据 e>=0
    默认保留一个 1×1 conv（可以看作轻量适配层），然后 Softplus 得到 evidence
    """
    def __init__(self, in_ch: int, num_classes: int,
                 beta: float = 1.0, threshold: float = 20.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=True)
        self.act  = nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, feat):
        e = self.act(self.conv(feat))           # (B,C,H,W), evidence>=0
        alpha = e + 1.0                         # Dirichlet concentration
        S = alpha.sum(dim=1, keepdim=True)      # (B,1,H,W)
        probs = alpha / (S + 1e-8)              # E[p] under Dirichlet
        return {"evidence": e, "alpha": alpha, "S": S, "probs": probs}


# ------------- Dirichlet 工具函数 -------------
def dirichlet_kl(alpha, beta):
    """
    KL(Dir(alpha) || Dir(beta))，逐像素对通道维求和，返回 (B,H,W)
    """
    lgamma = torch.lgamma
    digamma = torch.digamma

    sum_alpha = alpha.sum(dim=1)
    sum_beta  = beta.sum(dim=1)

    t1 = lgamma(sum_alpha) - lgamma(sum_beta)
    t2 = (lgamma(beta)).sum(dim=1) - (lgamma(alpha)).sum(dim=1)
    t3 = ((alpha - beta) * (digamma(alpha) - digamma(sum_alpha.unsqueeze(1)))).sum(dim=1)
    return t1 + t2 + t3


def edl_expected_ce(alpha, y_onehot):
    """
    E[-log p(y)] = - sum_k y_k * (psi(alpha_k) - psi(S))
    """
    S = alpha.sum(dim=1, keepdim=True)
    expected_log = torch.digamma(alpha) - torch.digamma(S)
    ce = -(y_onehot * expected_log).sum(dim=1)   # (B,H,W)
    return ce


def devis_loss_from_evidence(evidence, y_onehot, lambda_kl=1e-3):
    """
    总损失 = 期望交叉熵 + λ * KL(Dir(alpha) || Dir(1))
    返回标量 mean
    """
    alpha = evidence + 1.0
    ce = edl_expected_ce(alpha, y_onehot)                     # (B,H,W)
    kl = dirichlet_kl(alpha, torch.ones_like(alpha))          # (B,H,W)
    loss_map = ce + lambda_kl * kl
    return loss_map.mean()
