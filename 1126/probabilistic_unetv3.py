# probabilistic_unetv3.py
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent

from unet import Unet
from unet_blocks import *
from utils import init_weights, init_weights_orthogonal_normal
from DEvis import DEviSHead, devis_loss_from_evidence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------- Encoder ----------------------
class Encoder(nn.Module):
    def __init__(self, input_channels, num_filters, no_convs_per_block, padding=True):
        super().__init__()
        layers = []
        for i in range(len(num_filters)):
            in_ch = input_channels if i == 0 else out_ch
            out_ch = num_filters[i]

            if i != 0:
                layers.append(nn.AvgPool2d(2, 2))

            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block - 1):
                layers.append(nn.Conv2d(out_ch, out_ch, 3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)


# ---------------- AxisAlignedConvGaussian ----------------
class AxisAlignedConvGaussian(nn.Module):
    """
    prior:  输入 image
    posterior: 输入 image + mask + doctor_onehot
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block,
                 latent_dim, initializers, posterior=False, num_doctors=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.posterior = posterior
        self.num_doctors = num_doctors

        if posterior:
            enc_in = input_channels + 1 + num_doctors
        else:
            enc_in = input_channels

        self.encoder = Encoder(enc_in, num_filters, no_convs_per_block)

        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * latent_dim, 1)
        if initializers["w"] == "orthogonal":
            init_weights_orthogonal_normal(self.conv_layer)
        else:
            init_weights(self.conv_layer)

    def forward(self, x, mask=None, doctor_onehot=None):
        if self.posterior:
            # x: (B,1,H,W), mask: (B,1,H,W), doctor_onehot: (B,R,H,W)
            x = torch.cat((x, mask, doctor_onehot), dim=1)

        enc = self.encoder(x)                         # (B,C,H',W')
        enc = enc.mean(dim=[2, 3], keepdim=True)      # 全局平均池化 -> (B,C,1,1)

        mu_log_sigma = self.conv_layer(enc).squeeze(-1).squeeze(-1)  # (B,2*latent)
        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:].clamp(-7, 7)
        scale = torch.exp(log_sigma).clamp(min=1e-6)

        return Independent(Normal(loc=mu, scale=scale), 1)


# ---------------------- Fcomb ----------------------
class Fcomb(nn.Module):
    def __init__(self, num_filters, latent_dim, num_classes,
                 no_convs_fcomb, initializers):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_filters = num_filters

        layers = [
            nn.Conv2d(num_filters[0] + latent_dim, num_filters[0], 1),
            nn.ReLU(inplace=True)
        ]
        for _ in range(no_convs_fcomb - 2):
            layers += [
                nn.Conv2d(num_filters[0], num_filters[0], 1),
                nn.ReLU(inplace=True)
            ]

        self.layers = nn.Sequential(*layers)
        self.last = nn.Conv2d(num_filters[0], num_classes, 1)

        if initializers["w"] == "orthogonal":
            self.layers.apply(init_weights_orthogonal_normal)
            self.last.apply(init_weights_orthogonal_normal)
        else:
            self.layers.apply(init_weights)
            self.last.apply(init_weights)

    def forward(self, fmap, z):
        B, C, H, W = fmap.shape
        z = z[:, :, None, None].expand(B, self.latent_dim, H, W)  # (B,latent,H,W)
        x = torch.cat([fmap, z], dim=1)
        x = self.layers(x)
        return self.last(x)   # 线性 logits (B,num_classes,H,W)


# ------------------ ProbabilisticUNet --------------------
class ProbabilisticUnet(nn.Module):
    def __init__(self, input_channels=1, num_classes=2,
                 num_filters=[32, 64, 128, 192], latent_dim=8,
                 no_convs_fcomb=4, beta=0.03, edl_lambda=1e-4,
                 dice_weight=0.5, num_doctors=4):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.num_doctors = num_doctors

        self.initializers = {'w': 'he_normal', 'b': 'normal'}

        self.beta = beta
        self.beta_t = beta
        self.edl_lambda = edl_lambda
        self.dice_weight = dice_weight

        # -------- UNet backbone（不做最后 1×1 分类层）--------
        self.unet = Unet(
            self.input_channels,
            self.num_classes,
            self.num_filters,
            self.initializers,
            apply_last_layer=False,
            padding=True
        ).to(device)

        # -------- prior / posterior --------
        self.prior = AxisAlignedConvGaussian(
            self.input_channels,
            self.num_filters,
            self.no_convs_per_block,
            self.latent_dim,
            self.initializers,
            posterior=False,
            num_doctors=self.num_doctors
        ).to(device)

        self.posterior = AxisAlignedConvGaussian(
            self.input_channels,
            self.num_filters,
            self.no_convs_per_block,
            self.latent_dim,
            self.initializers,
            posterior=True,
            num_doctors=self.num_doctors
        ).to(device)

        # -------- Fcomb & DEviS --------
        self.fcomb = Fcomb(
            self.num_filters,
            self.latent_dim,
            self.num_classes,
            self.no_convs_fcomb,
            self.initializers
        ).to(device)

        self.devis = DEviSHead(in_ch=self.num_classes,
                               num_classes=self.num_classes)

        # 监控用
        self.kl = torch.tensor(0.0, device=device)
        self.mean_recon = torch.tensor(0.0, device=device)

    # ---------------------- 前向 ----------------------
    def forward(self, img, mask, doctor_id, training=True):
        """
        img: (B,1,H,W)
        mask: (B,1,H,W)
        doctor_id: (B,R,H,W)  one-hot
        """
        if training:
            self.posterior_latent_space = self.posterior(img, mask, doctor_id)
        self.prior_latent_space = self.prior(img)
        self.unet_features = self.unet(img, False)

    # ---------------------- 解码 ----------------------
    def _decode(self, z):
        logits = self.fcomb(self.unet_features, z)
        return self.devis(logits)      # dict: evidence, alpha, S, probs

    # ---------------------- 采样 ----------------------
    def sample(self):
        z = self.prior_latent_space.sample()
        return self._decode(z)

    # ---------------------- ELBO ----------------------
    def elbo(self, mask):
        # 1) posterior sample
        z_post = self.posterior_latent_space.rsample()
        log_q = self.posterior_latent_space.log_prob(z_post)
        log_p = self.prior_latent_space.log_prob(z_post)
        self.kl = (log_q - log_p).mean()

        # 2) reconstruction: Evidential CE + Dice
        out = self._decode(z_post)
        prob_fg = out["probs"][:, 1:2]
        y = torch.cat([1 - mask, mask], dim=1)

        ce = devis_loss_from_evidence(out["evidence"], y,
                                      lambda_kl=self.edl_lambda)

        dice_num = 2 * (prob_fg * mask).sum()
        dice_den = (prob_fg + mask).sum() + 1e-7
        dice_loss = 1.0 - dice_num / dice_den

        recon = ce + self.dice_weight * dice_loss
        self.mean_recon = recon.detach()

        return -(recon + self.beta_t * self.kl)
