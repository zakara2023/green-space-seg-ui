"""
ppm.py

This module contains layers for feature pyramid pooling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.ModuleList):
    """Pooling pyramid layer."""

    def __init__(self,
                 pool_scales: tuple[int],
                 fc_dim: int,
                 channels: int,
                 align_corners: bool = False):
        super().__init__()
        self.align_corners = align_corners

        for pool_scale in pool_scales:
            self.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_scale),
                nn.Conv2d(fc_dim, channels, 1),
                nn.ReLU()
            ))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through pyramid pooling."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)

        return ppm_outs
