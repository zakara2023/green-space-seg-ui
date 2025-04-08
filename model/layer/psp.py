"""
psp.py

This module contains layers for pyramid scene parsing.
"""
import torch
import torch.nn as nn
from model.layer.ppm import PPM


class PSP(nn.ModuleList):
    """Pyamid scene parsing layer."""

    def __init__(self,
                 pool_scales: tuple[int],
                 fc_dim: int,
                 channels: int,
                 align_corners: bool = False):
        super().__init__()
        self.psp_modules = PPM(
            pool_scales,
            fc_dim,
            channels,
            align_corners=align_corners
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(fc_dim + len(pool_scales) * channels,
                      channels,
                      3,
                      stride=1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of PSP module."""
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output