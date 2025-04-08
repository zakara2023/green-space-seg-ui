"""
aegis.py

This module contains the AeGIS model for greenspace segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layer.mmae import MMAEEncoder, MMAEFpnEncoder
from model.layer.fpn import FPNUpscale
from model.layer.upernet import UPernetHead
from model.layer.segformer import SegformerHead


class AeGIS(nn.Module):
    """The Aerial Greenery Image Segmentation (AeGIS) model."""

    def __init__(self,
                 embed_dim: int,
                 encoder: MMAEEncoder,
                 decoder: nn.Module,
                 inindex: list[int] = [-1],
                 finalnorm: bool = False):
        super().__init__()
        self.encoder = MMAEFpnEncoder(encoder=encoder,
                                      inindex=inindex,
                                      finalnorm=finalnorm)
        self.neck = FPNUpscale(embed_dim=embed_dim)
        self.decoder = decoder

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the AeGIS model.

        :param x: the input image tensor (B, C, H, W)
        :type x: torch.Tensor
        :param meta: the meta data for location and date of the aerial imagery
        :type meta: torch.Tensor
        :return: the output segmentation mask
        :rtype: torch.Tensor"""
        y = self.encoder(x, meta)  # fpn of encodings [(B, D, H, W)]
        y = self.neck(y)  # upscale the fon encodings
        y = self.decoder(y)  # (B, 1, H, W)
        y = F.interpolate(
            input=y,
            size=(x.shape[-2], x.shape[-1]),
            mode='bilinear',
            align_corners=False)

        return y


class AeGISUpernet(AeGIS):
    """The Aerial Greenery Image Segmentation (AeGIS) model."""

    def __init__(self,
                 embed_dim: int,
                 encoder: MMAEEncoder,
                 inindex: list[int] = [-1],
                 finalnorm: bool = False):
        super().__init__(embed_dim,
                         encoder,
                         UPernetHead(fpn_inplanes=[embed_dim]*len(inindex),
                                     channels=embed_dim),
                         inindex,
                         finalnorm)


class AeGISFormer(AeGIS):
    """The Aerial Greenery Image Segmentation (AeGIS) model."""

    def __init__(self,
                 embed_dim: int,
                 encoder: MMAEEncoder,
                 inindex: list[int] = [-1],
                 finalnorm: bool = False):
        super().__init__(embed_dim,
                         encoder,
                         SegformerHead(fpn_inplanes=[embed_dim]*len(inindex),
                                       channels=embed_dim),
                         inindex,
                         finalnorm)
