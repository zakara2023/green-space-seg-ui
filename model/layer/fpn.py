"""
fpn.py

This module contains layers for working with feature pyramid networks.
"""
import torch
import torch.nn as nn


class FPNUpscale(nn.Module):
    """Feature pyramid upsampling layer.
    
    Primarily used with transformer blocks to adjust feature pyramid layers as multiscale resolutions.
    """

    def __init__(self, embed_dim: int):
        super().__init__()

        self.upsample_4x = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim,
                                kernel_size=2,
                                stride=2),
            nn.SyncBatchNorm(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim,
                                kernel_size=2,
                                stride=2),
        )

        self.upsample_2x = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, 
                                kernel_size=2,
                                stride=2))

        self.identity = nn.Identity()
        self.downsample_2x = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ops = [
            self.upsample_4x,
            self.upsample_2x,
            self.identity,
            self.downsample_2x
        ]

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Upsample the input feature pyramid.

        :param inputs: the input feature pyramid
        :type inputs: list[torch.Tensor]
        :return: the upscaled features
        :rtype: list[torch.Tensor]
        :throws AssertionError: the number of input feature layers does match the operations (4).
        """
        assert len(inputs) == len(self.ops)
        return [op(inp) for op, inp in zip(self.ops, inputs)]


class FPNDownscale(nn.Module):
    """Feature pyramid downsampling layer.
    
    Primarily used with transformer blocks to adjust feature pyramid layers as multiscale resolutions.
    """

    def __init__(self, embed_dim: int):
        super().__init__()

        self.upsample_2x = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, 
                                kernel_size=2,
                                stride=2))

        self.identity = nn.Identity()
        self.downsample_2x = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downsample_4x = nn.MaxPool2d(kernel_size=4, stride=4)

        self.ops = [
            self.upsample_2x,
            self.identity,
            self.downsample_2x,
            self.downsample_4x
        ]

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Downscale the input feature pyramid.

        :param inputs: the input feature pyramid
        :type inputs: list[torch.Tensor]
        :return: the upscaled features
        :rtype: list[torch.Tensor]
        :throws AssertionError: the number of input feature layers does match the operations (4).
        """
        assert len(inputs) == len(self.ops)
        return [op(inp) for op, inp in zip(self.ops, inputs)]
