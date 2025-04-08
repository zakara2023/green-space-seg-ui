"""
patch.py

Utilities for patch embedding images.
"""
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Patch and embed input image."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        """
        :param in_channels: the number of input channels.
        :type in_channels: int
        :param embed_dim: the size of the embedding dimension.
        :type embed_dim: int
        :param patch_size: the patch size
        :type patch_size: int
        """
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

        nn.init.xavier_uniform_(self.proj.weight.view([self.proj.weight.shape[0], -1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to patch and embed the image.

        :param x: the image
        :type x: torch.Tensor
        :return: the patched and embedded image (B, H*W, D)
        :rtype: torch.Tensor
        """
        x = self.proj(x) # perform the patch embedding (B, D, H, W)
        x = x.flatten(2).transpose(1, 2) # reshape for usage (B, H*W, D)
        return x
