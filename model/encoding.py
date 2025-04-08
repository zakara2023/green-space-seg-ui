"""
encoding.py

Utilities for encoding the positional and geo locational data into the model.
"""
import torch
import torch.nn as nn
import numpy as np


def positional_encoding_1d(temperature: float,
                           embed_dim: int,
                           pos: torch.Tensor) -> torch.Tensor:
    """Get the 1d positional encoding

    :param temperature: The temperature
    :type temperature: float
    :param embed_dim: The embedding dimension
    :type embed_dim: int
    :param pos:  The positional mesh
    :type pos: torch.Tensor
    :return: The positional encoding
    :rtype: torch.Tensor
    """
    pos = pos.reshape(-1)

    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega = 1.0 / temperature ** (omega / (embed_dim / 2))

    pos = torch.outer(pos, omega)

    embed = torch.concatenate([
        torch.sin(pos),
        torch.cos(pos)
    ], dim=-1)

    return embed


class PositionalEncoding2D(nn.Module):
    """Encode the position of the pixels in the image."""

    def __init__(self,
                 d_model: int,
                 width: int,
                 height: int,
                 temperature: float=10_000.0,
                 normalize: bool=False,
                 cls_token: bool=False):
        """
        :param d_model: The model dimension
        :type d_model: int
        :param width: The image width
        :type width: int
        :param height: The image height
        :type height: int
        :param temperature: The temperature for the positional encoding (optional, default=10000.0)
        :type temperature: float
        :param normalize: Normalize the positional encoding (optional, default=False)
        :type normalize: bool
        :param cls_token: Add a cls token to the positional encoding (optional, default=False)
        :type cls_token: bool
        """
        super().__init__()
        self.width = width
        self.height = height
        self.d_model = d_model
        self.hd_model = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        self.cls_token = cls_token
        self.scale = 2 * np.pi
        self.eps = 1e-6

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize the positional encoding weights."""
        self.register_buffer('pe', self._positional_encoding_2d())

    def _positional_encoding_2d(self):
        x_pos, y_pos = torch.meshgrid(
            torch.arange(self.width, dtype=torch.float32),
            torch.arange(self.height, dtype=torch.float32),
            indexing='ij'
        )

        x_pos = x_pos.unsqueeze(-1)
        y_pos = y_pos.unsqueeze(-1)

        if self.normalize:
            y_pos = y_pos / (y_pos[-1:, :] + self.eps) * self.scale
            x_pos = x_pos / (x_pos[:, -1:] + self.eps) * self.scale

        omega = torch.arange(self.hd_model // 2, dtype=torch.float32)
        omega = 1.0 / self.temperature ** (omega / (self.hd_model / 2))

        y_pos = y_pos * omega
        x_pos = x_pos * omega

        embed = torch.concatenate([
            torch.sin(y_pos),
            torch.cos(y_pos),
            torch.sin(x_pos),
            torch.cos(x_pos)
        ], dim=-1).flatten(0, 1)

        if self.cls_token:
            embed = torch.cat([torch.zeros((1, self.d_model)), embed], dim=0)

        return embed.unsqueeze(0)

    def get_pe(self, x: torch.Tensor) -> torch.Tensor:
        """Get the positional encoding."""
        pe = self.pe.to(x.device)
        pe.requires_grad = False
        return pe

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Add the positional encoding to the input tensor."""
        pe = self.get_pe(x)
        return x + pe


class ExtendedPositionalEncoding2D(PositionalEncoding2D):
    """Extends the 2D positional encoding with additional meta data encodes."""

    def __init__(self,
                 g_model: int,
                 d_model: int,
                 width: int,
                 height: int,
                 temperature: float=10000.0,
                 normalize: bool=False,
                 cls_token: bool=False):
        """
        :param g_model: The geo model dimension
        :type g_model: int
        :param d_model: The model dimension
        :type d_model: int
        :param width: The image width
        :type width: int
        :param height: The image height
        :type height: int
        :param temperature: The temperature for the positional encoding (optional, default=10000.0)
        :type temperature: float
        :param normalize: Normalize the positional encoding (optional, default=False)
        :type normalize: bool
        :param cls_token: Add a cls token to the positional encoding (optional, default=False)
        :type cls_token: bool
        """
        super().__init__(
            d_model = d_model - g_model,
            width=width,
            height=height,
            temperature=temperature,
            normalize=normalize,
            cls_token=cls_token
        )

        self.g_model = g_model

    def get_embed(self, meta: torch.Tensor) -> torch.Tensor:
        """Get the extension embeddings."""
        embed_dim = self.g_model // meta.shape[1]
        return torch.cat([
            positional_encoding_1d(self.temperature, embed_dim, meta[:, i])
            for i in range(meta.shape[1])
        ], dim=1) # (B, L, D)

    def get_pe(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """Get the extended positional encoding."""
        embed = self.get_embed(meta)

        embed = embed.reshape(-1, 1, embed.shape[-1]).unsqueeze(2) # add additional dim for x (B, L, 1, D)
        embed_ = embed.expand(-1, -1, x.shape[1], -1) # expand into new dim (B, L, X/L, D)
        embed = embed_.reshape(x.shape[0], -1, embed.shape[-1]) # compress the middle region to (B, X, D)

        pos = self.pe.repeat(x.shape[0], 1, 1) # the positional encoder does not care about batches so expand to (B, X, D)
        pe = torch.cat([pos, embed], dim=-1)
        pe = pe.to(x.device)
        pe.requires_grad = False

        return pe

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """Add the positional encoding to the input tensor."""
        pe = self.get_pe(x, meta)
        return x + pe


def create_pos_encoding_2d(g_model: int,
                           d_model: int,
                           width: int,
                           height: int,
                           temperature: float=10000.0,
                           normalize: bool=False,
                           cls_token: bool=False) -> PositionalEncoding2D:
    """
    :param g_model: The geo model dimension
    :type g_model: int
    :param d_model: The model dimension
    :type d_model: int
    :param width: The image width
    :type width: int
    :param height: The image height
    :type height: int
    :param temperature: The temperature for the positional encoding (optional, default=10000.0)
    :type temperature: float
    :param normalize: Normalize the positional encoding (optional, default=False)
    :type normalize: bool
    :param cls_token: Add a cls token to the positional encoding (optional, default=False)
    :type cls_token: bool
    """
    if g_model == 0:
        return PositionalEncoding2D(
            d_model=d_model,
            width=width,
            height=height,
            temperature=temperature,
            normalize=normalize,
            cls_token=cls_token
        )

    return ExtendedPositionalEncoding2D(
        g_model=g_model,
        d_model=d_model,
        width=width,
        height=height,
        temperature=temperature,
        normalize=normalize,
        cls_token=cls_token
    )
