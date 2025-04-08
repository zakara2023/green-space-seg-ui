"""
mmae.py

This module contains layers for meta masked auto encoders.
these layers are built on the original MAE with support for
embedding additional metadata into the positional encodings.
"""
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from model.patch import PatchEmbed
from model.encoding import create_pos_encoding_2d


def _init_weights(m):
    """Based on MAE implementation we want linear layers to be xavier uniform"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def _init_token(m):
    """Based on MAE implementation we want cls and mask tokens to be normal"""
    nn.init.normal_(m, std=.02)


class MMAEEncoder(nn.Module):
    """The meta data masked encoder layer."""
    def __init__(self,
                 img_size: tuple[int]=(256, 256),
                 meta_dim: int=32,
                 model_dim: int=512,
                 patch_size: int=16,
                 num_layers: int=8,
                 num_heads: int=16,
                 ff_mul: int=4):
        """
        :param img_size: the image size
        :type img_size: tuple[int, int]
        :param meta_dim: the meta dimension
        :type meta_dim: int
        :param model_dim: the model dimension
        :type model_dim: int
        :param patch_size: the size of the patches
        :type patch_size: int
        :param num_layers: the number of transformer layers
        :type num_layers: int
        :param num_heads: the number of heads in the transformer
        :type num_heads: int
        :param ff_mul: the multiplier for the dimension of the feed forward layer
        :type ff_mul: int
        """
        super().__init__()

        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        _init_token(self.cls)

        self.src_embed = PatchEmbed(1, model_dim, patch_size)

        self.pos_encoder = create_pos_encoding_2d(
            meta_dim,
            model_dim,
            img_size[0]//patch_size,
            img_size[1]//patch_size)

        self.encoder = nn.ModuleList([
            Block(model_dim,
                  num_heads,
                  ff_mul,
                  qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(model_dim, eps=1e-6)

        self.apply(_init_weights)

    def random_mask(self, x: torch.Tensor, mask_pct: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a random mask for the input image tensor"""
        N, L, D = x.shape

        noise = torch.rand(N, L, device=x.device)
        keep = int(L * (1 - mask_pct))

        # generate masks by taking up to exactly mask_pct number of entries
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :keep].unsqueeze(-1).repeat(1, 1, D)
        x_masked = torch.gather(x, dim=1, index=ids_keep)

        # when creating the mask we need to unshuffle the sorted indices
        mask = torch.ones((N, L), device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x: torch.Tensor, meta: torch.Tensor, mask_pct: float=0.75) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the input image tensor

        :param x: the input image tensor (B, C, H, W)
        :type x: torch.Tensor
        :param meta: the meta data tensor (B, M)
        :type meta: torch.Tensor
        :return: the encoded image tensor and the mask used (B, H*W, D), (B, H*W, 1)
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        x = self.src_embed(x) # perform patch embedding (B, H*W, D)
        x = self.pos_encoder(x, meta) # positional encoding (B, H*W, D)

        x, mask, mask_ids = self.random_mask(x, mask_pct) # apply the mask to the image (B, M, D)

        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1) # add the cls token

        for layer in self.encoder: # encode the image (B, M, D)
            x = layer(x)
        x = self.norm(x)

        return x, mask, mask_ids


class MMAEDecoder(nn.Module):
    """The meta data masked decoder layer."""
    def __init__(self,
                 img_size: tuple[int]=(256, 256),
                 meta_dim: int=32,
                 embed_dim: int=512,
                 model_dim: int=512,
                 patch_size: int=16,
                 num_layers: int=8,
                 num_heads: int=16,
                 ff_mul: int=4):
        """
        :param img_size: the image size
        :type img_size: tuple[int, int]
        :param meta_dim: the metadata dimension
        :type meta_dim: int
        :param embed_dim: the embedding dimension of the input
        :type embed_dim: int
        :param model_dim: the model dimension
        :type model_dim: int
        :param patch_size: the size of the patches
        :type patch_size: int
        :param num_layers: the number of transformer layers
        :type num_layers: int
        :param num_heads: the number of heads in the transformer
        :type num_heads: int
        :param ff_mul: the multiplier for the dimension of the feed forward layer
        :type ff_mul: int
        """
        super().__init__()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        _init_token(self.mask_token)

        self.tgt_embed = nn.Linear(embed_dim, model_dim)

        self.pos_encoder = create_pos_encoding_2d(
            meta_dim,
            model_dim,
            img_size[0]//patch_size,
            img_size[1]//patch_size,
            cls_token=True)

        self.decoder = nn.ModuleList([
            Block(model_dim,
                  num_heads,
                  ff_mul,
                  qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(model_dim, eps=1e-6)
        self.fc = nn.Linear(model_dim, patch_size**2)

        self.apply(_init_weights)

    def restore_mask(self, x: torch.Tensor, mask_ids: torch.Tensor) -> torch.Tensor:
        """Restore the masked out positions"""
        N, L, D = x.shape
        mask_tokens = self.mask_token.repeat(N, mask_ids.shape[1] + 1 - L, 1) # number of masked tokens
        mask_ids = mask_ids.unsqueeze(-1).repeat(1, 1, D)

        x_tmp = torch.cat([x[:, 1:, :], mask_tokens], dim=1) # restore shape without cls (B, M, D) -> (B, HxW, D)
        x_tmp = torch.gather(x_tmp, dim=1, index=mask_ids) # put tokens back into their positions
        x = torch.cat([x[:, :1, :], x_tmp], dim=1) # restore clas (B, HxW+1, D)
        return x

    def forward(self, x: torch.Tensor, meta: torch.Tensor, mask_ids: torch.Tensor) -> torch.Tensor:
        """Decode the input image tensor

        :param x: the input encoded tensor (B, H*W, D)
        :type x: torch.Tensor
        :param meta: the meta data tensor (B, M)
        :type meta: torch.Tensor
        :param mask_ids: the ids to restore for masked out patches during encoding
        :type mask_ids: torch.Tensor
        :return: the decoded image tensor (B, C, H, W)
        :rtype: torch.Tensor
        """
        x = self.tgt_embed(x) # perfoem embedding for the decoder (B, M, D)
        x = self.restore_mask(x, mask_ids) # restore the masked patches (B, H*W, D)
        x = self.pos_encoder(x, meta) # positional encoding (B, H*W, D)

        for layer in self.decoder: # decode the image (B, H*W, D)
            x = layer(x)
        x = self.norm(x)
        x = self.fc(x) # map the output to the original image size (B, C, H, W)
        x = x[:, 1:, :] # remove the cls token

        return x


class MMAEFpnEncoder(nn.Module):
    """The mmae fpn encoder."""
    def __init__(self,
                 encoder: MMAEEncoder,
                 inindex: list[int] = [-1],
                 finalnorm: bool = False):
        super().__init__()
        self.encoder = encoder
        self.depth = len(encoder.encoder)
        self.inindex = self.correct_index(inindex, self.depth - 1)
        self.finalnorm = finalnorm

    def correct_index(self, inindex: list[int], last_index: int) -> list[int]:
        """Correct the input indexes"""
        return [val if val != -1 else last_index for val in inindex]

    def unpatch_embedded(self, embeds: torch.Tensor) -> torch.Tensor:
        """Unpatch the embedded tensors"""
        b, hw, embed_size = embeds.shape
        h = w = int(hw**.5)
        return (embeds.transpose(1, 2)
                      .reshape(b, embed_size, h, w)
                      .contiguous())

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> list[torch.Tensor]:
        """Perform a forward pass through the masked auto encoder layers.

        :param x: the input image tensor (B, C, H, W)
        :type x: torch.Tensor
        :param meta: the meta data of the aerial imagery
        :type meta: torch.Tensor
        :return: The fpn outputs of the encoder layer
        :rtype: list[torch.Tensor]"""
        y = self.encoder.src_embed(x) # perform patch embedding (B, H*W, D)
        y = self.encoder.pos_encoder(y, meta) # positional encoding (B, H*W, D)

        y = torch.cat([self.encoder.cls.expand(x.shape[0], -1, -1), y], dim=1) # add the cls token

        layers = self.encoder.encoder
        outputs = []

        for i, layer in enumerate(layers):
            y = layer(y)
            if i in self.inindex:
                if self.finalnorm and i == len(layers) - 1:  # apply the final normalization if active
                    y = self.norm(y)

                out = y[:, 1:]  # remove CLS token
                out = self.unpatch_embedded(out)  # unwind (B, D, H, W)
                outputs.append(out)

        return outputs
