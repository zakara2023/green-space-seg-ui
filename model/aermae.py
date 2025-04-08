"""
aer_mae.py

The ViT masked encoder used for aerial photos.
"""
import torch
import torch.nn as nn

from model.layer.mmae import MMAEEncoder, MMAEDecoder


def mae_loss(y_true: torch.Tensor,
             y_pred:torch.Tensor,
             mask: torch.Tensor,
             weight: float=0.,
             eps: float=1e-6) -> torch.Tensor:
    """Calculate masked auto encoder loss.

    :param y_true: true values
    :type y_true: torch.Tensor
    :param y_pred: predicted values
    :type y_pred: torch.Tensor
    :param mask: mask
    :type mask: torch.Tensor
    :param weight: the unmasked weights (Optional, default=0)
    :type weight: float
    :param eps: safety term for division (Optional, default=1e-6)
    :type eps: float
    :return: masked auto encoder loss
    :rtype: torch.Tensor
    """
    loss = (y_true - y_pred) ** 2
    loss = loss.mean(dim=-1)
    masked_loss = (loss * mask).sum() / (mask.sum() + eps)

    unmasked = 1 - mask
    unmasked_loss = (loss * unmasked * weight).sum() / (unmasked.sum() + eps)
    return masked_loss + unmasked_loss


class MAELoss(nn.Module):
    """masked auto envoder loss."""

    def __init__(self, weight: float=0., eps: float=1e-6):
        """
        :param weight: the unmasked weights (Optional, default=0)
        :type weight: float
        :param eps: safety term for division (Optional, default=1e-6)
        :type eps: float
        """
        super().__init__()

        self.weight = weight
        self.eps = eps

    def forward(self,
                y_true: torch.Tensor,
                y_pred: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Calculate masked auto encoder loss.

        :param y_true: true values
        :type y_true: torch.Tensor
        :param y_pred: predicted values
        :type y_pred: torch.Tensor
        :param mask: mask
        :type mask: torch.Tensor
        :return: masked auto encoder loss
        :rtype: torch.Tensor
        """
        return mae_loss(y_true, y_pred, mask, self.weight, self.eps)


class AerMAE(nn.Module):
    """A masked auto encoder used for aerial photos.

    Photos should be gray scale
    """
    def __init__(self,
                 img_size: tuple[int]=(256, 256),
                 patch_size: int=16,
                 enc_meta_dim: int=32,
                 enc_dim: int=512,
                 dec_meta_dim: int=32,
                 dec_dim: int=512,
                 enc_layers: int=8,
                 enc_heads: int=16,
                 dec_layers: int=8,
                 dec_heads: int=16,
                 ff_mul: int=4,
                 mask_pct: float=0.75):
        """
        :param img_size: the image size
        :type img_size: tuple[int, int]
        :param patch_size: the size of the patches
        :type patch_size: int
        :param enc_layers: the number of encoder transformer layers
        :param enc_meta_dim: the metadata dimension for the encoder
        :type enc_meta_dim: int
        :param enc_dim: the encoder model dimension
        :type enc_dim: int
        :param dec_meta_dim: the metadata dimension for the decoder
        :type dec_meta_dim: int
        :param dec_dim: the decoder model dimension
        :type dec_dim: int
        :type enc_layers: int
        :param enc_heads: the number of heads in the encoder transformers
        :type enc_heads: int
        :param dec_layers: the number of decoder transformer layers
        :type dec_layers: int
        :param dec_heads: the number of heads in the decoder transformers
        :type dec_heads: int
        :param ff_mul: the multiplier for the dimension of the feed forward layer
        :type ff_mul: int
        :param mask_pct: the masking rate
        :type mask_pct: float
        """
        super().__init__()

        self.patch_size = patch_size
        self.mask_pct = mask_pct

        self.encoder = MMAEEncoder(
            img_size=img_size,
            meta_dim=enc_meta_dim,
            model_dim=enc_dim,
            patch_size=patch_size,
            num_layers=enc_layers,
            num_heads=enc_heads,
            ff_mul=ff_mul,
        )

        self.decoder = MMAEDecoder(
            img_size=img_size,
            meta_dim=dec_meta_dim,
            embed_dim=enc_dim,
            model_dim=dec_dim,
            patch_size=patch_size,
            num_layers=dec_layers,
            num_heads=dec_heads,
            ff_mul=ff_mul
        )

    def forward(self,
        x: torch.Tensor,
        loc: torch.Tensor,
        mask_pct: float=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass through the masked auto encoder model

        :param x: the input image tensor (B, C, H, W)
        :type x: torch.Tensor
        :param loc: the meta data location of the aerial data (B, 4)
        :type loc: torch.Tensor
        :param mask_pct: the masking rate
        :type mask_pct: float
        :return: the output image tensor and the mask used (B, C, H, W), (B, H*W, 1)
        :rtype: tuple[torch.Tensor, torch.Tensor]"""
        mask_pct = mask_pct if mask_pct != None else self.mask_pct

        mem, mask, mask_ids = self.encoder(x, loc, mask_pct)
        y = self.decoder(mem, loc, mask_ids)
        return y, mask
