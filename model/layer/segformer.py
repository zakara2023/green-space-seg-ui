"""
segformer.py

This module contains segformer layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegformerHead(nn.Module):
    """Segformer segmentation head."""

    def __init__(self,
                 fpn_inplanes = (256, 512, 1024, 2048),
                 channels: int = 512,
                 dropout_ratio: float = 0.1,
                 align_corners: bool=False):
        super().__init__()
        self.align_corners = align_corners

        # fc heads
        self.conv_seg = nn.Conv2d(channels, 1, 1)
        self.dropout = nn.Dropout2d(dropout_ratio)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_in, channels, 1, stride=1),
                nn.ReLU(inplace=True)
            )
            for fpn_in in fpn_inplanes
        ])

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(len(fpn_inplanes) * channels, channels, 1),
            nn.ReLU(inplace=True)
        )

    def _forward_feature(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Forward function for feature maps."""
        size = inputs[0].shape[2:]
        outs = [
            F.interpolate(
                conv(x),
                size=size,
                mode='bilinear',
                align_corners=self.align_corners
            )
            for x, conv in zip(inputs, self.convs)
        ]

        y = self.fusion_conv(torch.cat(outs, dim=1))
        return y

    def _cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self._cls_seg(output)
        return output
