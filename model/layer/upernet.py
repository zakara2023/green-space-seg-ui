"""
upernet.py

This module contains unified perceptual parsing layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layer.psp import PSP


class UPernetHead(nn.ModuleList):
    """Unified Perceptual Parsing for Scene Understanding."""

    def __init__(self,
                 pool_scales = (1, 2, 3, 6),
                 fpn_inplanes = (256, 512, 1024, 2048),
                 channels: int = 512,
                 dropout_ratio: float = 0.1,
                 align_corners: bool = False):
        super().__init__()
        self.align_corners = align_corners

        # fc heads
        self.conv_seg = nn.Conv2d(channels, 1, 1)
        self.dropout = nn.Dropout2d(dropout_ratio)

        # PSP Module
        self.psp_head = PSP(
            pool_scales,
            fpn_inplanes[-1],
            channels,
            align_corners=align_corners
        )

        # FPN Module
        self.fpn_in = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_inplane, channels, 1, bias=False),
                nn.ReLU(inplace=True)
            )
            for fpn_inplane in fpn_inplanes[:-1]
        ])

        self.fpn_out = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )
            for _ in fpn_inplanes[:-1]
        ])

        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(fpn_inplanes) * channels,
                      channels,
                      3,
                      padding=1),
            nn.ReLU(inplace=True)
        )

    def _forward_feature(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Forward function for feature maps."""
        # build laterals
        fpn_ins = [fpn(inputs[i]) for i, fpn in enumerate(self.fpn_in)]
        fpn_ins.append(self.psp_head(inputs[-1]))

        # build top-down path
        for i in range(len(fpn_ins) - 1, 0, -1):
            prev_shape = fpn_ins[i - 1].shape[2:]
            fpn_ins[i - 1] = fpn_ins[i - 1] + F.interpolate(
                fpn_ins[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            fpn_out(fpn_in)
            for fpn_out, fpn_in in zip(self.fpn_out, fpn_ins[:-1])
        ]
        # append psp feature
        fpn_outs.append(fpn_ins[-1])

        for i in range(len(fpn_ins) - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.conv_fusion(fpn_outs)
        return feats

    def _cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self._cls_seg(output)
        return output
