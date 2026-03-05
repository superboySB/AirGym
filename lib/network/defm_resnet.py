# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# This file is adapted for AirGym from the official DeFM repository:
# https://github.com/leggedrobotics/defm

from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor


class WeightedFusion(nn.Module):
    """Learnable weighted sum for BiFPN feature fusion."""

    def __init__(self, n_inputs: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))

    def forward(self, features):
        w = F.relu(self.weights)
        w = w / (w.sum() + 1e-6)
        return sum(w[i] * f for i, f in enumerate(features))


class BiFPNLayer(nn.Module):
    def __init__(self, out_channels: int, n_levels: int):
        super().__init__()
        self.out_channels = out_channels
        self.n_levels = n_levels

        self.fuse_topdown = nn.ModuleList([WeightedFusion(2) for _ in range(n_levels - 1)])
        self.fuse_bottomup = nn.ModuleList([WeightedFusion(2) for _ in range(n_levels - 1)])
        self.convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in range(n_levels)]
        )

    def forward(self, feats):
        td_feats = [None] * self.n_levels
        td_feats[-1] = feats[-1]
        for i in range(self.n_levels - 2, -1, -1):
            up = F.interpolate(td_feats[i + 1], size=feats[i].shape[-2:], mode="nearest")
            td_feats[i] = self.fuse_topdown[i]([feats[i], up])

        out_feats = [None] * self.n_levels
        out_feats[0] = td_feats[0]
        for i in range(1, self.n_levels):
            down = F.max_pool2d(out_feats[i - 1], kernel_size=2, stride=2)
            out_feats[i] = self.fuse_bottomup[i - 1]([td_feats[i], down])

        out_feats = [conv(f) for conv, f in zip(self.convs, out_feats)]
        return out_feats


class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels: int, n_blocks: int = 1):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list])
        self.blocks = nn.ModuleList([BiFPNLayer(out_channels, len(in_channels_list)) for _ in range(n_blocks)])

    def forward(self, inputs: OrderedDict):
        feats = [proj(inputs[k]) for proj, k in zip(self.proj, inputs.keys())]
        for block in self.blocks:
            feats = block(feats)
        return OrderedDict([(k, f) for k, f in zip(inputs.keys(), feats)])


def replace_bn_with_gn(model: nn.Module, num_groups: int = 32):
    """Recursively replace all BatchNorm2d with GroupNorm."""
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            gn = nn.GroupNorm(num_groups=num_groups, num_channels=module.num_features)
            setattr(model, name, gn)
        else:
            replace_bn_with_gn(module, num_groups=num_groups)
    return model


class ResNetBiFPN(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        out_channels: int = 128,
        n_blocks: int = 1,
        pretrained: bool = False,
        gn_groups: int = 32,
    ):
        super().__init__()
        if backbone_name != "resnet18":
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        backbone = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        backbone = replace_bn_with_gn(backbone, num_groups=gn_groups)

        return_nodes = {
            "layer2": "c3",  # /8
            "layer3": "c4",  # /16
            "layer4": "c5",  # /32
        }
        self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            feats = self.backbone(dummy)
            cls = self.global_pool(feats["c5"]).flatten(1)
            embed_dim = cls.shape[1]
        in_channels_list = [f.shape[1] for f in feats.values()]
        print(f"Backbone {backbone_name} feature channels: {in_channels_list}")

        self.fpn = BiFPN(in_channels_list, out_channels, n_blocks=n_blocks)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm_cls = norm_layer(embed_dim)
        self.norm_patch = norm_layer(out_channels)

    def forward(self, x, norm: bool = True):
        feats = self.backbone(x)
        feats = OrderedDict([(k, v) for k, v in feats.items()])
        global_feat_backbone = self.global_pool(feats["c5"]).flatten(1)
        feats_bifpn = self.fpn(feats)

        if norm:
            global_feat_backbone = self.norm_cls(global_feat_backbone)
            for k in feats_bifpn.keys():
                feats_bifpn[k] = feats_bifpn[k].permute(0, 2, 3, 1)
                feats_bifpn[k] = self.norm_patch(feats_bifpn[k])
                feats_bifpn[k] = feats_bifpn[k].permute(0, 3, 1, 2)

        dense_feats_bifpn = {
            "P3": feats_bifpn["c3"],
            "P4": feats_bifpn["c4"],
            "P5": feats_bifpn["c5"],
        }
        return {
            "global_backbone": global_feat_backbone,
            "dense_bifpn": dense_feats_bifpn,
        }

    def forward_no_bifpn(self, x, norm: bool = True):
        feats = self.backbone(x)
        feats = OrderedDict([(k, v) for k, v in feats.items()])
        global_feat_backbone = self.global_pool(feats["c5"]).flatten(1)

        if norm:
            global_feat_backbone = self.norm_cls(global_feat_backbone)

        dense_feats = {
            "P3": feats["c3"],
            "P4": feats["c4"],
            "P5": feats["c5"],
        }
        return {
            "global_backbone": global_feat_backbone,
            "dense_feats": dense_feats,
        }


def build_defm_resnet18(out_channels: int = 128, n_blocks: int = 1):
    return ResNetBiFPN(
        backbone_name="resnet18",
        out_channels=out_channels,
        n_blocks=n_blocks,
        pretrained=False,
        gn_groups=32,
    )


def load_defm_resnet_weights(model: nn.Module, ckpt_path: str, strict: bool = True):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unexpected checkpoint type: {type(state_dict)}")

    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    clean_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        clean_dict[key] = value

    return model.load_state_dict(clean_dict, strict=strict)
