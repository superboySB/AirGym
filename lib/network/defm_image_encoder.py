import contextlib
import os
from typing import Sequence, Union

import torch
import torch.nn as nn

from lib.network.defm_preprocess import preprocess_depth_batch
from lib.network.defm_vit import build_defm_vit_s14, load_defm_weights


class DeFMImageEncoder(nn.Module):
    """
    Frozen (or optionally trainable) DeFM ViT-S/14 encoder for depth images.

    This module:
      1) loads DeFM ViT-S/14 checkpoint,
      2) applies DeFM metric-aware depth preprocessing,
      3) extracts intermediate-layer tokens,
      4) projects pooled tokens to a compact RL feature vector.
    """

    def __init__(self, config, device: str = "cuda:0"):
        super().__init__()
        self.config = config

        self.output_dim = int(getattr(config, "output_dim"))
        self.model_name = str(getattr(config, "model_name", "vit_s14")).lower()
        self.freeze_backbone = bool(getattr(config, "freeze_backbone", True))
        self.max_depth_c1 = float(getattr(config, "max_depth_c1", 100.0))
        self.max_depth_c2 = float(getattr(config, "max_depth_c2", 9.0))
        self.depth_scale_m = float(getattr(config, "depth_scale_m", 4.5))
        self.interpolation_mode = str(getattr(config, "interpolation_mode", "bilinear"))

        patch_size = int(getattr(config, "patch_size", 14))
        self.patch_size = patch_size
        self.target_size = self._parse_target_size(getattr(config, "target_size", [224, 224]))

        self.include_class_token = bool(getattr(config, "include_class_token", True))
        self.include_spatial_pool = bool(getattr(config, "include_spatial_pool", True))
        if not (self.include_class_token or self.include_spatial_pool):
            raise ValueError("At least one of include_class_token/include_spatial_pool must be True")

        self.intermediate_layers = getattr(config, "intermediate_layers", 1)
        self.layer_aggregation = str(getattr(config, "layer_aggregation", "last")).lower()

        if self.model_name not in {"vit_s14", "defm_vit_s14"}:
            raise ValueError(f"Unsupported DeFM model_name: {self.model_name}")

        self.backbone = build_defm_vit_s14(img_size=self.target_size)

        weight_file_path = os.path.join(config.model_folder, config.model_file)
        print("Loading DeFM weights from file:", weight_file_path)
        msg = load_defm_weights(self.backbone, weight_file_path, strict=True)
        print("Loaded DeFM weights:", msg)

        self.backbone.to(device)

        if self.freeze_backbone:
            self.backbone.eval()
            self.backbone.requires_grad_(False)

        base_dim = self.backbone.embed_dim
        fused_dim = 0
        if self.include_spatial_pool:
            fused_dim += base_dim
        if self.include_class_token:
            fused_dim += base_dim

        projector_hidden_dim = int(getattr(config, "projector_hidden_dim", 0))
        if projector_hidden_dim > 0:
            self.projector = nn.Sequential(
                nn.Linear(fused_dim, projector_hidden_dim),
                nn.ELU(),
                nn.Linear(projector_hidden_dim, self.output_dim),
            )
        else:
            self.projector = nn.Linear(fused_dim, self.output_dim)

        self.output_norm = nn.LayerNorm(self.output_dim)

    @staticmethod
    def _parse_target_size(target_size) -> tuple:
        if isinstance(target_size, int):
            return (target_size, target_size)
        if isinstance(target_size, (list, tuple)) and len(target_size) == 2:
            return (int(target_size[0]), int(target_size[1]))
        raise ValueError(f"Invalid target_size: {target_size}")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            # Keep the pretrained backbone in eval mode even when PPO model is in train mode.
            self.backbone.eval()
        return self

    def encode(self, image_tensors: torch.Tensor) -> torch.Tensor:
        processed = preprocess_depth_batch(
            image_tensors,
            target_size=self.target_size,
            patch_size=self.patch_size,
            max_depth_c1=self.max_depth_c1,
            max_depth_c2=self.max_depth_c2,
            depth_scale_m=self.depth_scale_m,
            interpolation_mode=self.interpolation_mode,
        )

        ctx = torch.no_grad() if self.freeze_backbone else contextlib.nullcontext()
        with ctx:
            layer_outputs = self.backbone.get_intermediate_layers(
                processed,
                n=self.intermediate_layers,
                reshape=True,
                return_class_token=True,
                norm=True,
            )

        fused_layers = []
        for spatial_tokens, class_token in layer_outputs:
            parts = []
            if self.include_spatial_pool:
                spatial_pool = torch.flatten(
                    torch.nn.functional.adaptive_avg_pool2d(spatial_tokens, output_size=1),
                    start_dim=1,
                )
                parts.append(spatial_pool)
            if self.include_class_token:
                parts.append(class_token)
            fused_layers.append(torch.cat(parts, dim=-1))

        if len(fused_layers) == 1 or self.layer_aggregation == "last":
            fused = fused_layers[-1]
        elif self.layer_aggregation == "mean":
            fused = torch.stack(fused_layers, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unsupported layer_aggregation: {self.layer_aggregation}")

        out = self.projector(fused)
        out = self.output_norm(out)
        return out

    def forward(self, image_tensors: torch.Tensor) -> torch.Tensor:
        return self.encode(image_tensors)

    def get_latent_dims_size(self) -> int:
        return self.output_dim
