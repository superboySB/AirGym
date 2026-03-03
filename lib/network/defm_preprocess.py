# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# This file is adapted for AirGym from the official DeFM repository:
# https://github.com/leggedrobotics/defm

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

try:
    from PIL.Image import Image as PILImageType
except Exception:  # pragma: no cover
    PILImageType = None


DEFM_MEAN = (0.248880, 0.495620, 0.492858)
DEFM_STD = (0.139357, 0.271314, 0.297177)


def get_target_size(
    current_size: Tuple[int, int],
    target_size: Optional[Union[int, Tuple[int, int], list]] = None,
    patch_size: Optional[int] = None,
) -> Tuple[int, int]:
    curr_h, curr_w = current_size

    if target_size is None:
        new_h, new_w = curr_h, curr_w
    elif isinstance(target_size, int):
        new_h, new_w = target_size, target_size
    else:
        if len(target_size) != 2:
            raise ValueError(f"target_size should have length 2, got: {target_size}")
        new_h, new_w = int(target_size[0]), int(target_size[1])

    if patch_size is not None:
        new_h = (new_h // patch_size) * patch_size
        new_w = (new_w // patch_size) * patch_size

    if new_h <= 0 or new_w <= 0:
        raise ValueError(
            f"Invalid resized shape ({new_h}, {new_w}). Check target_size={target_size}, patch_size={patch_size}."
        )

    return new_h, new_w


def preprocess_depth_batch(
    input_batch: Union[np.ndarray, torch.Tensor],
    target_size: Optional[Union[int, Tuple[int, int], list]] = (224, 224),
    patch_size: Optional[int] = 14,
    max_depth_c1: float = 100.0,
    max_depth_c2: float = 9.0,
    depth_scale_m: Optional[float] = None,
    interpolation_mode: str = "bilinear",
) -> torch.Tensor:
    """
    Vectorized DeFM depth preprocessing.

    Inputs:
      - input_batch: (N,H,W) or (N,1,H,W), metric depth by default.
      - depth_scale_m: if not None, rescales normalized depth to metric depth before DeFM transforms.

    Returns:
      - (N,3,H_out,W_out), float32.
    """
    if isinstance(input_batch, np.ndarray):
        depth = torch.from_numpy(input_batch)
    elif isinstance(input_batch, torch.Tensor):
        depth = input_batch
    else:
        raise TypeError("input_batch must be a numpy array or torch.Tensor")

    depth = depth.to(dtype=torch.float32)

    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    elif depth.ndim == 4:
        if depth.shape[1] != 1:
            depth = depth[:, :1, :, :]
    else:
        raise ValueError("input_batch must be shape (N,H,W) or (N,1,H,W)")

    n, _, h, w = depth.shape

    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    # AirGym depth is normalized to [0,1]; recover metric depth for DeFM preprocessing.
    if depth_scale_m is not None and depth_scale_m > 0:
        depth = depth * depth_scale_m

    depth = torch.clamp(depth, min=0.0, max=max_depth_c1)
    log_depth = torch.log1p(depth)

    denom_c1 = math.log1p(max_depth_c1)
    denom_c2 = math.log1p(max_depth_c2)

    c1 = log_depth / denom_c1
    c2 = torch.clamp(log_depth / denom_c2, min=0.0, max=1.0)

    log_flat = log_depth.view(n, -1)
    min_log = log_flat.min(dim=1)[0].view(n, 1, 1, 1)
    max_log = log_flat.max(dim=1)[0].view(n, 1, 1, 1)
    denom = max_log - min_log
    denom_safe = torch.where(denom > 0.0, denom, torch.ones_like(denom))
    c3 = (log_depth - min_log) / denom_safe
    c3 = torch.where(denom > 0.0, c3, torch.zeros_like(c3))

    x = torch.cat([c1, c2, c3], dim=1)

    out_h, out_w = get_target_size((h, w), target_size=target_size, patch_size=patch_size)
    if (out_h, out_w) != (h, w):
        if interpolation_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            x = F.interpolate(
                x,
                size=(out_h, out_w),
                mode=interpolation_mode,
                align_corners=False,
            )
        else:
            x = F.interpolate(x, size=(out_h, out_w), mode=interpolation_mode)

    mean = torch.tensor(DEFM_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(DEFM_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std

    return x


def _to_single_depth_tensor(input_data: Union[np.ndarray, torch.Tensor, "PILImageType"]) -> torch.Tensor:
    if isinstance(input_data, torch.Tensor):
        tensor = input_data.detach().to(dtype=torch.float32)
        if tensor.ndim == 2:
            return tensor
        if tensor.ndim == 3:
            if tensor.shape[0] in (1, 3):
                return tensor[0]
            if tensor.shape[-1] in (1, 3):
                return tensor[..., 0]
        raise ValueError(f"Unsupported torch input shape for single depth image: {tuple(tensor.shape)}")

    if PILImageType is not None and isinstance(input_data, PILImageType):
        array = np.array(input_data, dtype=np.float32)
    elif isinstance(input_data, np.ndarray):
        array = input_data.astype(np.float32, copy=False)
    else:
        raise TypeError("input_data must be numpy array, torch tensor, or PIL image")

    if array.ndim == 2:
        return torch.from_numpy(array)
    if array.ndim == 3:
        return torch.from_numpy(array[..., 0])
    raise ValueError(f"Unsupported numpy/PIL input shape for single depth image: {array.shape}")


def preprocess_depth_image(
    input_data: Union[np.ndarray, torch.Tensor, "PILImageType"],
    target_size: Optional[Union[int, Tuple[int, int], list]] = None,
    patch_size: Optional[int] = None,
    max_depth_c1: float = 100.0,
    max_depth_c2: float = 9.0,
    interpolation_mode: str = "bilinear",
) -> torch.Tensor:
    """
    DeFM-compatible single-image preprocessing.

    Args:
      input_data: Metric depth image in meters (numpy/tensor/PIL), shape (H,W) or single-channel 3D.
      target_size: int or (H,W). If set with patch_size, output will be patch-aligned.
      patch_size: patch alignment divisor (e.g., 14 for ViT-S/14).

    Returns:
      torch.Tensor with shape (1, 3, H_out, W_out).
    """
    depth = _to_single_depth_tensor(input_data)
    depth = depth.unsqueeze(0).unsqueeze(0)
    return preprocess_depth_batch(
        depth,
        target_size=target_size,
        patch_size=patch_size,
        max_depth_c1=max_depth_c1,
        max_depth_c2=max_depth_c2,
        depth_scale_m=None,
        interpolation_mode=interpolation_mode,
    )
