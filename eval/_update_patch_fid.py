# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import torch
import torch.nn.functional as F
import torchmetrics.image as metim
from torch import Tensor

from ._fid import FrechetInceptionDistance
from ._fid_swav import FrechetInceptionDistanceSwAV
from ._kid import KernelInceptionDistance


def image_to_255_scale(image: Tensor, dtype: Optional[torch.dtype] = None) -> Tensor:
    """
    Helper function for converting a floating point image to 255 scale.

    The input image is expected to be in the range [0.0, 1.0]. If it is outside
    this range, the function throws an error.

    Args:
        image: A 4-D PyTorch tensor.
        dtype: Output datatype. If not passed, the image is of the same dtype
            as the input.

    Returns:
        The image in [0, 255] scale.
    """
    if image.max() > 1.0:
        raise ValueError("Unexpected image max > 1.0")
    if image.min() < 0.0:
        raise ValueError("Unexpected image min < 0.0")

    image = torch.round(image * 255.0)

    if dtype is not None:
        image = image.to(dtype)

    return image

def update_patch_fid(
    input_images: Tensor,
    pred: Tensor,
    fid_metric: Optional[
        Union[metim.FrechetInceptionDistance, FrechetInceptionDistance]
    ] = None,
    fid_swav_metric: Optional[FrechetInceptionDistanceSwAV] = None,
    kid_metric: Optional[
        Union[metim.KernelInceptionDistance, KernelInceptionDistance]
    ] = None,
    patch_size: int = 256,
) -> int:
    """
    Update FID and KID metrics with patch-based calculation.

    This implements the FID/256 (and KID/256) method described in the following
    paper:

    High-Fidelity Generative Image Compression
    Fabian Mentzer, George D. Toderici, Michael Tschannen, Eirikur Agustsson

    First, a user defines a torchmetric class for FID or KID. Then, this
    function can be used to update the metrics using the FID/256 calculation.

    This method gives more stable FID calculations for small counts of
    high-resolution images, such as those in the CLIC2020 dataset. Each image
    is divided up into a grid of non-overlapping patches, and the normal FID
    calculation is run treating each patch as an image. Then, the calculation
    is re-run a second time with a patch/2 shift.

    Args:
        input_images: The ground truth images in [0.0, 1.0] range.
        pred: The compressed images in [0.0, 1.0] range.
        fid_metric: A torchmetric for calculationg FID.
        fid_swav_metric: A torchmetric for calculating FID with the SwAV
            backbone.
        kid_metric: A torchmetric for calculating KID.
        patch_size: The patch size to use for dividing up each image.

    Returns:
        The number of patches (metric are updated in-place). The number of
        patches can be used as debugging signal.
    """
    if fid_metric is None and kid_metric is None and fid_swav_metric is None:
        raise ValueError("At least one metric must not be None.")

    # this applies the FID/KID calculations from Mentzer 2020
    real = image_to_255_scale(
        F.unfold(input_images, kernel_size=patch_size, stride=patch_size)
        .permute(0, 2, 1)
        .reshape(-1, 3, patch_size, patch_size),
        dtype=torch.uint8,
    )
    fake = image_to_255_scale(
        F.unfold(pred, kernel_size=patch_size, stride=patch_size)
        .permute(0, 2, 1)
        .reshape(-1, 3, patch_size, patch_size),
        dtype=torch.uint8,
    )
    patch_count = real.shape[0]
    if fid_metric is not None:
        fid_metric.update(real, real=True)
        fid_metric.update(fake, real=False)
    if fid_swav_metric is not None:
        fid_swav_metric.update(real, real=True)
        fid_swav_metric.update(fake, real=False)
    if kid_metric is not None:
        kid_metric.update(real, real=True)
        kid_metric.update(fake, real=False)

    num_y, num_x = input_images.shape[2], input_images.shape[3]
    if num_y >= 1.5 * patch_size and num_x >= 1.5 * patch_size:
        real = image_to_255_scale(
            F.unfold(
                input_images[:, :, patch_size // 2 :, patch_size // 2 :],
                kernel_size=patch_size,
                stride=patch_size,
            )
            .permute(0, 2, 1)
            .reshape(-1, 3, patch_size, patch_size),
            dtype=torch.uint8,
        )
        fake = image_to_255_scale(
            F.unfold(
                pred[:, :, patch_size // 2 :, patch_size // 2 :],
                kernel_size=patch_size,
                stride=patch_size,
            )
            .permute(0, 2, 1)
            .reshape(-1, 3, patch_size, patch_size),
            dtype=torch.uint8,
        )
        patch_count += real.shape[0]
        if fid_metric is not None:
            fid_metric.update(real, real=True)
            fid_metric.update(fake, real=False)
        if fid_swav_metric is not None:
            fid_swav_metric.update(real, real=True)
            fid_swav_metric.update(fake, real=False)
        if kid_metric is not None:
            kid_metric.update(real, real=True)
            kid_metric.update(fake, real=False)

    return patch_count