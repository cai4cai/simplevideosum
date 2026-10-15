"""Aggregation of backbone activations and normalisation of the result, as pluggable modules.

An aggregation maps (N, C, H, W) activations to (N, P, C): P pooled descriptors per frame. A
normalisation acts on every pooled descriptor, and the P descriptors are then concatenated into
the (N, P * C) signature (so SCDA gives the paper's "avg&maxPool").
"""

from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from simplevideosum.scda import SCDA


class GlobalAveragePool(nn.Module):
    """(N, C, H, W) activations to their (N, 1, C) spatial mean."""

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return activations.mean(dim=(2, 3)).unsqueeze(1)


class GlobalAverageMaxPool(nn.Module):
    """(N, C, H, W) activations to their (N, 2, C) spatial mean and max: SCDA without the
    selection of the positions, so that it describes the whole frame."""

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return torch.stack([activations.mean(dim=(2, 3)), activations.amax(dim=(2, 3))], dim=1)


class L2Normalise(nn.Module):
    """Scale every pooled descriptor to unit L2 norm."""

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return F.normalize(pooled, dim=-1)


# SCDA variants: "scda" is the paper's (largest connected component of the mask), "scda_thresh"
# keeps every position above the threshold, and "scda_fill" also fills the holes of the mask,
# which the paper does not do but the videosum package does.
AGGREGATIONS: dict[str, Callable[[], nn.Module]] = {
    "avgpool": GlobalAveragePool,
    "avgcatmaxpool": GlobalAverageMaxPool,
    "scda": SCDA,
    "scda_thresh": partial(SCDA, largest_component=False),
    "scda_fill": partial(SCDA, fill_holes=True),
}
NORMALISATIONS: dict[str, Callable[[], nn.Module]] = {"none": nn.Identity, "l2": L2Normalise}
