"""Aggregation of backbone activations and normalisation of the result, as pluggable modules.

An aggregation maps (N, C, H, W) activations to (N, P, C): P pooled descriptors per frame. A
normalisation acts on every pooled descriptor, and the P descriptors are then concatenated into
the (N, P * C) signature (so SCDA gives the paper's "avg&maxPool").
"""

import torch
import torch.nn.functional as F
from torch import nn

from simplevideosum.scda import SCDA


class GlobalAveragePool(nn.Module):
    """(N, C, H, W) activations to their (N, 1, C) spatial mean."""

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return activations.mean(dim=(2, 3)).unsqueeze(1)


class L2Normalise(nn.Module):
    """Scale every pooled descriptor to unit L2 norm."""

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return F.normalize(pooled, dim=-1)


AGGREGATIONS: dict[str, type[nn.Module]] = {"avgpool": GlobalAveragePool, "scda": SCDA}
NORMALISATIONS: dict[str, type[nn.Module]] = {"none": nn.Identity, "l2": L2Normalise}
