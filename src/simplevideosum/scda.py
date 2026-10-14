"""Selective Convolutional Descriptor Aggregation (Wei et al., 2017), batched in PyTorch.

SCDA turns the last convolutional activations of a frame into one descriptor that mostly describes
the main object: the activations are summed over channels, the positions above the mean form a
mask, only the largest connected component of that mask is kept, and the descriptors at the
selected positions are average- and max-pooled. As in the reference ``videosum`` package, the
mask is 8-connected and its holes are filled. The connected component analysis runs as label
propagation on the (tiny) feature map grid, so a whole batch of frames is handled at once on the
device of the activations.
"""

import torch
import torch.nn.functional as F
from torch import nn


def dilate(mask: torch.Tensor, eight_connected: bool) -> torch.Tensor:
    """Grow the non-zero (positive) values of (N, 1, H, W) ``mask`` to their neighbours (max)."""
    if eight_connected:
        return F.max_pool2d(mask, 3, stride=1, padding=1)
    return torch.maximum(
        F.max_pool2d(mask, (1, 3), stride=1, padding=(0, 1)),
        F.max_pool2d(mask, (3, 1), stride=1, padding=(1, 0)),
    )


def largest_component(mask: torch.Tensor) -> torch.Tensor:
    """Largest 8-connected component of each (N, H, W) boolean mask (all False if it is empty).

    Components are labelled by propagating the largest pixel index of the component to all its
    pixels. Ties in size go to the component whose label is smallest.
    """
    n, h, w = mask.shape
    ids = torch.arange(1, h * w + 1, device=mask.device, dtype=torch.float32).view(1, 1, h, w)
    labels = ids * mask.unsqueeze(1)
    while True:
        propagated = dilate(labels, eight_connected=True) * mask.unsqueeze(1)
        if torch.equal(propagated, labels):
            break
        labels = propagated
    labels = labels.flatten(1).long()  # (N, H*W), 0 is the background
    sizes = torch.zeros(n, h * w + 1, device=mask.device, dtype=torch.long)
    sizes.scatter_add_(1, labels, torch.ones_like(labels))
    sizes[:, 0] = 0
    best = sizes.argmax(dim=1, keepdim=True)
    return ((labels == best) & (labels > 0)).view(n, h, w)


def fill_holes(mask: torch.Tensor) -> torch.Tensor:
    """Fill the holes of (N, H, W) boolean masks: background not 4-connected to the border."""
    n, h, w = mask.shape
    background = (~mask).unsqueeze(1)
    border = torch.zeros(1, 1, h, w, dtype=torch.bool, device=mask.device)
    border[..., 0, :] = border[..., -1, :] = border[..., :, 0] = border[..., :, -1] = True
    outside = border & background
    while True:
        grown = dilate(outside.float(), eight_connected=False).bool() & background
        if torch.equal(grown, outside):
            break
        outside = grown
    return mask | (background & ~outside).squeeze(1)


class SCDA(nn.Module):
    """Maps (N, C, H, W) convolutional activations to (N, 2, C) pooled SCDA descriptors.

    Row 0 is the average and row 1 the max of the descriptors at the selected positions; the
    paper L2-normalises each of them before concatenating them (see ``aggregation``).
    """

    def __init__(self, holes: bool = True) -> None:
        super().__init__()
        self.holes = holes

    def selection_mask(self, activations: torch.Tensor) -> torch.Tensor:
        """(N, H, W) boolean mask of the selected positions of (N, C, H, W) activations."""
        aggregation = activations.sum(dim=1)
        mask = aggregation > aggregation.mean(dim=(1, 2), keepdim=True)
        mask = mask | ~mask.flatten(1).any(dim=1).view(-1, 1, 1)  # flat map: keep every position
        mask = largest_component(mask)
        return fill_holes(mask) if self.holes else mask

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        if activations.ndim != 4:
            raise ValueError(f"expected (N, C, H, W) activations, got {tuple(activations.shape)}")
        mask = self.selection_mask(activations).unsqueeze(1)
        average = (activations * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3))
        maximum = activations.masked_fill(~mask, float("-inf")).amax(dim=(2, 3))
        return torch.stack([average, maximum], dim=1)
