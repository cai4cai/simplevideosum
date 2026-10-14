"""Selective Convolutional Descriptor Aggregation (Wei et al., 2017), batched in PyTorch.

SCDA turns the last convolutional activations of a frame into one descriptor that mostly describes
the main object: the activations are summed over channels, the positions above the mean form a
mask, only the largest (8-connected) component of that mask is kept, and the descriptors at the
selected positions are average- and max-pooled. A whole batch of frames is handled at once, on
the device of the activations (see ``morphology``).
"""

import torch
from torch import nn

from simplevideosum.morphology import fill_holes, largest_connected_component


class SCDA(nn.Module):
    """Maps (N, C, H, W) convolutional activations to (N, 2, C) pooled SCDA descriptors.

    Row 0 is the average and row 1 the max of the descriptors at the selected positions; the
    paper L2-normalises each of them before concatenating them (see ``aggregation``).
    """

    def __init__(self, largest_component: bool = True, fill_holes: bool = False) -> None:
        """``largest_component`` keeps only the largest connected component of the mask, as in
        the paper, instead of all the positions above the mean. ``fill_holes`` also fills the holes
        of the mask, which the paper does not do but videosum does."""
        super().__init__()
        self.largest_component = largest_component
        self.fill_holes = fill_holes

    def selection_mask(self, activations: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) boolean mask of the selected positions of (B, C, H, W) activations."""
        aggregation = activations.sum(dim=1, keepdim=True)
        mask = aggregation > aggregation.mean(dim=(2, 3), keepdim=True)
        mask = mask | ~mask.flatten(1).any(dim=1).view(-1, 1, 1, 1)  # flat map: keep everything
        if self.largest_component:
            mask = largest_connected_component(mask)
        return fill_holes(mask) if self.fill_holes else mask

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        if activations.ndim != 4:
            raise ValueError(f"expected (B, C, H, W) activations, got {tuple(activations.shape)}")
        mask = self.selection_mask(activations)
        average = (activations * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3))
        maximum = activations.masked_fill(~mask, float("-inf")).amax(dim=(2, 3))
        return torch.stack([average, maximum], dim=1)
