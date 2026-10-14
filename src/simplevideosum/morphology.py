"""Binary mathematical morphology on batched masks, in pure PyTorch (CPU, CUDA and MPS).

Masks are boolean (B, C, H, W) tensors and every (b, c) map is processed independently.
``connectivity`` is 4 (edge neighbours) or 8 (edge and corner neighbours) of the foreground.
"""

import math
from functools import lru_cache

import torch
import torch.nn.functional as F

# Largest map (in pixels) whose connected components are found by transitive closure, which
# needs a (H*W, H*W) matrix per map; larger maps use label propagation.
CLOSURE_MAX_PIXELS = 256


def _check(mask: torch.Tensor, connectivity: int) -> None:
    if mask.ndim != 4 or mask.dtype != torch.bool:
        raise ValueError(
            f"expected a boolean (B, C, H, W) mask, got {mask.dtype} {tuple(mask.shape)}"
        )
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")


def _neighbourhood_max(x: torch.Tensor, connectivity: int) -> torch.Tensor:
    """Maximum of each pixel and its neighbours in a (B, C, H, W) float tensor."""
    if connectivity == 8:
        return F.max_pool2d(x, 3, stride=1, padding=1)
    return torch.maximum(
        F.max_pool2d(x, (1, 3), stride=1, padding=(0, 1)),
        F.max_pool2d(x, (3, 1), stride=1, padding=(1, 0)),
    )


def dilate(mask: torch.Tensor, connectivity: int = 8) -> torch.Tensor:
    """Grow every foreground pixel of the (B, C, H, W) boolean mask to its neighbours."""
    _check(mask, connectivity)
    return _neighbourhood_max(mask.float(), connectivity) > 0


@lru_cache(maxsize=16)
def _adjacency(h: int, w: int, connectivity: int, device: torch.device) -> torch.Tensor:
    """(H*W, H*W) float matrix, 1 where two pixels are neighbours or the same pixel."""
    eye = torch.eye(h * w, device=device).view(h * w, 1, h, w)
    return _neighbourhood_max(eye, connectivity).view(h * w, h * w)


def _largest_by_closure(mask: torch.Tensor, connectivity: int) -> torch.Tensor:
    """Transitive closure of the pixel adjacency by repeated squaring, as batched matmuls.

    Entry (i, j) of the closure is 1 when pixels i and j are in the same component, so a row sum
    is the size of the component of a pixel. There is no data-dependent loop or host sync.
    """
    b, c, h, w = mask.shape
    hw = h * w
    flat = mask.reshape(b * c, hw).to(torch.float32)
    reach = flat[:, :, None] * flat[:, None, :] * _adjacency(h, w, connectivity, mask.device)
    for _ in range(math.ceil(math.log2(hw)) if hw > 1 else 0):  # paths of up to 2**k steps
        reach = (reach @ reach).clamp_(max=1)
    best = reach.sum(dim=2).argmax(dim=1)  # first pixel of the largest component
    chosen = reach[torch.arange(len(reach), device=mask.device), best] > 0
    return chosen.view(b, c, h, w)


def _largest_by_propagation(mask: torch.Tensor, connectivity: int) -> torch.Tensor:
    """Label propagation: every pixel takes the largest label of its neighbours until stable.

    Pixels are labelled from H*W down to 1 in raster order, so a component ends up labelled by
    its first pixel. The iterations stop on convergence, which synchronises with the host.
    """
    b, c, h, w = mask.shape
    hw = h * w
    foreground = mask.reshape(b * c, 1, h, w)
    ids = torch.arange(hw, 0, -1, device=mask.device, dtype=torch.float32).view(1, 1, h, w)
    labels = ids * foreground
    while True:
        spread = _neighbourhood_max(labels, connectivity) * foreground
        if torch.equal(spread, labels):
            break
        labels = spread
    labels = labels.flatten(1).long()  # 0 is the background
    sizes = torch.zeros(len(labels), hw + 1, device=mask.device, dtype=torch.long)
    sizes.scatter_add_(1, labels, torch.ones_like(labels))
    sizes[:, 0] = 0
    best = hw - sizes.flip(1).argmax(dim=1, keepdim=True)  # the largest label among the biggest
    return ((labels == best) & (labels > 0)).view(b, c, h, w)


def largest_connected_component(
    mask: torch.Tensor, connectivity: int = 8, closure_max_pixels: int = CLOSURE_MAX_PIXELS
) -> torch.Tensor:
    """Largest connected component of each map of the (B, C, H, W) boolean mask.

    Ties go to the component whose first pixel comes first in raster order; an empty map gives an
    empty map. Maps of up to ``closure_max_pixels`` pixels are handled by transitive closure,
    larger ones by label propagation; both give the same result.
    """
    _check(mask, connectivity)
    if mask.shape[2] * mask.shape[3] <= closure_max_pixels:
        return _largest_by_closure(mask, connectivity)
    return _largest_by_propagation(mask, connectivity)


def fill_holes(mask: torch.Tensor, connectivity: int = 8) -> torch.Tensor:
    """Add to every map of the (B, C, H, W) boolean mask the background that is enclosed by it.

    The background is what is not connected to the border through the complementary
    connectivity (4 for an 8-connected foreground, 8 for a 4-connected one).
    """
    _check(mask, connectivity)
    background = ~mask
    border = torch.zeros_like(mask)
    border[..., 0, :] = border[..., -1, :] = border[..., :, 0] = border[..., :, -1] = True
    outside = border & background
    while True:
        grown = dilate(outside, 12 - connectivity) & background
        if torch.equal(grown, outside):
            return ~outside
        outside = grown
