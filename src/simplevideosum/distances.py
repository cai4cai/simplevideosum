"""Distances between per-frame signatures: squared L2 or cosine, optionally smoothed in time."""

import torch
import torch.nn.functional as F

DISTANCES = ("l2sq", "cosine")


def distance_matrix(
    signatures: torch.Tensor, time_smoothing: float = 0.0, metric: str = "l2sq"
) -> torch.Tensor:
    """(N, N) float32 matrix of normalised signature distances, blended with temporal distance.

    ``signatures`` is (N, D), a tensor or anything ``torch.as_tensor`` accepts, such as a NumPy
    array. The matrix is computed, and returned, on the device of the tensor.

    ``metric`` is ``"l2sq"`` (squared L2) or ``"cosine"`` (one minus the cosine similarity).

    ``time_smoothing`` in [0, 1]: 0 uses only the signatures, 1 only the frame order (which
    makes the clusters temporally contiguous, like the ``time`` algorithm).
    """
    if not 0.0 <= time_smoothing <= 1.0:
        raise ValueError("time_smoothing must be in [0, 1]")
    if metric not in DISTANCES:
        raise ValueError(f"unknown distance '{metric}' (available: {', '.join(DISTANCES)})")
    x = torch.as_tensor(signatures, dtype=torch.float32)
    if metric == "cosine":
        x = F.normalize(x, dim=1)
        dist = (1 - x @ x.T).clamp_(min=0, max=2)
    else:
        x = x - x.mean(dim=0)  # distances are translation invariant, centring avoids cancellation
        squares = (x * x).sum(dim=1)
        dist = (squares[:, None] + squares[None, :] - 2 * x @ x.T).clamp_(min=0)
    dist.fill_diagonal_(0)
    peak = dist.max()
    if peak > 0:
        dist /= peak
    if time_smoothing > 0 and len(x) > 1:
        order = torch.arange(len(x), dtype=torch.float32, device=x.device)
        temporal = (order[:, None] - order[None, :]).abs() / (len(x) - 1)
        dist = (1 - time_smoothing) * dist + time_smoothing * temporal
    return dist
