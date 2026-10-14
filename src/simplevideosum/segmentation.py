"""Temporally contiguous segmentation of per-frame signatures.

The cost of a segment is the sum of squared distances from its signatures to their mean.
Prefix sums of the signatures and of their squared norms give this cost in O(D) for any
segment, instead of O(length * D). Segments are half-open index ranges ``[start, end)``.
``dissect`` cuts greedily by recursive bisection; ``fisher_segments`` finds the optimal cuts.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CumulativeStats:
    """Prefix sums of signatures and of their squared norms, with a leading zero row."""

    sums: torch.Tensor  # (N + 1, D)
    squares: torch.Tensor  # (N + 1,)

    @classmethod
    def from_signatures(
        cls, signatures: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> "CumulativeStats":
        # The cost is translation invariant: centring the signatures keeps the prefix sums
        # small, which is what makes float32 accurate enough (see the tests).
        x = torch.as_tensor(signatures, dtype=dtype)
        x = x - x.mean(dim=0)
        sums = torch.cat([x.new_zeros(1, x.shape[1]), x.cumsum(dim=0)])
        squares = torch.cat([x.new_zeros(1), (x * x).sum(dim=1).cumsum(dim=0)])
        return cls(sums, squares)

    def cost(self, starts: torch.Tensor | int, ends: torch.Tensor | int) -> torch.Tensor:
        """Cost of the segments ``[starts, ends)`` (broadcast): sum ||x - mean||^2."""
        device = self.sums.device
        starts, ends = torch.as_tensor(starts, device=device), torch.as_tensor(ends, device=device)
        total = self.sums[ends] - self.sums[starts]
        energy = self.squares[ends] - self.squares[starts]
        length = (ends - starts).to(total.dtype)
        return (energy - (total * total).sum(dim=-1) / length).clamp(min=0)


def is_power_of_two(n: int) -> bool:
    return n >= 1 and n & (n - 1) == 0


def dissect(
    signatures: torch.Tensor, n_segments: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Split the frames into ``n_segments`` contiguous segments by recursive bisection.

    Each segment is split where the sum of the costs of the two halves is smallest, until
    ``n_segments`` (a power of two) segments are obtained. The costs are evaluated on the device
    of the signatures, with one host synchronisation per cut. Returns the ``n_segments + 1``
    boundaries ``b`` on that device: segment ``k`` is ``[b[k], b[k + 1])``.
    """
    if not is_power_of_two(n_segments):
        raise ValueError(f"the number of segments must be a power of 2, got {n_segments}")
    signatures = torch.as_tensor(signatures)
    n_frames = len(signatures)
    if n_frames < n_segments:
        raise ValueError(f"cannot split {n_frames} frames into {n_segments} segments")
    stats = CumulativeStats.from_signatures(signatures, dtype)

    def split(start: int, end: int, n: int) -> list[int]:
        if n == 1:
            return [start]
        half = n // 2  # each side keeps enough frames for its own segments
        candidates = torch.arange(start + half, end - half + 1, device=signatures.device)
        costs = stats.cost(start, candidates) + stats.cost(candidates, end)
        mid = start + half + int(costs.argmin())
        return split(start, mid, half) + split(mid, end, half)

    return torch.tensor(split(0, n_frames, n_segments) + [n_frames], device=signatures.device)


# Rows of the cost matrix combined at once by the dynamic programme: bounds its temporary memory.
_ROW_CHUNK = 1024


def fisher_segments(
    signatures: torch.Tensor, n_segments: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Split the frames into ``n_segments`` contiguous segments with the least total cost.

    Fisher's exact 1-D segmentation (Bellman's dynamic programme): ``F[j][e]``, the least cost
    of splitting the first ``e`` frames into ``j`` segments, is ``min_s F[j-1][s] + cost(s, e)``.
    The segments can have any length and ``n_segments`` any value. The (N + 1) x (N + 1) cost
    matrix is held on the device, so the memory grows as N^2 and the time as n_segments * N^2.
    Returns the ``n_segments + 1`` boundaries ``b`` on that device: segment ``k`` is
    ``[b[k], b[k + 1])``.
    """
    signatures = torch.as_tensor(signatures)
    n_frames = len(signatures)
    if n_segments < 1:
        raise ValueError(f"the number of segments must be >= 1, got {n_segments}")
    if n_frames < n_segments:
        raise ValueError(f"cannot split {n_frames} frames into {n_segments} segments")
    stats = CumulativeStats.from_signatures(signatures, dtype)
    device = stats.sums.device
    inf = torch.tensor(float("inf"), dtype=dtype, device=device)

    # cost[s, e] for s < e: energy - ||sum||^2 / length. The pairwise distances between prefix
    # sums are computed from the differences, as an expansion of the square loses precision.
    positions = torch.arange(n_frames + 1, device=device)
    span = positions[None, :] - positions[:, None]
    sums_sq = torch.cdist(stats.sums, stats.sums, compute_mode="donot_use_mm_for_euclid_dist") ** 2
    cost = stats.squares[None, :] - stats.squares[:, None] - sums_sq / span.clamp(min=1)
    cost = torch.where(span > 0, cost.clamp(min=0), inf)
    del span, sums_sq

    best = torch.where(positions == 0, 0.0, float("inf")).to(dtype)  # F[0]
    origins = []
    for _ in range(n_segments):
        new_best = torch.full_like(best, float("inf"))
        origin = torch.zeros(n_frames + 1, dtype=torch.long, device=device)
        for start in range(0, n_frames + 1, _ROW_CHUNK):
            rows = slice(start, start + _ROW_CHUNK)
            value, arg = (best[rows, None] + cost[rows]).min(dim=0)
            better = value < new_best
            new_best = torch.where(better, value, new_best)
            origin = torch.where(better, arg + start, origin)
        best = new_best
        origins.append(origin)

    origins = torch.stack(origins).tolist()  # a single host synchronisation
    bounds = [n_frames]
    for origin in reversed(origins):
        bounds.append(origin[bounds[-1]])
    return torch.tensor(bounds[::-1], device=device)
