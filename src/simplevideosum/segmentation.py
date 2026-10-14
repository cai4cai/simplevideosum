"""Temporally contiguous segmentation of per-frame signatures.

The cost of a segment is the sum of squared distances from its signatures to their mean.
Prefix sums of the signatures and of their squared norms give this cost in O(D) for any
segment, instead of O(length * D), so no quadratic pass over the frames is needed.
Segments are half-open index ranges ``[start, end)``.
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
