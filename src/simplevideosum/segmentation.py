"""Temporally contiguous segmentation of per-frame signatures.

The cost of a segment is the sum of squared distances from its signatures to their mean.
Prefix sums of the signatures and of their squared norms give this cost in O(D) for any
segment, instead of O(length * D), so no quadratic pass over the frames is needed.
Segments are half-open index ranges ``[start, end)``.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CumulativeStats:
    """Prefix sums of signatures and of their squared norms, with a leading zero row."""

    sums: np.ndarray  # (N + 1, D)
    squares: np.ndarray  # (N + 1,)

    @classmethod
    def from_signatures(cls, signatures: np.ndarray, dtype=np.float32) -> "CumulativeStats":
        # The cost is translation invariant: centring the signatures keeps the prefix sums
        # small, which is what makes float32 accurate enough (see the tests).
        x = np.asarray(signatures, dtype=dtype)
        x = x - x.mean(axis=0)
        sums = np.zeros((len(x) + 1, x.shape[1]), dtype=dtype)
        np.cumsum(x, axis=0, out=sums[1:])
        squares = np.zeros(len(x) + 1, dtype=dtype)
        np.cumsum((x * x).sum(axis=1), out=squares[1:])
        return cls(sums, squares)

    def cost(self, starts: np.ndarray | int, ends: np.ndarray | int) -> np.ndarray:
        """Cost of the segments ``[starts, ends)`` (broadcast): sum ||x - mean||^2."""
        starts, ends = np.asarray(starts), np.asarray(ends)
        total = self.sums[ends] - self.sums[starts]
        energy = self.squares[ends] - self.squares[starts]
        length = (ends - starts).astype(total.dtype)
        return np.maximum(energy - (total * total).sum(axis=-1) / length, 0)


def is_power_of_two(n: int) -> bool:
    return n >= 1 and n & (n - 1) == 0


def dissect(signatures: np.ndarray, n_segments: int, dtype=np.float32) -> np.ndarray:
    """Split the frames into ``n_segments`` contiguous segments by recursive bisection.

    Each segment is split where the sum of the costs of the two halves is smallest, until
    ``n_segments`` (a power of two) segments are obtained. Returns the ``n_segments + 1``
    boundaries ``b``: segment ``k`` is ``[b[k], b[k + 1])``.
    """
    if not is_power_of_two(n_segments):
        raise ValueError(f"the number of segments must be a power of 2, got {n_segments}")
    n_frames = len(signatures)
    if n_frames < n_segments:
        raise ValueError(f"cannot split {n_frames} frames into {n_segments} segments")
    stats = CumulativeStats.from_signatures(signatures, dtype)

    def split(start: int, end: int, n: int) -> list[int]:
        if n == 1:
            return [start]
        half = n // 2  # each side keeps enough frames for its own segments
        candidates = np.arange(start + half, end - half + 1)
        costs = stats.cost(start, candidates) + stats.cost(candidates, end)
        mid = int(candidates[np.argmin(costs)])
        return split(start, mid, half) + split(mid, end, half)

    return np.array(split(0, n_frames, n_segments) + [n_frames])
