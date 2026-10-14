"""k-medoids clustering of per-frame signatures, as in the original videosum package.

Distances are squared L2 distances between signatures, min-max normalised, optionally blended
with the normalised temporal distance between frames (time smoothing). The medoids are
initialised with k-medoids++ and refined with PAM swaps (FastPAM1 formulation, vectorised, so
all the swaps are evaluated in O(N^2) per iteration).
"""

import math

import numpy as np
import torch


def distance_matrix(signatures: np.ndarray, time_smoothing: float = 0.0) -> torch.Tensor:
    """(N, N) float32 matrix of normalised squared L2 distances, blended with temporal distance.

    ``time_smoothing`` in [0, 1]: 0 uses only the signatures, 1 only the frame order (which
    makes the clusters temporally contiguous, like the ``time`` algorithm).
    """
    if not 0.0 <= time_smoothing <= 1.0:
        raise ValueError("time_smoothing must be in [0, 1]")
    x = torch.as_tensor(np.asarray(signatures, dtype=np.float32))
    x = x - x.mean(dim=0)  # distances are translation invariant, centring avoids cancellation
    squares = (x * x).sum(dim=1)
    dist = (squares[:, None] + squares[None, :] - 2 * x @ x.T).clamp_(min=0)
    dist.fill_diagonal_(0)
    peak = dist.max()
    if peak > 0:
        dist /= peak
    if time_smoothing > 0 and len(x) > 1:
        order = torch.arange(len(x), dtype=torch.float32)
        temporal = (order[:, None] - order[None, :]).abs() / (len(x) - 1)
        dist = (1 - time_smoothing) * dist + time_smoothing * temporal
    return dist


def kmedoids_pp_init(dist: torch.Tensor, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-medoids++ seeding: medoids are sampled proportionally to their distance to the
    closest medoid so far, keeping the best of a few local trials at each step."""
    n = len(dist)
    n_trials = 2 + int(math.log(k)) if k > 1 else 1
    medoids = [int(rng.integers(n))]
    closest = dist[medoids[0]].clone()
    for _ in range(1, k):
        total = float(closest.sum())
        if total <= 0:  # all the remaining frames coincide with a medoid
            unused = np.setdiff1d(np.arange(n), medoids)
            medoids.append(int(rng.choice(unused)))
            closest = torch.minimum(closest, dist[medoids[-1]])
            continue
        cumulative = np.cumsum(closest.double().numpy())
        picks = np.searchsorted(cumulative, rng.random(n_trials) * total)
        picks = np.clip(picks, 0, n - 1)
        potentials = torch.minimum(closest[None, :], dist[picks]).sum(dim=1)
        best = int(picks[int(potentials.argmin())])
        medoids.append(best)
        closest = torch.minimum(closest, dist[best])
    return np.array(medoids)


def pam(
    dist: torch.Tensor,
    medoids: np.ndarray,
    max_iter: int = 300,
    chunk: int = 1024,
    tol: float = 1e-7,
) -> np.ndarray:
    """Refine ``medoids`` by repeatedly applying the best medoid/non-medoid swap (PAM).

    The cost change of swapping medoid ``i`` for candidate ``h`` is evaluated for every pair at
    once from the nearest and second nearest medoid distances of each point. Candidates are
    processed in chunks to bound the memory of the (candidates x N) temporaries.
    """
    n, k = len(dist), len(medoids)
    medoids = torch.as_tensor(medoids, dtype=torch.long).clone()
    if k >= n:
        return medoids.numpy()
    for _ in range(max_iter):
        to_medoids = dist[:, medoids]  # (N, k)
        if k > 1:
            two_nearest, nearest = to_medoids.topk(2, dim=1, largest=False)
            d1, d2 = two_nearest[:, 0], two_nearest[:, 1]
            nearest = nearest[:, 0]
        else:
            d1 = to_medoids[:, 0]
            d2 = torch.full_like(d1, float("inf"))
            nearest = torch.zeros(n, dtype=torch.long)
        membership = torch.nn.functional.one_hot(nearest, k).to(dist.dtype)  # (N, k)

        best_delta, best_swap = 0.0, None
        for start in range(0, n, chunk):
            rows = dist[start : start + chunk]  # (c, N): distance from candidate h to every p
            gain = (rows - d1[None, :]).clamp(max=0)
            # points whose nearest medoid is removed move to h or to their second nearest
            loss = torch.minimum(rows, d2[None, :]) - d1[None, :] - gain
            delta = gain.sum(dim=1, keepdim=True) + loss @ membership  # (c, k)
            delta[torch.isin(torch.arange(start, start + len(rows)), medoids)] = float("inf")
            value, flat = delta.flatten().min(dim=0)
            if float(value) < best_delta:
                h, i = divmod(int(flat), k)
                best_delta, best_swap = float(value), (start + h, i)
        if best_swap is None or best_delta > -tol:
            break
        medoids[best_swap[1]] = best_swap[0]
    return medoids.numpy()


def kmedoids(
    signatures: np.ndarray, k: int, time_smoothing: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster the frames into ``k`` clusters around medoid frames.

    Returns ``(indices, labels)``: the medoid frame indices sorted in time, and for every frame
    the position in ``indices`` of the medoid of its cluster. Clusters are not necessarily
    contiguous in time (unless ``time_smoothing`` is 1).
    """
    n = len(signatures)
    if k < 1:
        raise ValueError("k must be >= 1")
    if n <= k:  # nothing to cluster: every frame is its own cluster
        return np.arange(n), np.arange(n)
    dist = distance_matrix(signatures, time_smoothing)
    medoids = pam(dist, kmedoids_pp_init(dist, k, np.random.default_rng(seed)))
    medoids = np.sort(medoids)
    labels = dist[:, torch.as_tensor(medoids)].argmin(dim=1).numpy()
    return medoids, labels
