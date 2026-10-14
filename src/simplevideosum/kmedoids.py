"""k-medoids clustering of per-frame signatures, as in the original videosum package.

The frames are compared with ``distances.distance_matrix``. The medoids are initialised with
k-medoids++ and refined with PAM swaps (FastPAM1 formulation, vectorised, so all the swaps are
evaluated in O(N^2) per iteration).
"""

import math

import numpy as np
import torch

from simplevideosum.distances import distance_matrix


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
        cumulative = np.cumsum(closest.cpu().double().numpy())  # float64: not on MPS
        picks = np.searchsorted(cumulative, rng.random(n_trials) * total)
        picks = np.clip(picks, 0, n - 1)
        potentials = torch.minimum(
            closest[None, :], dist[torch.as_tensor(picks, device=dist.device)]
        )
        potentials = potentials.sum(dim=1)
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
    medoids = torch.as_tensor(medoids, dtype=torch.long, device=dist.device).clone()
    if k >= n:
        return medoids.cpu().numpy()
    for _ in range(max_iter):
        to_medoids = dist[:, medoids]  # (N, k)
        if k > 1:
            two_nearest, nearest = to_medoids.topk(2, dim=1, largest=False)
            d1, d2 = two_nearest[:, 0], two_nearest[:, 1]
            nearest = nearest[:, 0]
        else:
            d1 = to_medoids[:, 0]
            d2 = torch.full_like(d1, float("inf"))
            nearest = torch.zeros(n, dtype=torch.long, device=dist.device)
        membership = torch.nn.functional.one_hot(nearest, k).to(dist.dtype)  # (N, k)

        best_delta, best_swap = 0.0, None
        for start in range(0, n, chunk):
            rows = dist[start : start + chunk]  # (c, N): distance from candidate h to every p
            gain = (rows - d1[None, :]).clamp(max=0)
            # points whose nearest medoid is removed move to h or to their second nearest
            loss = torch.minimum(rows, d2[None, :]) - d1[None, :] - gain
            delta = gain.sum(dim=1, keepdim=True) + loss @ membership  # (c, k)
            delta[
                torch.isin(torch.arange(start, start + len(rows), device=dist.device), medoids)
            ] = float("inf")
            value, flat = delta.flatten().min(dim=0)
            if float(value) < best_delta:
                h, i = divmod(int(flat), k)
                best_delta, best_swap = float(value), (start + h, i)
        if best_swap is None or best_delta > -tol:
            break
        medoids[best_swap[1]] = best_swap[0]
    return medoids.cpu().numpy()


def kmedoids(
    signatures: torch.Tensor,
    k: int,
    time_smoothing: float = 0.0,
    seed: int = 0,
    metric: str = "l2sq",
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster the frames into ``k`` clusters around medoid frames.

    ``signatures`` is (N, D), on any device; the distances and the PAM swaps run on it.

    Returns ``(indices, labels)``: the medoid frame indices sorted in time, and for every frame
    the position in ``indices`` of the medoid of its cluster. Clusters are not necessarily
    contiguous in time (unless ``time_smoothing`` is 1).
    """
    n = len(signatures)
    if k < 1:
        raise ValueError("k must be >= 1")
    if n <= k:  # nothing to cluster: every frame is its own cluster
        return np.arange(n), np.arange(n)
    dist = distance_matrix(signatures, time_smoothing, metric)
    medoids = pam(dist, kmedoids_pp_init(dist, k, np.random.default_rng(seed)))
    medoids = np.sort(medoids)
    labels = dist[:, torch.as_tensor(medoids, device=dist.device)].argmin(dim=1).cpu().numpy()
    return medoids, labels
