"""k-medoids clustering of per-frame signatures, as in the original videosum package.

The frames are compared with ``distances.distance_matrix``. The medoids are initialised with
k-medoids++ and refined with PAM swaps (FastPAM1 formulation, vectorised, so all the swaps are
evaluated in O(N^2) per iteration).
"""

import math

import torch

from simplevideosum.distances import distance_matrix

# The (N, N) float32 distance matrix of 20,000 frames takes 1.6 GB, and PAM needs more temporaries.
MAX_FRAMES = 20_000


def kmedoids_pp_init(dist: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
    """k-medoids++ seeding: medoids are sampled proportionally to their distance to the
    closest medoid so far, keeping the best of a few local trials at each step.

    ``generator`` lives on the device of ``dist``, and nothing is copied to the host. Returns
    the (k,) medoid indices.
    """
    n, device = len(dist), dist.device
    n_trials = 2 + int(math.log(k)) if k > 1 else 1
    medoids = torch.randint(n, (1,), generator=generator, device=device)
    is_medoid = torch.zeros(n, dtype=torch.bool, device=device)
    is_medoid[medoids] = True
    closest = dist[medoids[0]].clone()
    for _ in range(1, k):
        # if all the remaining frames coincide with a medoid, any other frame will do
        weights = torch.where(closest.sum() > 0, closest, (~is_medoid).float())
        picks = torch.multinomial(weights, n_trials, replacement=True, generator=generator)
        potentials = torch.minimum(closest[None, :], dist[picks]).sum(dim=1)
        best = picks[potentials.argmin()]
        medoids = torch.cat([medoids, best[None]])
        is_medoid[best] = True
        closest = torch.minimum(closest, dist[best])
    return medoids


def pam(
    dist: torch.Tensor,
    medoids: torch.Tensor,
    max_iter: int = 300,
    chunk: int = 1024,
    tol: float = 1e-7,
) -> torch.Tensor:
    """Refine ``medoids`` by repeatedly applying the best medoid/non-medoid swap (PAM).

    The cost change of swapping medoid ``i`` for candidate ``h`` is evaluated for every pair at
    once from the nearest and second nearest medoid distances of each point. Candidates are
    processed in chunks to bound the memory of the (candidates x N) temporaries. The best swap
    is tracked on the device: the only host synchronisation is the test for convergence, once
    per iteration.
    """
    n, k = len(dist), len(medoids)
    device = dist.device
    medoids = torch.as_tensor(medoids, dtype=torch.long, device=device).clone()
    if k >= n:
        return medoids
    for _ in range(max_iter):
        to_medoids = dist[:, medoids]  # (N, k)
        if k > 1:
            two_nearest, nearest = to_medoids.topk(2, dim=1, largest=False)
            d1, d2 = two_nearest[:, 0], two_nearest[:, 1]
            nearest = nearest[:, 0]
        else:
            d1 = to_medoids[:, 0]
            d2 = torch.full_like(d1, float("inf"))
            nearest = torch.zeros(n, dtype=torch.long, device=device)
        membership = torch.nn.functional.one_hot(nearest, k).to(dist.dtype)  # (N, k)

        best_delta = torch.zeros((), dtype=dist.dtype, device=device)
        best_h = torch.zeros((), dtype=torch.long, device=device)
        best_i = torch.zeros((), dtype=torch.long, device=device)
        for start in range(0, n, chunk):
            rows = dist[start : start + chunk]  # (c, N): distance from candidate h to every p
            gain = (rows - d1[None, :]).clamp(max=0)
            # points whose nearest medoid is removed move to h or to their second nearest
            loss = torch.minimum(rows, d2[None, :]) - d1[None, :] - gain
            delta = gain.sum(dim=1, keepdim=True) + loss @ membership  # (c, k)
            candidates = torch.arange(start, start + len(rows), device=device)
            delta[torch.isin(candidates, medoids)] = float("inf")
            value, flat = delta.flatten().min(dim=0)
            better = value < best_delta
            best_delta = torch.where(better, value, best_delta)
            best_h = torch.where(better, start + flat // k, best_h)
            best_i = torch.where(better, flat % k, best_i)
        if best_delta > -tol:
            break
        medoids[best_i] = best_h
    return medoids


def kmedoids(
    signatures: torch.Tensor,
    k: int,
    time_smoothing: float = 0.0,
    seed: int = 0,
    metric: str = "l2sq",
    max_frames: int = MAX_FRAMES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster the frames into ``k`` clusters around medoid frames.

    ``signatures`` is (N, D), on any device (see ``distances.distance_matrix``); the distances,
    the seeding and the PAM swaps run on it. The result is reproducible for a given ``seed`` on
    a given device, but not across devices.

    Returns ``(indices, labels)`` on that device: the (k,) medoid frame indices sorted in time,
    and for every frame the position in ``indices`` of the medoid of its cluster. Clusters are
    not necessarily contiguous in time (unless ``time_smoothing`` is 1). More than ``max_frames``
    frames are refused, as the (N, N) distance matrix would not fit in memory.
    """
    signatures = torch.as_tensor(signatures)
    n = len(signatures)
    if k < 1:
        raise ValueError("k must be >= 1")
    if n <= k:  # nothing to cluster: every frame is its own cluster
        every = torch.arange(n, device=signatures.device)
        return every, every.clone()
    if n > max_frames:
        raise ValueError(
            f"k-medoids compares all the pairs of frames: {n} frames is more than the "
            f"{max_frames} supported"
        )
    dist = distance_matrix(signatures, time_smoothing, metric)
    generator = torch.Generator(device=dist.device).manual_seed(seed)
    medoids = pam(dist, kmedoids_pp_init(dist, k, generator)).sort().values
    labels = dist[:, medoids].argmin(dim=1)
    labels[medoids] = torch.arange(k, device=labels.device)  # identical frames can tie
    return medoids, labels
