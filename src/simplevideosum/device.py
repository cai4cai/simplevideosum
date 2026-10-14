"""Device selection helpers."""

import torch


def get_device() -> torch.device:
    """Return the best available device (CUDA, then Apple MPS, then CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
