"""Per-frame signatures from a pre-trained InceptionV3."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from simplevideosum.device import get_device
from simplevideosum.io import VideoReader

INPUT_SIZE = 299
SIGNATURE_DIM = 2048
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_feature_extractor(
    pretrained: bool = True, device: str | torch.device | None = None
) -> nn.Module:
    """InceptionV3 trunk mapping normalised (N, 3, 299, 299) frames to (N, 2048) signatures.

    ``pretrained=True`` uses the ImageNet weights, downloaded on first use (about 100 MB) to the
    torch hub cache (``~/.cache/torch/hub/checkpoints``) with a progress bar on stderr; later
    runs reuse the cached file. The model is placed on ``device`` (default: CUDA, MPS, then CPU).
    """
    if pretrained:
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1, progress=True
        )
    else:
        model = models.inception_v3(weights=None, init_weights=False)
    model.fc = nn.Identity()
    return model.eval().to(device if device is not None else get_device())


def preprocess(frames: torch.Tensor) -> torch.Tensor:
    """Resize (N, 3, H, W) uint8 frames to 299x299 and apply the ImageNet normalisation."""
    x = F.interpolate(
        frames.float() / 255,
        size=(INPUT_SIZE, INPUT_SIZE),
        mode="bilinear",
        antialias=True,
        align_corners=False,
    )
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def extract_features(model: nn.Module, frames: torch.Tensor) -> torch.Tensor:
    """Signatures (N, 2048) of (N, 3, H, W) uint8 frames, computed on the model's device."""
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"expected (N, 3, H, W) frames, got {tuple(frames.shape)}")
    device = next(model.parameters()).device
    return model(preprocess(frames.to(device)))


def compute_signatures(
    reader: VideoReader, indices: list[int], model: nn.Module, batch_size: int = 32
) -> torch.Tensor:
    """Signatures (len(indices), 2048) of the given video frames, decoded batch by batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    batches = [
        extract_features(model, reader.read(indices[i : i + batch_size]))
        for i in range(0, len(indices), batch_size)
    ]
    return torch.cat(batches) if batches else torch.empty(0, SIGNATURE_DIM)
