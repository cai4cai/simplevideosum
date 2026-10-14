"""Per-frame signatures: a pre-trained backbone, then an aggregation and a normalisation."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from simplevideosum.aggregation import AGGREGATIONS, NORMALISATIONS
from simplevideosum.device import get_device
from simplevideosum.io import VideoReader

INPUT_SIZE = 299
SIGNATURE_DIM = 2048
FEATURE_MAP_SIZE = 8  # side of the last InceptionV3 convolutional activations for 299x299 inputs
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InceptionMaps(nn.Module):
    """InceptionV3 up to its last convolutional block: (N, 3, 299, 299) frames to the
    (N, 2048, 8, 8) activations that the global average pooling normally collapses."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.avgpool = nn.Identity()
        self.model.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.model(x)  # eval mode: the flattened activations, without the aux classifier
        return flat.reshape(len(x), SIGNATURE_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)


def build_inception_v3(pretrained: bool) -> nn.Module:
    if pretrained:
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1, progress=True
        )
    else:
        model = models.inception_v3(weights=None, init_weights=False)
    return InceptionMaps(model)


BACKBONES: dict[str, Callable[[bool], nn.Module]] = {"inceptionv3": build_inception_v3}


def build_feature_extractor(
    pretrained: bool = True,
    device: str | torch.device | None = None,
    backbone: str = "inceptionv3",
    aggregation: str = "avgpool",
    normalisation: str = "none",
) -> nn.Module:
    """Model mapping normalised (N, 3, 299, 299) frames to (N, D) signatures, in three steps:
    the ``backbone`` gives convolutional activations, the ``aggregation`` pools them into one or
    more descriptors per frame and the ``normalisation`` is applied to each (see ``aggregation``
    module); the descriptors are then concatenated. With the defaults, for InceptionV3, this is
    the 2048-d globally average pooled activations.

    ``pretrained=True`` uses the ImageNet weights, downloaded on first use (about 100 MB) to the
    torch hub cache (``~/.cache/torch/hub/checkpoints``) with a progress bar on stderr; later
    runs reuse the cached file. The model is placed on ``device`` (default: CUDA, MPS, then CPU).
    """
    for kind, name, registry in (
        ("backbone", backbone, BACKBONES),
        ("aggregation", aggregation, AGGREGATIONS),
        ("normalisation", normalisation, NORMALISATIONS),
    ):
        if name not in registry:
            raise ValueError(f"unknown {kind} '{name}' (available: {', '.join(registry)})")
    model = nn.Sequential(
        BACKBONES[backbone](pretrained),
        AGGREGATIONS[aggregation](),
        NORMALISATIONS[normalisation](),
        nn.Flatten(),
    )
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
    """Signatures (N, D) of (N, 3, H, W) uint8 frames, computed on the model's device."""
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"expected (N, 3, H, W) frames, got {tuple(frames.shape)}")
    device = next(model.parameters()).device
    return model(preprocess(frames.to(device)))


def compute_signatures(
    reader: VideoReader, indices: list[int], model: nn.Module, batch_size: int = 32
) -> torch.Tensor:
    """Signatures (len(indices), D) of the given video frames, decoded batch by batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    batches = [
        extract_features(model, reader.read(indices[i : i + batch_size]))
        for i in range(0, len(indices), batch_size)
    ]
    return torch.cat(batches) if batches else torch.empty(0, SIGNATURE_DIM)
