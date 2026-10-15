"""Per-frame signatures: a pre-trained backbone, then an aggregation and a normalisation."""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from simplevideosum.aggregation import AGGREGATIONS, NORMALISATIONS
from simplevideosum.backbones import IMAGENET_MEAN, IMAGENET_STD, build_backbone, validate_backbone
from simplevideosum.device import get_device

if TYPE_CHECKING:  # TorchCodec is slow to import, and only the type is needed here
    from simplevideosum.io import VideoReader

# Size of the frames and normalisation when the backbone does not say (InceptionV3's).
INPUT_SIZE = 299


def build_feature_extractor(
    pretrained: bool = True,
    device: str | torch.device | None = None,
    backbone: str = "inceptionv3",
    aggregation: str = "avgpool",
    normalisation: str = "none",
) -> nn.Module:
    """Model mapping uint8 (N, 3, H, W) frames to (N, D) signatures, in three steps: the
    ``backbone`` (see ``backbones``) gives convolutional activations, the ``aggregation`` pools
    them into one or more descriptors per frame and the ``normalisation`` is applied to each (see
    ``aggregation`` module); the descriptors are then concatenated. With the defaults, for
    InceptionV3, this is the 2048-d globally average pooled activations. The frames are resized
    and normalised for the backbone by ``extract_features``.

    ``pretrained=True`` uses the pre-trained weights, downloaded on first use (about 100 MB for
    InceptionV3, 85 MB for DINOv2) to the torch hub cache (``~/.cache/torch/hub/checkpoints``)
    with a progress bar on stderr; later runs reuse the cached file. The model is placed on
    ``device`` (default: CUDA, MPS, then CPU).
    """
    validate_backbone(backbone)
    for kind, name, registry in (
        ("aggregation", aggregation, AGGREGATIONS),
        ("normalisation", normalisation, NORMALISATIONS),
    ):
        if name not in registry:
            raise ValueError(f"unknown {kind} '{name}' (available: {', '.join(registry)})")
    model = nn.Sequential(
        build_backbone(backbone, pretrained),
        AGGREGATIONS[aggregation](),
        NORMALISATIONS[normalisation](),
        nn.Flatten(),
    )
    return model.eval().to(device if device is not None else get_device())


def preprocess(
    frames: torch.Tensor,
    input_size: int = INPUT_SIZE,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> torch.Tensor:
    """Resize (N, 3, H, W) uint8 frames to ``input_size`` squared and normalise them."""
    x = F.interpolate(
        frames.float() / 255,
        size=(input_size, input_size),
        mode="bilinear",
        antialias=True,
        align_corners=False,
    )
    mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def extract_features(model: nn.Module, frames: torch.Tensor) -> torch.Tensor:
    """Signatures (N, D) of (N, 3, H, W) uint8 frames, computed on the model's device."""
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"expected (N, 3, H, W) frames, got {tuple(frames.shape)}")
    device = next(model.parameters()).device
    backbone = model[0]
    x = preprocess(frames.to(device), backbone.input_size, backbone.mean, backbone.std)
    return model(x)


def compute_signatures(
    reader: "VideoReader", indices: list[int], model: nn.Module, batch_size: int = 32
) -> torch.Tensor:
    """Signatures (len(indices), D) of the given video frames, decoded batch by batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not indices:
        raise ValueError("indices must not be empty")
    return torch.cat(
        [
            extract_features(model, reader.read(indices[i : i + batch_size]))
            for i in range(0, len(indices), batch_size)
        ]
    )
