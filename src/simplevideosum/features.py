"""Per-frame signatures: a pre-trained backbone, then an aggregation and a normalisation."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from simplevideosum.aggregation import AGGREGATIONS, NORMALISATIONS
from simplevideosum.device import get_device

if TYPE_CHECKING:  # TorchCodec is slow to import, and only the type is needed here
    from simplevideosum.io import VideoReader

# InceptionV3: input size, channels and side of its last convolutional activations. Every backbone
# module has an ``input_size`` attribute: the side of the square frames it expects.
INPUT_SIZE = 299
SIGNATURE_DIM = 2048
FEATURE_MAP_SIZE = 8
CONVNEXT_INPUT_SIZE = 224  # ConvNeXt-Tiny: 768 channels on a 7 x 7 grid
DINOV2_INPUT_SIZE = 224  # a multiple of the patch size 14: a 16 x 16 grid of patches
DINOV2_PATCH_SIZE = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InceptionMaps(nn.Module):
    """InceptionV3 up to its last convolutional block: (N, 3, 299, 299) frames to the
    (N, 2048, 8, 8) activations that the global average pooling normally collapses."""

    input_size = INPUT_SIZE

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.avgpool = nn.Identity()
        self.model.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.model(x)  # eval mode: the flattened activations, without the aux classifier
        return flat.reshape(len(x), SIGNATURE_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)


def build_inception_v3(pretrained: bool) -> nn.Module:
    from torchvision import models  # deferred: importing torchvision is slow (see run.py)

    if pretrained:
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1, progress=True
        )
    else:
        model = models.inception_v3(weights=None, init_weights=False)
    return InceptionMaps(model)


class ConvNeXtMaps(nn.Module):
    """ConvNeXt up to its last stage: (N, 3, 224, 224) frames to the (N, 768, 7, 7) activations
    that its global average pooling and classifier normally consume."""

    input_size = CONVNEXT_INPUT_SIZE

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.features = model.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def build_convnext_tiny(pretrained: bool) -> nn.Module:
    from torchvision import models  # deferred: importing torchvision is slow (see run.py)

    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    return ConvNeXtMaps(models.convnext_tiny(weights=weights, progress=True))


class DinoV2Maps(nn.Module):
    """DINOv2 up to its last block: (N, 3, 224, 224) frames to the (N, C, 16, 16) map of the
    normalised patch tokens (the class token is dropped)."""

    input_size = DINOV2_INPUT_SIZE

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(x)["x_norm_patchtokens"]  # (N, 16 * 16, C)
        side = x.shape[-1] // DINOV2_PATCH_SIZE
        return tokens.transpose(1, 2).reshape(len(x), -1, side, side)


def build_dinov2_vits14(pretrained: bool) -> nn.Module:
    """DINOv2 ViT-S/14 (22M parameters, 384 channels), from the authors' repository through
    ``torch.hub``: the first use downloads its code from GitHub (``facebookresearch/dinov2``,
    main branch) and, if ``pretrained``, its weights, both to the torch hub cache."""
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", pretrained=pretrained, trust_repo=True
    )
    return DinoV2Maps(model)


BACKBONES: dict[str, Callable[[bool], nn.Module]] = {
    "inceptionv3": build_inception_v3,
    "convnext_tiny": build_convnext_tiny,
    "dinov2_vits14": build_dinov2_vits14,
}


def build_feature_extractor(
    pretrained: bool = True,
    device: str | torch.device | None = None,
    backbone: str = "inceptionv3",
    aggregation: str = "avgpool",
    normalisation: str = "none",
) -> nn.Module:
    """Model mapping uint8 (N, 3, H, W) frames to (N, D) signatures, in three steps:
    the ``backbone`` gives convolutional activations, the ``aggregation`` pools them into one or
    more descriptors per frame and the ``normalisation`` is applied to each (see ``aggregation``
    module); the descriptors are then concatenated. With the defaults, for InceptionV3, this is
    the 2048-d globally average pooled activations. The frames are resized and normalised
    for the backbone by ``extract_features``.

    ``pretrained=True`` uses the pre-trained weights, downloaded on first use (about 100 MB for
    InceptionV3, 85 MB for DINOv2) to the torch hub cache (``~/.cache/torch/hub/checkpoints``)
    with a progress bar on stderr; later runs reuse the cached file. The model is placed on
    ``device`` (default: CUDA, MPS, then CPU).
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


def preprocess(frames: torch.Tensor, input_size: int = INPUT_SIZE) -> torch.Tensor:
    """Resize (N, 3, H, W) uint8 frames to ``input_size`` squared and apply the ImageNet
    normalisation."""
    x = F.interpolate(
        frames.float() / 255,
        size=(input_size, input_size),
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
    input_size = getattr(model[0], "input_size", INPUT_SIZE)  # model[0] is the backbone
    return model(preprocess(frames.to(device), input_size))


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
