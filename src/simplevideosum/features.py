"""Frame feature extraction built on torchvision."""

import torch
from torch import nn
from torchvision import models


def build_feature_extractor(pretrained: bool = False) -> nn.Module:
    """Return a ResNet-18 trunk mapping (N, 3, H, W) frames to (N, 512) features."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    backbone = models.resnet18(weights=weights)
    backbone.fc = nn.Identity()
    return backbone.eval()


@torch.no_grad()
def extract_features(model: nn.Module, frames: torch.Tensor) -> torch.Tensor:
    """Extract per-frame features from a (N, 3, H, W) float tensor."""
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"expected (N, 3, H, W) frames, got {tuple(frames.shape)}")
    return model(frames)
