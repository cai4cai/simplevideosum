import shutil

import pytest
import torch
from torch import nn

from simplevideosum.backbones import (
    ALIAS_NAMES,
    DEFAULT_BACKBONE,
    DINOV2,
    build_backbone,
    torchvision_weights,
    validate_backbone,
)
from simplevideosum.features import build_feature_extractor, extract_features, preprocess

# Representative families, with random weights: (model, channels, side of the map at its size).
FAMILIES = [
    ("resnet18", 512, 7),
    ("densenet121", 1024, 7),
    ("mobilenet_v3_small", 576, 7),
    ("efficientnet_b0", 1280, 7),
    ("swin_t", 768, 7),
    ("regnet_x_400mf", 400, 7),
    ("shufflenet_v2_x0_5", 1024, 7),
]


def test_specs_designate_torchvision_weights():
    expected = ("mobilenet_v3_large", "MobileNet_V3_Large_Weights")
    for spec in (
        "MobileNet_V3_Large_Weights.IMAGENET1K_V2",
        "MobileNet_V3_Large_Weights.DEFAULT",
        "mobilenet_v3_large",  # the model: its default weights
    ):
        model_name, weights = torchvision_weights(spec)
        assert (model_name, type(weights).__name__) == expected
    assert (
        torchvision_weights("mobilenet_v3_large")[1]
        is torchvision_weights("MobileNet_V3_Large_Weights.DEFAULT")[1]
    )
    assert torchvision_weights("inceptionv3")[0] == "inception_v3"  # aliases
    assert torchvision_weights("mobilenet") == torchvision_weights("mobilenet_v3_large")
    assert DEFAULT_BACKBONE == "mobilenet"


def test_unknown_specs_are_rejected_with_suggestions():
    with pytest.raises(ValueError, match="unknown backbone") as error:
        validate_backbone("MobileNet_V3_Large_Weights.IMAGENET1K_V3")
    assert "MobileNet_V3_Large_Weights.IMAGENET1K_V2" in str(error.value)
    with pytest.raises(ValueError, match="unknown backbone"):
        validate_backbone("bogus")
    for alias in ALIAS_NAMES:
        validate_backbone(alias)  # no torchvision needed


def test_input_size_and_normalisation_come_from_the_weights():
    b3 = build_backbone("EfficientNet_B3_Weights.IMAGENET1K_V1", pretrained=False)
    assert b3.input_size == 300
    assert b3.mean == pytest.approx((0.485, 0.456, 0.406))
    assert b3.std == pytest.approx((0.229, 0.224, 0.225))
    assert (
        build_backbone("MobileNet_V3_Large_Weights.IMAGENET1K_V2", pretrained=False).input_size
        == 224
    )
    assert build_backbone("inceptionv3", pretrained=False).input_size == 299


@pytest.mark.parametrize("model_name, channels, side", FAMILIES)
def test_maps_of_the_torchvision_families(model_name, channels, side):
    backbone = build_backbone(model_name, pretrained=False)
    frames = torch.randint(0, 255, (2, 3, 48, 64), dtype=torch.uint8)
    maps = backbone(preprocess(frames, backbone.input_size, backbone.mean, backbone.std))
    assert maps.shape == (2, channels, side, side)
    model = build_feature_extractor(
        pretrained=False,
        device="cpu",
        backbone=model_name,
        aggregation="avgpool",
        normalisation="none",
    )
    assert extract_features(model, frames).shape == (2, channels)  # frames resized to its size


def test_layers_after_the_maps_do_not_run():
    backbone = build_backbone("resnet18", pretrained=False)
    backbone.model.fc.register_forward_hook(lambda *args: pytest.fail("the classifier ran"))
    seen = []
    backbone.model.layer4.register_forward_hook(lambda module, inputs, out: seen.append(out))
    maps = backbone(torch.randn(1, 3, 224, 224))
    assert maps.shape == (1, 512, 7, 7) and torch.equal(maps, seen[-1])


class Twice(nn.Module):
    """A conv block applied twice by the same module: its last call gives the final maps."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.conv(self.conv(x))).flatten(1)


def test_the_forward_pass_stops_at_the_last_call_of_a_reused_module():
    from simplevideosum.backbones import TorchvisionMaps, find_maps_module

    model = Twice().eval()
    assert find_maps_module(model, 8) == ("conv", 2)
    maps = TorchvisionMaps(model, ("conv", 2), 8, (0, 0, 0), (1, 1, 1))
    x = torch.randn(1, 3, 8, 8)
    assert torch.allclose(maps(x), model.conv(model.conv(x)), atol=1e-6)


@pytest.mark.parametrize(
    "spec",
    [
        "FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1",  # segmentation
        "ResNet50_QuantizedWeights.IMAGENET1K_FBGEMM_V1",  # quantised
        "Raft_Large_Weights.DEFAULT",  # optical flow
    ],
)
def test_weights_that_are_not_those_of_a_classification_model_are_rejected(spec):
    with pytest.raises(ValueError, match="not supported"):
        validate_backbone(spec)


class FakeDinoV2(nn.Module):
    """Stands in for the torch.hub model: 14-pixel patches and 8 channels."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 8, kernel_size=14, stride=14)

    def forward_features(self, x):
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        return {"x_norm_clstoken": tokens.mean(dim=1), "x_norm_patchtokens": tokens}


def test_dinov2_is_loaded_through_torch_hub_and_gives_a_map_of_patch_tokens(monkeypatch):
    calls = []

    def fake_hub_load(repo, name, **kwargs):
        calls.append((repo, name, kwargs))
        return FakeDinoV2()

    monkeypatch.setattr(torch.hub, "load", fake_hub_load)
    model = build_feature_extractor(
        pretrained=False, device="cpu", backbone=DINOV2, aggregation="avgpool", normalisation="none"
    )
    assert calls == [
        ("facebookresearch/dinov2", "dinov2_vits14", {"pretrained": False, "trust_repo": True})
    ]
    frames = torch.randint(0, 255, (2, 3, 64, 96), dtype=torch.uint8)
    assert model[0](preprocess(frames, 224)).shape == (2, 8, 16, 16)  # 224 / 14 patches per side
    assert extract_features(model, frames).shape == (2, 8)
    scda = build_feature_extractor(
        pretrained=False, device="cpu", backbone=DINOV2, aggregation="scda"
    )
    assert extract_features(scda, frames).shape == (2, 16)  # average and max pooling


# The backbones we care about, checked with their real weights by the manual test below (about
# 500 MB in total, the largest being the 109 MB of InceptionV3 and ConvNeXt-Tiny).
CURATED_BACKBONES = [
    "mobilenet",  # the default
    "inceptionv3",
    DINOV2,
    "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
    "MobileNet_V3_Large_Weights.IMAGENET1K_V2",
    "EfficientNet_B0_Weights.IMAGENET1K_V1",
    "EfficientNet_B3_Weights.IMAGENET1K_V1",
    "RegNet_Y_400MF_Weights.IMAGENET1K_V2",
    "ResNet50_Weights.IMAGENET1K_V2",
    "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
]


@pytest.mark.pretrained_backbones
def test_curated_backbones_with_their_pretrained_weights(request, tmp_path):
    """Manual (--pretrained-backbones): the weights are downloaded one at a time into a temporary
    torch hub directory and deleted as soon as their model is checked, so nothing stays in the
    torch cache of the user."""
    pattern = request.config.getoption("--backbones-filter").lower()
    specs = [spec for spec in CURATED_BACKBONES if pattern in spec.lower()]
    assert specs, f"no curated backbone contains '{pattern}'"
    original_dir = torch.hub.get_dir()
    torch.hub.set_dir(str(tmp_path))
    failures = []
    try:
        for spec in specs:
            try:
                backbone = build_backbone(spec, pretrained=True)
                x = torch.zeros(1, 3, backbone.input_size, backbone.input_size)
                with torch.inference_mode():
                    maps = backbone(x)
                assert maps.ndim == 4 and min(maps.shape[2:]) > 1, tuple(maps.shape)
                assert bool(torch.isfinite(maps).all())
                print(f"{spec}: {tuple(maps.shape)}")
            except Exception as error:
                failures.append(f"{spec}: {error!r}")
            finally:
                shutil.rmtree(tmp_path / "checkpoints", ignore_errors=True)
    finally:
        torch.hub.set_dir(original_dir)
    assert not failures, "\n".join(failures)


def test_curated_backbones_are_valid_specs():
    for spec in CURATED_BACKBONES:
        validate_backbone(spec)
