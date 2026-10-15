import pytest
import torch
import torch.nn.functional as F
from torch import nn

from simplevideosum.aggregation import AGGREGATIONS
from simplevideosum.features import (
    build_feature_extractor,
    compute_signatures,
    extract_features,
    preprocess,
)
from simplevideosum.io import VideoReader

SIGNATURE_DIM = 960  # MobileNetV3-Large, the default backbone: its number of channels
FEATURE_MAP_SIZE = 7


@pytest.fixture(scope="module")
def model():
    return build_feature_extractor(
        pretrained=False, device="cpu", aggregation="avgpool", normalisation="none"
    )


def test_preprocess_resizes_and_normalises():
    out = preprocess(torch.full((2, 3, 100, 200), 255, dtype=torch.uint8), 224)
    assert out.shape == (2, 3, 224, 224)
    assert out[0, 0, 0, 0] == pytest.approx((1 - 0.485) / 0.229, abs=1e-4)


def test_signature_shape(model):
    feats = extract_features(model, torch.randint(0, 255, (4, 3, 64, 64), dtype=torch.uint8))
    assert feats.shape == (4, SIGNATURE_DIM)


def test_rejects_bad_input(model):
    with pytest.raises(ValueError):
        extract_features(model, torch.rand(3, 64, 64))


def test_compute_signatures_batches_consistently(model, video_path):
    reader = VideoReader(video_path)
    indices = [0, 500, 1000, 1500, 2000]
    batched = compute_signatures(reader, indices, model, batch_size=2)
    assert batched.shape == (5, SIGNATURE_DIM)
    single = compute_signatures(reader, indices, model, batch_size=5)
    assert torch.allclose(batched, single, atol=1e-4)
    with pytest.raises(ValueError):
        compute_signatures(reader, indices, model, batch_size=0)
    with pytest.raises(ValueError, match="empty"):
        compute_signatures(reader, [], model)


def test_scda_signatures_concatenate_the_normalised_pooled_descriptors():
    scda_model = build_feature_extractor(
        pretrained=False, device="cpu", aggregation="scda", normalisation="l2"
    )
    frames = torch.randint(0, 255, (3, 3, 64, 64), dtype=torch.uint8)
    feats = extract_features(scda_model, frames)
    assert feats.shape == (3, 2 * SIGNATURE_DIM)
    assert torch.allclose(feats[:, :SIGNATURE_DIM].norm(dim=1), torch.ones(3), atol=1e-4)
    assert torch.allclose(feats[:, SIGNATURE_DIM:].norm(dim=1), torch.ones(3), atol=1e-4)


def test_default_signature_is_the_global_average_of_the_maps(model):
    frames = torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8)
    maps = model[0](preprocess(frames, 224))
    assert maps.shape == (2, SIGNATURE_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
    assert torch.allclose(extract_features(model, frames), maps.mean(dim=(2, 3)), atol=1e-5)


@pytest.mark.parametrize("kind", ["backbone", "aggregation", "normalisation"])
def test_unknown_component(kind):
    with pytest.raises(ValueError, match=f"unknown {kind}"):
        build_feature_extractor(pretrained=False, device="cpu", **{kind: "bogus"})


def test_scda_aggregation_variants():
    from simplevideosum.aggregation import AGGREGATIONS

    options = {
        name: (AGGREGATIONS[name]().largest_component, AGGREGATIONS[name]().fill_holes)
        for name in ("scda", "scda_thresh", "scda_fill")
    }
    assert options == {
        "scda": (True, False),  # the paper
        "scda_thresh": (False, False),  # no morphology
        "scda_fill": (True, True),  # videosum
    }


def test_avgcatmaxpool_concatenates_the_global_average_and_max_of_the_maps():
    from simplevideosum.aggregation import AGGREGATIONS

    maps = torch.randn(3, 5, 4, 6)
    pooled = AGGREGATIONS["avgcatmaxpool"]()(maps)
    assert pooled.shape == (3, 2, 5)
    assert torch.allclose(pooled[:, 0], maps.mean(dim=(2, 3)), atol=1e-6)
    assert torch.equal(pooled[:, 1], maps.amax(dim=(2, 3)))

    model = build_feature_extractor(
        pretrained=False, device="cpu", aggregation="avgcatmaxpool", normalisation="l2"
    )
    feats = extract_features(model, torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8))
    assert feats.shape == (2, 2 * SIGNATURE_DIM)  # [average, max], each of unit norm
    assert torch.allclose(feats[:, :SIGNATURE_DIM].norm(dim=1), torch.ones(2), atol=1e-4)
    assert torch.allclose(feats[:, SIGNATURE_DIM:].norm(dim=1), torch.ones(2), atol=1e-4)


def test_default_signature_is_the_normalised_average_and_max_of_the_maps():
    model = build_feature_extractor(pretrained=False, device="cpu")  # the default components
    frames = torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8)
    maps = model[0](preprocess(frames, 224))
    average, maximum = maps.mean(dim=(2, 3)), maps.amax(dim=(2, 3))
    expected = torch.cat([F.normalize(average, dim=-1), F.normalize(maximum, dim=-1)], dim=1)
    assert torch.allclose(extract_features(model, frames), expected, atol=1e-5)


class Prepared(nn.Module):
    """A backbone-like module that asks for 8-pixel frames and a unit normalisation."""

    input_size, mean, std = 8, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))  # extract_features reads the device from it

    def forward(self, x):
        assert x.shape[-2:] == (8, 8)
        return x.new_zeros(len(x), 3, 20, 20) + torch.arange(400).reshape(20, 20)  # 400 positions


def test_a_model_that_is_not_from_build_feature_extractor_gets_the_inception_preparation():
    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, x):
            assert x.shape[-2:] == (299, 299)
            return x.mean(dim=(2, 3)) * self.weight

    frames = torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8)
    assert extract_features(Plain(), frames).shape == (2, 3)


def test_scda_aggregation_on_maps_with_more_than_256_positions():
    model = nn.Sequential(Prepared(), AGGREGATIONS["scda"](), nn.Flatten())
    model.eval()
    feats = extract_features(model, torch.randint(0, 255, (2, 3, 32, 32), dtype=torch.uint8))
    assert feats.shape == (2, 2 * 3) and bool(torch.isfinite(feats).all())
