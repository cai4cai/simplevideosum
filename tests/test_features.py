import pytest
import torch

from simplevideosum.features import (
    FEATURE_MAP_SIZE,
    SIGNATURE_DIM,
    build_feature_extractor,
    compute_signatures,
    extract_features,
    preprocess,
)
from simplevideosum.io import VideoReader


@pytest.fixture(scope="module")
def model():
    return build_feature_extractor(pretrained=False, device="cpu")


def test_preprocess_resizes_and_normalises():
    out = preprocess(torch.full((2, 3, 100, 200), 255, dtype=torch.uint8))
    assert out.shape == (2, 3, 299, 299)
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
    maps = model[0](preprocess(frames))
    assert maps.shape == (2, SIGNATURE_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
    assert torch.allclose(extract_features(model, frames), maps.mean(dim=(2, 3)), atol=1e-5)


@pytest.mark.parametrize("kind", ["backbone", "aggregation", "normalisation"])
def test_unknown_component(kind):
    with pytest.raises(ValueError, match=f"unknown {kind}"):
        build_feature_extractor(pretrained=False, device="cpu", **{kind: "bogus"})
