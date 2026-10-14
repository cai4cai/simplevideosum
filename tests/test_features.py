import pytest
import torch

from simplevideosum.features import build_feature_extractor, extract_features


def test_feature_shape():
    model = build_feature_extractor(pretrained=False)
    feats = extract_features(model, torch.rand(4, 3, 64, 64))
    assert feats.shape == (4, 512)


def test_rejects_bad_input():
    model = build_feature_extractor(pretrained=False)
    with pytest.raises(ValueError):
        extract_features(model, torch.rand(3, 64, 64))
