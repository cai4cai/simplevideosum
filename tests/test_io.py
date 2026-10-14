import pytest
import torch

from simplevideosum.features import build_feature_extractor, extract_features
from simplevideosum.io import VideoReader


@pytest.fixture(scope="module")
def reader(video_path):
    return VideoReader(video_path)


def test_metadata(reader):
    assert reader.num_frames == 3827
    assert reader.frame_size == (480, 640)
    assert reader.fps == pytest.approx(29.97, abs=0.01)
    assert reader.duration == pytest.approx(127.69, abs=0.01)


def test_read_returns_uint8_nchw(reader):
    frames = reader.read([0, 10, 100])
    assert frames.shape == (3, 3, 480, 640)
    assert frames.dtype == torch.uint8


def test_sample_is_uniform_and_clamped(reader):
    assert reader.sample(8).shape[0] == 8
    assert reader.sample(10**6).shape[0] == reader.num_frames


def test_batches_cover_strided_frames(reader):
    sizes = [b.shape[0] for b in reader.batches(batch_size=16, stride=500)]
    assert sum(sizes) == len(range(0, reader.num_frames, 500))
    assert max(sizes) <= 16


def test_invalid_arguments(reader):
    with pytest.raises(ValueError):
        reader.sample(0)
    with pytest.raises(ValueError):
        next(reader.batches(0))


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoReader(tmp_path / "nope.mp4")


def test_frames_feed_feature_extractor(reader):
    model = build_feature_extractor(pretrained=False, device="cpu")
    feats = extract_features(model, reader.sample(4))
    assert feats.shape == (4, 2048)
