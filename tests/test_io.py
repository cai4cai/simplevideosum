from types import SimpleNamespace

import pytest
import torch

from simplevideosum.features import build_feature_extractor, extract_features
from simplevideosum.io import VideoReader


@pytest.fixture(scope="module")
def reader(video_path):
    return VideoReader(video_path)


def test_metadata(reader):
    assert reader.num_frames == 3827
    assert reader.fps == pytest.approx(29.97, abs=0.01)
    assert reader.duration == pytest.approx(127.69, abs=0.01)


def test_read_returns_uint8_nchw(reader):
    frames = reader.read([0, 10, 100])
    assert frames.shape == (3, 3, 480, 640)
    assert frames.dtype == torch.uint8


def test_threaded_decoding_gives_the_same_frames_as_single_threaded(video_path, reader):
    from torchcodec.decoders import VideoDecoder

    indices = [0, 7, 500, 1500, 3826]
    single = VideoDecoder(video_path, num_ffmpeg_threads=1).get_frames_at(indices).data
    assert torch.equal(reader.read(indices), single)


class FakeDecoder:
    """Stands in for a TorchCodec decoder with 100 frames and the given metadata."""

    def __init__(self, **metadata):
        self.metadata = SimpleNamespace(**metadata)

    def __len__(self):
        return 100


def fake_reader(reader, **metadata):
    reader._decoder = FakeDecoder(**metadata)
    return reader


def test_fps_falls_back_on_the_duration(video_path):
    reader = fake_reader(VideoReader(video_path), average_fps=None, duration_seconds=4.0)
    assert reader.fps == 25.0
    assert reader.sampled_indices(5) == list(range(0, 100, 5))


def test_unknown_frame_rate_is_an_error(video_path):
    reader = fake_reader(VideoReader(video_path), average_fps=None, duration_seconds=None)
    with pytest.raises(ValueError, match="frame rate"):
        reader.fps  # noqa: B018


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoReader(tmp_path / "nope.mp4")


def test_frames_feed_feature_extractor(reader):
    model = build_feature_extractor(pretrained=False, device="cpu")
    feats = extract_features(model, reader.read([0, 500, 1000, 1500]))
    assert feats.shape == (4, 2048)
