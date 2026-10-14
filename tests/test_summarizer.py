import pytest

from simplevideosum.io import VideoReader
from simplevideosum.summarizer import VideoSummarizer


def test_storyboard_with_time_segmentation(video_path):
    summarizer = VideoSummarizer("time", 16, 1920, 1080, time_segmentation=True)
    storyboard = summarizer.summarize(video_path)
    assert tuple(storyboard.shape) == (3, 1080 + 32, 1920)


def test_storyboard_without_time_segmentation(video_path):
    storyboard = VideoSummarizer("time", 16, 960, 540).summarize(video_path)
    assert tuple(storyboard.shape) == (3, 540, 960)


def test_fps_resamples_the_video(video_path):
    reader = VideoReader(video_path)
    indices = reader.sampled_indices(fps=2)
    assert len(indices) == int(reader.duration * 2)
    assert indices[0] == 0 and indices[-1] < reader.num_frames
    assert reader.sampled_indices(None) == list(range(reader.num_frames))
    with pytest.raises(ValueError):
        reader.sampled_indices(0)


@pytest.mark.parametrize("algo", ["frechet", "nope"])
def test_unsupported_algorithms(algo):
    with pytest.raises(ValueError):
        VideoSummarizer(algo, 16, 1920, 1080)


def test_time_algorithm_ignores_fps_and_decodes_only_key_frames(video_path, monkeypatch, capsys):
    requested = []
    original = VideoReader.read

    def spy(self, indices):
        requested.append(list(indices))
        return original(self, indices)

    monkeypatch.setattr(VideoReader, "read", spy)
    VideoSummarizer("time", 16, 640, 360, fps=0.5).summarize(video_path)
    assert len(requested) == 1 and len(requested[0]) == 16
    assert requested[0][0] == 0 and requested[0][-1] == VideoReader(video_path).num_frames - 1
    assert "ignores the sampling frequency" in capsys.readouterr().out
