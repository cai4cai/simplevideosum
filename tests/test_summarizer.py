import pytest

from simplevideosum.features import build_feature_extractor
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
    assert "ignores --fps" in capsys.readouterr().out


@pytest.fixture(scope="module")
def random_extractor():
    return build_feature_extractor(pretrained=False, device="cpu")


@pytest.mark.parametrize("clustering", ["kmedoids", "dissection"])
def test_inception_algorithm_uses_fps_and_decodes_candidates_once(
    video_path, random_extractor, monkeypatch, clustering
):
    seen = []
    original = VideoReader.read

    def spy(self, indices):
        seen.append(len(indices))
        return original(self, indices)

    monkeypatch.setattr(VideoReader, "read", spy)
    summarizer = VideoSummarizer(
        "inception", 8, 640, 360, fps=0.5, time_segmentation=True,
        feature_extractor=random_extractor, clustering=clustering,
    )  # fmt: skip
    storyboard = summarizer.summarize(video_path)
    layout = summarizer.layout
    expected_width = layout.tiles_per_row * layout.tile_width
    assert tuple(storyboard.shape) == (
        3,
        layout.tiles_per_col * layout.tile_height + 32,
        expected_width,
    )
    n_candidates = len(VideoReader(video_path).sampled_indices(0.5))
    assert sum(seen[:-1]) == n_candidates  # every candidate was decoded for its signature
    assert seen[-1] == 8  # then only the key frames are decoded again for the collage


def test_inception_algorithm_needs_enough_sampled_frames(video_path, random_extractor):
    summarizer = VideoSummarizer("inception", 64, 640, 360, fps=0.02, clustering="dissection",
                                 feature_extractor=random_extractor)  # fmt: skip
    with pytest.raises(ValueError, match="cannot split"):
        summarizer.summarize(video_path)


def test_inception_requires_power_of_two_frames():
    with pytest.raises(ValueError, match="power of 2"):
        VideoSummarizer("inception", 12, 640, 360, clustering="dissection")


def test_kmedoids_accepts_any_number_of_frames_and_time_smoothing(video_path, random_extractor):
    summarizer = VideoSummarizer("inception", 12, 640, 360, fps=0.5, time_smoothing=0.3,
                                 feature_extractor=random_extractor)  # fmt: skip
    assert summarizer.summarize(video_path).shape[0] == 3


def test_time_algorithm_warns_about_ignored_clustering(capsys):
    VideoSummarizer("time", 16, 640, 360, fps=2, clustering="kmedoids")
    assert "ignores --fps and --clustering" in capsys.readouterr().out


def test_pretrained_inception_end_to_end(video_path):
    storyboard = VideoSummarizer("inception", 16, 640, 360, fps=0.5).summarize(video_path)
    assert tuple(storyboard.shape) == (3, 360, 640)
