from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

import pytest
import torch
from torch import nn

from simplevideosum.algorithms import CLUSTERINGS, DEFAULT_ALGORITHM, DEFAULT_FPS
from simplevideosum.collage import SET3
from simplevideosum.features import build_feature_extractor
from simplevideosum.io import VideoReader
from simplevideosum.summarizer import VideoSummarizer, chronological


def test_storyboard_with_time_bar_plot(video_path):
    summarizer = VideoSummarizer("time", 16, 1920, 1080, time_bar_plot=True)
    storyboard = summarizer.summarize(video_path)
    assert tuple(storyboard.shape) == (3, 1080 + 32, 1920)


def test_storyboard_without_time_bar_plot(video_path):
    storyboard = VideoSummarizer("time", 16, 960, 540).summarize(video_path)
    assert tuple(storyboard.shape) == (3, 540, 960)


def test_fps_resamples_the_video(video_path):
    reader = VideoReader(video_path)
    indices = reader.sampled_indices(fps=2)
    assert len(indices) == int(reader.duration * 2)
    assert indices[0] == 0 and indices[-1] < reader.num_frames
    everything = list(range(reader.num_frames))
    assert reader.sampled_indices(None) == reader.sampled_indices(0) == everything
    assert reader.sampled_indices(1000) == everything  # at least the frame rate of the video
    with pytest.raises(ValueError, match=">= 0"):
        reader.sampled_indices(-1)


def test_unknown_algorithm():
    with pytest.raises(ValueError, match="unknown algorithm"):
        VideoSummarizer("nope", 16, 1920, 1080)


def test_time_algorithm_honours_fps_and_decodes_only_key_frames(video_path, monkeypatch, caplog):
    requested = []
    original = VideoReader.read

    def spy(self, indices):
        requested.append(list(indices))
        return original(self, indices)

    monkeypatch.setattr(VideoReader, "read", spy)
    VideoSummarizer("time", 16, 640, 360, fps=0.5).summarize(video_path)
    sampled = VideoReader(video_path).sampled_indices(0.5)
    assert len(requested) == 1 and len(requested[0]) == 16
    assert set(requested[0]) <= set(sampled)  # evenly spaced among the frames at 0.5 fps
    assert requested[0][0] == sampled[0] and requested[0][-1] == sampled[-1]
    assert "ignores" not in caplog.text  # --fps is not ignored any more


@pytest.fixture(scope="module")
def random_extractor():
    return build_feature_extractor(pretrained=False, device="cpu")


@pytest.mark.parametrize("clustering", ["kmedoids", "dissection", "fisher"])
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
        "videosum_inception", 8, 640, 360, fps=0.5, time_bar_plot=True,
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
    summarizer = VideoSummarizer(
        "videosum_inception", 64, 640, 360, fps=0.02, clustering="dissection",
        feature_extractor=random_extractor,
    )  # fmt: skip
    with pytest.raises(ValueError, match="cannot split"):
        summarizer.summarize(video_path)


def test_inception_requires_power_of_two_frames():
    with pytest.raises(ValueError, match="power of 2"):
        VideoSummarizer("videosum_inception", 12, 640, 360, clustering="dissection")


def test_kmedoids_accepts_any_number_of_frames_and_time_smoothing(video_path, random_extractor):
    summarizer = VideoSummarizer("videosum_inception", 12, 640, 360, fps=0.5, time_smoothing=0.3,
                                 feature_extractor=random_extractor)  # fmt: skip
    assert summarizer.summarize(video_path).shape[0] == 3


def test_time_algorithm_warns_about_ignored_options(caplog):
    VideoSummarizer("time", 16, 640, 360, fps=2, clustering="kmedoids", distance="cosine")
    assert "ignores --clustering and --distance" in caplog.text


def test_default_algorithm_with_the_pretrained_default_backbone_end_to_end(request, video_path):
    from simplevideosum.backbones import DEFAULT_BACKBONE, torchvision_weights

    weights_file = Path(torch.hub.get_dir()) / "checkpoints"
    weights_file /= Path(urlparse(torchvision_weights(DEFAULT_BACKBONE)[1].url).path).name
    if not weights_file.exists() and not request.config.getoption("--pretrained-backbones"):
        pytest.skip("the default weights are not in the torch cache: --pretrained-backbones")
    storyboard = VideoSummarizer(DEFAULT_ALGORITHM, 16, 640, 360, fps=0.5).summarize(video_path)
    assert tuple(storyboard.shape) == (3, 360, 640)  # (the weights stay in the torch cache)


def test_scda_algorithm_clusters_scda_signatures(video_path):
    extractor = build_feature_extractor(
        pretrained=False, device="cpu", aggregation="scda", normalisation="l2"
    )
    summarizer = VideoSummarizer("videosum_scda", 8, 640, 360, fps=0.5, feature_extractor=extractor)
    storyboard = summarizer.summarize(video_path)
    layout = summarizer.layout
    assert tuple(storyboard.shape) == (
        3,
        layout.tiles_per_col * layout.tile_height,
        layout.tiles_per_row * layout.tile_width,
    )


def test_component_overrides_reach_the_clustering(video_path, random_extractor, monkeypatch):
    seen = []
    original = CLUSTERINGS["kmedoids"].select

    def spy(signatures, n, time_smoothing, distance):
        seen.append(distance)
        return original(signatures, n, time_smoothing, distance)

    monkeypatch.setitem(CLUSTERINGS, "kmedoids", replace(CLUSTERINGS["kmedoids"], select=spy))
    summarizer = VideoSummarizer("videosum_inception", 8, 640, 360, fps=0.5, distance="cosine",
                                 feature_extractor=random_extractor)  # fmt: skip
    summarizer.summarize(video_path)
    assert seen == ["cosine"]


def test_too_many_frames_for_the_clustering_fail_before_computing_signatures(
    video_path, monkeypatch
):
    monkeypatch.setitem(CLUSTERINGS, "kmedoids", replace(CLUSTERINGS["kmedoids"], max_frames=10))

    class Unused(torch.nn.Module):
        def forward(self, x):  # pragma: no cover
            raise AssertionError("the signatures should not be computed")

    summarizer = VideoSummarizer(
        "videosum_inception", 8, 640, 360, fps=2, feature_extractor=Unused()
    )
    with pytest.raises(ValueError, match="supports at most 10 frames.*lower --fps"):
        summarizer.summarize(video_path)


def test_short_summary_warns_through_logging(video_path, random_extractor, caplog):
    summarizer = VideoSummarizer("videosum_inception", 64, 640, 360, fps=0.05,
                                 feature_extractor=random_extractor)  # fmt: skip
    summarizer.summarize(video_path)
    assert "requested frames available" in caplog.text


def test_chronological_sorts_key_frames_and_keeps_labels_consistent():
    indices = torch.tensor([7, 2, 5])
    labels = torch.tensor([1, 1, 1, 2, 2, 2, 0, 0, 0, 0])  # frame i belongs to indices[labels[i]]
    sorted_indices, new_labels = chronological(indices, labels)
    assert sorted_indices.tolist() == [2, 5, 7]
    assert [int(sorted_indices[k]) for k in new_labels] == [int(indices[k]) for k in labels]


def test_kmedoids_storyboard_is_chronological_with_matching_borders(
    video_path, random_extractor, monkeypatch
):
    seen = []
    original = VideoReader.read

    def spy(self, indices):
        seen.append(list(indices))
        return original(self, indices)

    monkeypatch.setattr(VideoReader, "read", spy)
    summarizer = VideoSummarizer(
        "videosum_inception", 6, 640, 360, fps=2, time_bar_plot=True,
        feature_extractor=random_extractor,
    )  # fmt: skip
    storyboard = summarizer.summarize(video_path)
    assert seen[-1] == sorted(seen[-1])  # the key frames are read, and shown, in time order
    layout = summarizer.layout
    bar = storyboard[:, -summarizer.segbar_height, :]  # colours of the time bar plot
    for k in range(6):
        row, col = divmod(k, layout.tiles_per_row)
        corner = storyboard[:, row * layout.tile_height, col * layout.tile_width]
        assert corner.tolist() == SET3[k].tolist()
        assert corner.tolist() in bar.T.tolist()  # the border colour is a colour of the bar


def test_algorithms_with_signatures_sample_the_video_at_one_hertz_by_default(caplog):
    assert DEFAULT_FPS == 1.0
    assert VideoSummarizer(DEFAULT_ALGORITHM, 8, 640, 360).fps == DEFAULT_FPS
    assert VideoSummarizer("videosum_scda", 8, 640, 360).fps == DEFAULT_FPS
    assert VideoSummarizer("videosum_inception", 8, 640, 360, fps=4).fps == 4  # explicit wins
    assert VideoSummarizer("time", 8, 640, 360).fps == 0  # every frame is a candidate, no warning
    assert (
        VideoSummarizer(DEFAULT_ALGORITHM, 8, 640, 360, fps=0).fps == 0
    )  # explicit 0: every frame
    assert "ignores" not in caplog.text


def test_default_sampling_decodes_one_candidate_per_second(
    video_path, random_extractor, monkeypatch
):
    seen = []
    original = VideoReader.read

    def spy(self, indices):
        seen.append(len(indices))
        return original(self, indices)

    monkeypatch.setattr(VideoReader, "read", spy)
    VideoSummarizer(DEFAULT_ALGORITHM, 8, 640, 360, feature_extractor=random_extractor).summarize(
        video_path
    )
    assert sum(seen[:-1]) == len(VideoReader(video_path).sampled_indices(1.0))  # ~128 s of video


class Tiny(nn.Module):
    """Stands in for a feature extractor: 2-d signatures from the colours of 8-pixel frames."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    input_size, mean, std = 8, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    def forward(self, x):
        return x.mean(dim=(2, 3))[:, :2] * self.weight


def test_a_video_shorter_than_the_key_frames_in_seconds_is_sampled_at_every_frame(
    video_path, caplog
):
    seconds = int(VideoReader(video_path).duration)  # about 127 s: 1 Hz gives fewer than 130
    summarizer = VideoSummarizer(DEFAULT_ALGORITHM, seconds + 3, 640, 360, feature_extractor=Tiny())
    layout = summarizer.layout
    storyboard = summarizer.summarize(video_path)
    assert tuple(storyboard.shape) == (
        3,
        layout.tiles_per_col * layout.tile_height,
        layout.tiles_per_row * layout.tile_width,
    )
    assert "using every frame" in caplog.text


def test_an_explicit_fps_is_never_raised_for_lack_of_frames(video_path):
    summarizer = VideoSummarizer(
        DEFAULT_ALGORITHM, 40, 640, 360, fps=0.1, feature_extractor=Tiny()
    )  # 12 frames only
    with pytest.raises(ValueError, match="cannot split"):
        summarizer.summarize(video_path)


def test_time_segmentation_is_a_deprecated_name_of_time_bar_plot():
    with pytest.warns(DeprecationWarning, match="time_bar_plot"):
        summarizer = VideoSummarizer("time", 4, 640, 360, time_segmentation=True)
    assert summarizer.time_bar_plot is True
