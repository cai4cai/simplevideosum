import subprocess
import sys

import pytest
from torchcodec.decoders import decode_image as read_image

from simplevideosum.run import main


def args(video_path, output, *extra):
    return ["-i", str(video_path), "-o", str(output), "-n", "16", "-x", "640", "-y", "360",
            "-a", "time", *extra]  # fmt: skip


def test_help_documents_every_option(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for option in ("--input", "--output", "--nframes", "--width", "--height", "--algo",
                   "--time-bar-plot", "--clustering", "--backbone", "--aggregation",
                   "--normalisation", "--distance", "--time-smoothing", "--fps", "--log",
                   "example:"):  # fmt: skip
        assert option in out


@pytest.mark.parametrize("ext", [".jpg", ".png"])
def test_main_writes_storyboard(video_path, tmp_path, ext):
    output = tmp_path / f"story{ext}"
    assert main(args(video_path, output, "-t", "1", "-f", "2")) == 0
    assert tuple(read_image(output).shape) == (3, 360 + 32, 640)


def test_bare_time_bar_plot_flag_enables_bar(video_path, tmp_path):
    output = tmp_path / "story.png"
    main(args(video_path, output, "-t"))
    assert read_image(output).shape[1] == 360 + 32


def test_log_file_is_written(video_path, tmp_path):
    log = tmp_path / "run.log"
    main(args(video_path, tmp_path / "story.jpg", "-l", str(log)))
    assert "summarized" in log.read_text()


@pytest.mark.parametrize(
    "algo, nframes, extra, message",
    [
        ("bogus", "16", [], "unknown algorithm"),
        ("time", "0", [], "--nframes"),
        ("time", "16", ["-s", "0.5"], "cannot use time smoothing"),
        ("videosum_inception", "16", ["-c", "bogus"], "unknown clustering"),
        ("videosum_inception", "12", ["-c", "dissection"], "power of 2"),
        (
            "videosum_inception",
            "16",
            ["-c", "dissection", "-s", "0.5"],
            "cannot use time smoothing",
        ),
        ("videosum_inception", "16", ["-s", "1.5"], "time smoothing must be in"),
        ("videosum_inception", "16", ["--distance", "bogus"], "unknown distance"),
        ("videosum_inception", "16", ["--aggregation", "bogus"], "unknown aggregation"),
        ("videosum_scda", "16", ["-c", "dissection"], "only supports the l2sq distance"),
    ],
)
def test_invalid_arguments_exit_with_usage_error(
    video_path, tmp_path, capsys, algo, nframes, extra, message
):
    argv = ["-i", str(video_path), "-o", str(tmp_path / "story.jpg"), "-n", nframes,
            "-x", "640", "-y", "360", "-a", algo, *extra]  # fmt: skip
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_missing_input_exits_with_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(args(tmp_path / "missing.mp4", tmp_path / "story.jpg"))
    assert exc.value.code == 2
    assert "does not exist" in capsys.readouterr().err


def test_existing_output_is_refused(video_path, tmp_path, capsys):
    output = tmp_path / "story.jpg"
    output.touch()
    with pytest.raises(SystemExit):
        main(args(video_path, output))
    assert "already exists" in capsys.readouterr().err


def test_warnings_reach_the_log_file(video_path, tmp_path):
    log = tmp_path / "run.log"
    main(args(video_path, tmp_path / "story.jpg", "-f", "2", "-l", str(log)))
    assert "ignores --fps" in log.read_text()


def test_usage_does_not_import_the_slow_libraries():
    code = (
        "import sys, simplevideosum.run;"
        "print('torchcodec' in sys.modules, 'torchvision' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]
