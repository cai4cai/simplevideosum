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
                   "--time-segmentation", "--fps", "--log", "example:"):  # fmt: skip
        assert option in out


@pytest.mark.parametrize("ext", [".jpg", ".png"])
def test_main_writes_storyboard(video_path, tmp_path, ext):
    output = tmp_path / f"story{ext}"
    assert main(args(video_path, output, "-t", "1", "-f", "2")) == 0
    assert tuple(read_image(output).shape) == (3, 360 + 32, 640)


def test_bare_time_segmentation_flag_enables_bar(video_path, tmp_path):
    output = tmp_path / "story.png"
    main(args(video_path, output, "-t"))
    assert read_image(output).shape[1] == 360 + 32


def test_log_file_is_written(video_path, tmp_path):
    log = tmp_path / "run.log"
    main(args(video_path, tmp_path / "story.jpg", "-l", str(log)))
    assert "summarized" in log.read_text()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda a, p: a.__setitem__(a.index("time"), "frechet"), "not implemented"),
        (lambda a, p: a.__setitem__(a.index("time"), "bogus"), "unknown algorithm"),
        (lambda a, p: a.__setitem__(1, "missing.mp4"), "does not exist"),
        (lambda a, p: a.__setitem__(a.index("-n") + 1, "0"), "--nframes"),
    ],
)
def test_invalid_arguments_exit_with_usage_error(video_path, tmp_path, capsys, mutate, message):
    argv = args(video_path, tmp_path / "story.jpg")
    mutate(argv, tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_existing_output_is_refused(video_path, tmp_path, capsys):
    output = tmp_path / "story.jpg"
    output.touch()
    with pytest.raises(SystemExit):
        main(args(video_path, output))
    assert "already exists" in capsys.readouterr().err
