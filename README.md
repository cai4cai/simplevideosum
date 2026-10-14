# simplevideosum

A barebones video summarisation tool, inspired by
[videosum](https://github.com/luiscarlosgph/videosum). Given a video, it produces a
**single-image storyboard**: a collage of key frames, optionally with a colour bar underneath
that shows which key frame stands for each moment of the video.

Videos are decoded with [TorchCodec](https://github.com/pytorch/torchcodec) straight into
PyTorch tensors, and only the selected key frames are decoded.

For now only the `time` algorithm is available: key frames are evenly spaced along the video.

## Setup

PyTorch is installed through an extra, so pick the flavour for your machine:

```bash
uv sync --extra cpu    # CPU-only wheels on Linux (CI); standard wheels on macOS (MPS)
uv sync --extra cuda   # NVIDIA GPUs (default PyPI CUDA wheels)
```

TorchCodec also needs the FFmpeg shared libraries (`brew install ffmpeg` / `apt install ffmpeg`).

## Usage

```bash
uv run simplevideosum -i tests/data/video.mp4 -o storyboard.jpg \
    -n 16 -x 1920 -y 1080 -a time -t 1
```

This writes a 1920x1112 image: a 4x4 grid of 16 frames filling 1920x1080, plus a 32 pixel
time segmentation bar. `python -m simplevideosum.run` is equivalent. Run
`uv run simplevideosum --help` for the full, self-contained documentation.

| Option | Description |
| --- | --- |
| `-i`, `--input` | Path to the input video. |
| `-o`, `--output` | Output image (`.jpg`, `.jpeg` or `.png`). It must not already exist. |
| `-n`, `--nframes` | Number of frames in the collage. |
| `-x`, `--width` | Width of the collage in pixels. |
| `-y`, `--height` | Height of the collage in pixels (the time segmentation bar is added below it). |
| `-a`, `--algo` | Frame selection algorithm. Only `time` is implemented; `frechet` is planned. |
| `-t`, `--time-segmentation` | Add the colour bar under the collage (`1` enables, `0` disables, bare `-t` enables). |
| `-f`, `--fps` | Sampling frequency in fps: key frames are chosen among the video resampled at this rate. Accepted for compatibility but ignored by `time`, which only needs the video length. Default: every frame. |
| `-l`, `--log` | Log file to append to. Default: terminal only. |

The options mirror `python -m videosum.run` from the original package. The ones that do not
apply yet are left out: `-s` (time smoothing, rejected by the `time` algorithm anyway),
`-p` (processes, only used for batch processing of folders) and `-m` (FID metric).
The input must be a single video file for now.

### How the storyboard is built

- **Tiles**: the largest tiles with the aspect ratio of the collage such that `n` of them fit
  in `x` by `y` pixels. Frames are resized to the tile size and placed row by row.
- **`time` algorithm**: key frames are `n` evenly spaced indices over the whole video. No frame is
  inspected, and only those `n` frames are decoded (`--fps` has no effect).
- **Time segmentation bar**: each column is coloured by the key frame that is nearest in time
  to it, and black lines mark the key frame positions.

## Development

```bash
uv run pytest        # tests
uv run ruff check .  # lint
```
