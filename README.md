# simplevideosum

A barebones video summarisation tool, inspired by
[videosum](https://github.com/luiscarlosgph/videosum). Given a video, it produces a
**single-image storyboard**: a collage of key frames, optionally with a colour bar underneath
that shows which key frame stands for each moment of the video.

Videos are decoded with [TorchCodec](https://github.com/pytorch/torchcodec) straight into
PyTorch tensors, and only the selected key frames are decoded.

Frame selection is split in two choices, like in the original package but made explicit:

- **`-a/--algo`: how frames are represented.** `time` selects evenly spaced frames and never
  looks at the images. `inception` represents each frame by a 2048-d signature from a
  pre-trained InceptionV3.
- **`-c/--clustering`: how signatures are grouped into key frames** (only for `inception`).
  `kmedoids` (default, as in videosum) and `dissection` (temporally contiguous segments).

## Setup

PyTorch is installed through an extra, so pick the flavour for your machine:

```bash
uv sync --extra cpu    # CPU-only wheels on Linux (CI); standard wheels on macOS (MPS)
uv sync --extra cuda   # NVIDIA GPUs (default PyPI CUDA wheels)
```

TorchCodec also needs the FFmpeg shared libraries (`brew install ffmpeg` / `apt install ffmpeg`).

The `inception` algorithm uses the pre-trained InceptionV3 ImageNet weights. torchvision downloads
them on first use (about 100 MB, with a progress bar on stderr) and caches them in
`~/.cache/torch/hub/checkpoints`; later runs reuse the cached file. The network runs on CUDA, then
MPS, then CPU, whichever is available first.

## Usage

```bash
uv run simplevideosum -i tests/data/video.mp4 -o storyboard.jpg \
    -n 16 -x 1920 -y 1080 -a time -t 1
uv run simplevideosum -i tests/data/video.mp4 -o storyboard.jpg \
    -n 16 -x 1920 -y 1080 -a inception -f 2 -t 1
uv run simplevideosum -i tests/data/video.mp4 -o storyboard.jpg \
    -n 16 -x 1920 -y 1080 -a inception -c dissection -f 2 -t 1
```

The first command writes a 1920x1112 image: a 4x4 grid of 16 frames filling 1920x1080, plus a 32 pixel
time segmentation bar. `python -m simplevideosum.run` is equivalent. Run
`uv run simplevideosum --help` for the full, self-contained documentation.

| Option | Description |
| --- | --- |
| `-i`, `--input` | Path to the input video. |
| `-o`, `--output` | Output image (`.jpg`, `.jpeg` or `.png`). It must not already exist. |
| `-n`, `--nframes` | Number of frames in the collage. |
| `-x`, `--width` | Width of the collage in pixels. |
| `-y`, `--height` | Height of the collage in pixels (the time segmentation bar is added below it). |
| `-a`, `--algo` | Frame representation: `time` (no features) or `inception` (InceptionV3 signatures). `frechet` is planned. |
| `-c`, `--clustering` | Clustering of the signatures, for `inception` only: `kmedoids` (default) or `dissection` (needs a power of 2 for `-n`). |
| `-s`, `--time-smoothing` | In [0, 1]: blends the signature distances with the temporal distance between frames, 1 gives contiguous clusters. `kmedoids` only. Default: 0. |
| `-t`, `--time-segmentation` | Add the colour bar under the collage (`1` enables, `0` disables, bare `-t` enables). |
| `-f`, `--fps` | Sampling frequency in fps: key frames are chosen among the video resampled at this rate. `inception` computes a signature for each resampled frame; `time` only needs the video length and ignores it. Default: every frame. |
| `-l`, `--log` | Log file to append to. Default: terminal only. |

The options mirror `python -m videosum.run` from the original package, plus `-c` which makes the
clustering explicit. The ones that do not apply yet are left out: `-p` (processes, only used for
batch processing of folders) and `-m` (FID metric).
The input must be a single video file for now.

### How the storyboard is built

- **Tiles** (the collage can be a few pixels smaller than requested, as tile sizes are rounded): the largest tiles with the aspect ratio of the collage such that `n` of them fit
  in `x` by `y` pixels. Frames are resized to the tile size and placed row by row.
- **`time` algorithm**: key frames are `n` evenly spaced indices over the whole video. No frame is
  inspected, and only those `n` frames are decoded (`--fps` has no effect).
- **`inception` algorithm**: every frame of the video resampled at `--fps` gets a 2048-d
  signature from InceptionV3 (float32, computed on the GPU when available). The signatures are then
  clustered:
  - **`kmedoids`**: squared L2 distances between signatures, min-max normalised and optionally
    blended with the temporal distance (`-s`), as in videosum. Medoids are initialised with
    k-medoids++ (fixed seed) and refined with PAM swaps, all evaluated at once for each iteration
    (FastPAM1 formulation), so `n` can be any number. Clusters are not temporally contiguous, and
    the key frames are the medoids.
  - **`dissection`**: the cost of a segment is the sum of squared distances from its signatures to
    their mean. With prefix sums of the signatures (centred, for numerical stability) and of their
    squared norms, the cost of any segment is O(D) to evaluate, so no quadratic pass over the frames
    is needed and the full N x N cost matrix is never built. The video is split recursively: each
    segment is cut where the summed cost of the two halves is smallest, until `n` segments are
    obtained, which is why `n` must be a power of 2. The key frame of a segment is the frame whose
    signature is closest to the segment mean.

  The greedy bisection can only produce balanced hierarchies, so on the demo video it spends half
  of the segments on the first 17 seconds. It is kept as an experiment, `kmedoids` is the baseline.
- **Time segmentation bar**: each column is coloured by the key frame that is nearest in time
  to it, and black lines mark the key frame positions.

## Development

```bash
uv run pytest        # tests
uv run ruff check .  # lint
```
