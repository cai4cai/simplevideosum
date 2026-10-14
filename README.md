# simplevideosum

A barebones video summarisation tool, inspired by
[videosum](https://github.com/luiscarlosgph/videosum). Given a video, it produces a
**single-image storyboard**: a collage of key frames, optionally with a colour bar underneath
that shows which key frame stands for each moment of the video.

Videos are decoded with [TorchCodec](https://github.com/pytorch/torchcodec) straight into
PyTorch tensors, and only the selected key frames are decoded.

Frame selection is made of explicit components, like in the original package but unpacked:

1. **backbone** (`--backbone`): the network giving convolutional activations: `inceptionv3`.
2. **aggregation** (`--aggregation`): how the activations are pooled into one or more descriptors
   per frame: `avgpool` (global average pooling, 2048-d for InceptionV3) or `scda` (Selective
   Convolutional Descriptor Aggregation, average and max pooled, 2 x 2048-d).
3. **normalisation** (`--normalisation`): applied to each pooled descriptor before they are
   concatenated into the frame signature: `none` or `l2`.
4. **distance** (`--distance`): between signatures: `l2sq` (squared L2) or `cosine`.
5. **clustering** (`-c/--clustering`): how signatures are grouped into key frames: `kmedoids`
   (default, as in videosum) or `dissection` (temporally contiguous segments, `l2sq` only).

`-a/--algo` is an alias for a configuration of these components; any component flag overrides
the one of the algorithm:

| `-a` | aggregation | normalisation | distance | clustering |
| --- | --- | --- | --- | --- |
| `time` | evenly spaced frames, never looks at the images (the components are ignored) | | | |
| `videosum_inception` | `avgpool` | `none` | `l2sq` | `kmedoids` |
| `videosum_scda` | `scda` | `none` | `cosine` | `kmedoids` |

All use the `inceptionv3` backbone. `videosum_scda` does not L2-normalise the pooled descriptors
before the cosine distance, while Wei et al. (2017) do: `-a videosum_scda --normalisation l2`.

## Setup

PyTorch is installed through an extra, so pick the flavour for your machine:

```bash
uv sync --extra cpu    # CPU-only wheels on Linux (CI); standard wheels on macOS (MPS)
uv sync --extra cuda   # NVIDIA GPUs (default PyPI CUDA wheels)
```

TorchCodec also needs the FFmpeg shared libraries (`brew install ffmpeg` / `apt install ffmpeg`).

The algorithms with signatures use the pre-trained InceptionV3 ImageNet weights. torchvision
downloads them on first use (about 100 MB, with a progress bar on stderr) and caches them in
`~/.cache/torch/hub/checkpoints`; later runs reuse the cached file. The network runs on CUDA, then
MPS, then CPU, whichever is available first.

## Usage

```bash
uv run simplevideosum -i tests/data/video.mp4 -o storyboard.jpg \
    -n 16 -x 1920 -y 1080 -a time -t 1
uv run simplevideosum -i tests/data/video.mp4 -o storyboard.jpg \
    -n 16 -x 1920 -y 1080 -a videosum_inception -f 2 -t 1
uv run simplevideosum -i tests/data/video.mp4 -o storyboard.jpg \
    -n 16 -x 1920 -y 1080 -a videosum_inception -c dissection -f 2 -t 1
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
| `-a`, `--algo` | Alias for a configuration of the components (table above): `time`, `videosum_inception` or `videosum_scda`. |
| `--backbone`, `--aggregation`, `--normalisation`, `--distance` | Override the corresponding component of the algorithm. |
| `-c`, `--clustering` | Clustering of the signatures: `kmedoids` or `dissection` (needs a power of 2 for `-n` and the `l2sq` distance). Overrides the algorithm's. |
| `-s`, `--time-smoothing` | In [0, 1]: blends the signature distances with the temporal distance between frames, 1 gives contiguous clusters. `kmedoids` only. Default: 0. |
| `-t`, `--time-segmentation` | Add the colour bar under the collage (`1` enables, `0` disables, bare `-t` enables). |
| `-f`, `--fps` | Sampling frequency in fps: key frames are chosen among the video resampled at this rate. the algorithms with signatures compute a signature for each resampled frame; `time` only needs the video length and ignores it. Default: every frame. |
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
- **Algorithms with signatures** (`videosum_inception`, `videosum_scda`): every frame of the video resampled at
  `--fps` gets a signature from the backbone, aggregation and normalisation (float32, computed on
  the GPU when available). The signatures are then clustered:
  - **`kmedoids`**: distances (`l2sq` or `cosine`) between signatures, min-max normalised and optionally
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
- **`scda` aggregation**: Selective Convolutional Descriptor Aggregation (Wei et al., 2017) of the
  8x8x2048 activations of InceptionV3's last convolutional block. The activations are summed over
  channels, the positions above the mean form a mask, and only its largest connected component
  (8-connected, holes filled, as in videosum) is kept. The descriptors at those positions are
  average- and max-pooled, so the signature mostly describes the main object rather than the
  background. The selection runs batched in PyTorch on the device. The SCDA+ multi-layer variant
  and the flipped-image ensemble are not implemented.
- **`cosine` distance**: one minus the cosine similarity of the signatures, min-max normalised
  (and blended with the temporal distance by `-s`) like `l2sq`. With `l2` normalised pooled
  descriptors it is the mean of the cosine distances of the descriptors.
- **Time segmentation bar**: each column is coloured by the key frame that is nearest in time
  to it, and black lines mark the key frame positions.

## Development

```bash
uv run pytest        # tests
uv run ruff check .  # lint
```
