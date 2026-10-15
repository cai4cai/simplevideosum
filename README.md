# simplevideosum

A barebones video summarisation tool, inspired by
[videosum](https://github.com/luiscarlosgph/videosum). Given a video, it produces a
**single-image storyboard**: a collage of key frames, optionally with a colour bar underneath
that shows which key frame stands for each moment of the video.

Videos are decoded with [TorchCodec](https://github.com/pytorch/torchcodec) straight into
PyTorch tensors, and only the selected key frames are decoded.

Frame selection is made of explicit components, like in the original package but unpacked:

1. **backbone** (`--backbone`): the network giving convolutional activations. Either
   `mobilenet` (the default: `MobileNet_V3_Large_Weights.DEFAULT`, 224 px input, 960 channels, 7x7
   map, 5M parameters), `inceptionv3` (299 px, 2048 channels, 8x8 map), `dinov2_vits14` (DINOv2
   ViT-S/14 through `torch.hub`; 224 px, 384 channels, a 16x16 map of patch tokens, 22M parameters),
   or any torchvision classification model: its weights, e.g.
   `MobileNet_V3_Large_Weights.IMAGENET1K_V2` or `EfficientNet_B3_Weights.IMAGENET1K_V1` (see the
   [table of weights](https://docs.pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights);
   `.DEFAULT` is a model's best), or a model name such as `mobilenet_v3_large` for its default
   weights. The maps are the last convolutional activations of the model, and the frames are
   resized and normalised as the weights require. The vision transformers (`vit_*`) and SqueezeNet
   are not supported, and DenseNet maps are taken before its last ReLU.
2. **aggregation** (`--aggregation`): how the activations are pooled into one or more descriptors
   per frame: `avgpool` (global average pooling, 960-d for the default backbone), `scda` (Selective
   Convolutional Descriptor Aggregation, average and max pooled, 2 x 960-d), `scda_thresh`
   (`scda` keeping every position above the threshold, with no morphology) or `scda_fill` (`scda`
   with the holes of the object mask filled, as in videosum).
3. **normalisation** (`--normalisation`): applied to each pooled descriptor before they are
   concatenated into the frame signature: `none` or `l2`.
4. **distance** (`--distance`): between signatures: `l2sq` (squared L2) or `cosine`.
5. **clustering** (`-c/--clustering`): how signatures are grouped into key frames: `kmedoids`
   (default, as in videosum), `dissection` (temporally contiguous segments by greedy bisection) or
   `fisher` (temporally contiguous segments of least total cost); both `l2sq` only.

`-a/--algo` is an alias for a configuration of these components; any component flag overrides
the one of the algorithm:

| `-a` | aggregation | normalisation | distance | clustering |
| --- | --- | --- | --- | --- |
| `time` | evenly spaced frames, never looks at the images (the components are ignored) | | | |
| `videosum_inception` | `avgpool` | `none` | `l2sq` | `kmedoids` |
| `videosum_scda` | `scda_fill` | `none` | `cosine` | `kmedoids` |

All use the default `mobilenet` backbone unless `--backbone` says otherwise (the videosum package uses `inceptionv3`). `videosum_scda` follows the videosum package, which departs
from Wei et al. (2017) in two ways: it fills the holes of the object mask, and it does not
L2-normalise the pooled descriptors before the cosine distance. The paper's version is
`-a videosum_scda --aggregation scda --normalisation l2`.

## Setup

PyTorch is installed through an extra, so pick the flavour for your machine:

```bash
uv sync --extra cpu    # CPU-only wheels on Linux (CI); standard wheels on macOS (MPS)
uv sync --extra cuda   # NVIDIA GPUs (default PyPI CUDA wheels)
```

TorchCodec also needs the FFmpeg shared libraries (`brew install ffmpeg` / `apt install ffmpeg`).

The algorithms with signatures use pre-trained weights, downloaded on first use (about 100 MB for
the default MobileNetV3-Large, 109 MB for InceptionV3, 85 MB for DINOv2, 10-300 MB for the
other torchvision models) with a progress
bar on stderr, and cached in `~/.cache/torch/hub/checkpoints`; later runs reuse the cached file.
Delete the files you no longer need from that folder. DINOv2 is loaded through `torch.hub` from
the authors' repository, so its first use also downloads their code from GitHub
(`facebookresearch/dinov2`, main branch, trusted by the `trust_repo` flag) to `~/.cache/torch/hub`.
The network runs on CUDA, then MPS, then CPU, whichever is available first.

The tests use random weights and download nothing, except one end-to-end test that downloads the
default weights (21 MB) into a temporary directory that goes with the test.
`uv run pytest --pretrained-backbones` also checks a short list of backbones with their real
weights (about 500 MB), one at a time in a temporary directory that is emptied after each, so the
torch cache stays untouched; `--backbones-filter NAME` restricts it.

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
| `-c`, `--clustering` | Clustering of the signatures: `kmedoids`, `dissection` (needs a power of 2 for `-n` and the `l2sq` distance) or `fisher` (`l2sq` distance, at most 10,000 sampled frames). Overrides the algorithm's. |
| `-s`, `--time-smoothing` | In [0, 1]: blends the signature distances with the temporal distance between frames, 1 gives contiguous clusters. `kmedoids` only. Default: 0. |
| `-t`, `--time-segmentation` | Add the colour bar under the collage and a border of the same colour around each key frame (`1` enables, `0` disables, bare `-t` enables). |
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

  - **`fisher`**: the same segment cost, but the `n` contiguous segments of least total cost are
    found exactly with Fisher's 1-D segmentation (dynamic programming over the cut points,
    `F[j][e] = min_s F[j-1][s] + cost(s, e)`). Segments have any length, so a long static shot gets
    one segment and a busy moment several, and `n` can be any number. It evaluates an (N + 1) x
    (N + 1) cost matrix on the device (400 MB for 10,000 frames, the limit) in n x N^2 operations,
    so lower `-f` for long videos. The key frame is chosen as for `dissection`.

  The greedy bisection can only produce balanced hierarchies, so on the demo video it spends half
  of the segments on the first 17 seconds. It is kept as an experiment, `kmedoids` is the baseline.
- **`scda` aggregation**: Selective Convolutional Descriptor Aggregation (Wei et al., 2017) of the
  activations of the last convolutional block of the backbone (7x7x960 for the default one). The
  activations are summed over channels, the positions above the mean form a mask, and only its largest 8-connected component
  is kept. The descriptors at those positions are average- and max-pooled, so the signature mostly
  describes the main object rather than the background. The selection runs batched on the device
  (CPU, CUDA or MPS) with the pure PyTorch morphology of `simplevideosum.morphology`: components
  of maps of up to 256 positions, such as these 8x8 maps, are found by transitive closure with
  batched matrix products (no data-dependent loop or host synchronisation), larger maps by label
  propagation. `scda_thresh` skips the connected component step and keeps every position above
  the mean, and `scda_fill` also fills the holes of the mask, which the paper does not do but
  videosum does (`SCDA(largest_component=..., fill_holes=...)` in code). The SCDA+ multi-layer
  variant and the flipped-image ensemble are not implemented.
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
