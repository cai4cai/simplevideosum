# simplevideosum

## Setup

PyTorch is installed through an extra, so pick the flavour for your machine:

```bash
uv sync --extra cpu    # CPU-only wheels on Linux (CI); standard wheels on macOS (MPS)
uv sync --extra cuda   # NVIDIA GPUs (default PyPI CUDA wheels)
```

TorchCodec also needs FFmpeg shared libraries (`brew install ffmpeg` / `apt install ffmpeg`).

## Development

```bash
uv run pytest        # tests
uv run ruff check .  # lint
```
