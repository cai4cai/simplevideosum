"""Command line interface: summarise a video into a storyboard image.

Run with ``simplevideosum --help`` or ``python -m simplevideosum.run --help``.

TorchCodec and torchvision take a couple of seconds to import, so they are only imported once the
arguments have been validated, which keeps ``--help`` and usage errors fast.
"""

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from simplevideosum.algorithms import (
    ALGORITHMS,
    COMPONENTS,
    DEFAULT_CLUSTERING,
    PIPELINE_COMPONENTS,
    resolve,
)

EXAMPLE = "simplevideosum -i video.mp4 -o storyboard.jpg -n 16 -x 1920 -y 1080 -a time -t 1"

HELP = {
    "-i": "Path to the input video.",
    "-o": "Path of the output image (.jpg, .jpeg or .png). Must not already exist.",
    "-n": "Number of frames in the collage.",
    "-x": "Width of the collage in pixels.",
    "-y": "Height of the collage in pixels (the time segmentation bar is added below it).",
    "-a": "Algorithm for frame selection: a named configuration of the components below. "
    "Available: "
    + ", ".join(ALGORITHMS)
    + ". time picks evenly spaced frames (no feature extraction, ignores the components), "
    "videosum_inception is the globally pooled InceptionV3 activations compared with squared L2, "
    "videosum_scda is SCDA (Wei et al., 2017) compared with the cosine distance, as in the "
    "videosum package.",
    "--backbone": "Network producing the convolutional activations; overrides the algorithm's. "
    "Available: " + ", ".join(COMPONENTS["backbone"]) + ", the name of a torchvision "
    "classification model (e.g. mobilenet_v3_large, for its default weights), or its weights "
    "(e.g. MobileNet_V3_Large_Weights.IMAGENET1K_V2; see the table of classification weights in "
    "the torchvision documentation). The frames are resized and normalised as the weights "
    "require.",
    "--aggregation": "How the activations are pooled into descriptors; overrides the algorithm's. "
    "Available: avgpool (global average pooling), scda (Selective Convolutional Descriptor "
    "Aggregation: average and max pooling of the descriptors of the largest connected component "
    "above the activation threshold), scda_thresh (scda keeping every position above the "
    "threshold, no morphology), scda_fill (scda with the holes of the object mask filled, as in "
    "videosum).",
    "--normalisation": "Normalisation of each pooled descriptor before they are concatenated; "
    "overrides the algorithm's. Available: none, l2.",
    "--distance": "Distance between signatures; overrides the algorithm's. Available: l2sq "
    "(squared L2), cosine (one minus the cosine similarity).",
    "-c": "Clustering method for the signatures (ignored by time); overrides the algorithm's. "
    "Available: kmedoids (k-medoids with PAM, as in videosum), dissection (temporally contiguous "
    "segments by recursive bisection; the number of frames must be a power of 2, and only "
    "l2sq is supported), fisher (temporally contiguous segments of least total cost, found "
    "exactly by dynamic programming; any number of frames, only l2sq, memory grows as the "
    "square of the number of sampled frames). "
    f"Default: {DEFAULT_CLUSTERING}.",
    "-s": "Time smoothing in [0, 1]: blends the signature distances with the temporal distance "
    "between frames, 1 gives evenly spaced contiguous clusters. Only for kmedoids. Default: 0.",
    "-t": "Add a colour bar below the collage showing which key frame represents each "
    "moment of the video (1 to enable, 0 to disable; a bare -t enables it). Default: 0.",
    "-f": "Sampling frequency in frames per second. Key frames are selected among the video "
    "resampled at this rate (inception computes a signature for each of them). Ignored by the "
    "time algorithm, which only needs the video length. Default: use every frame of the video.",
    "-l": "Path of a log file to append to. Default: log to the terminal only.",
}


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser; the options mirror ``videosum.run``."""
    parser = argparse.ArgumentParser(
        prog="simplevideosum",
        description="Summarise a video into a single-image storyboard of key frames.\n\n"
        "Frames are tiled in a collage of the requested size; with --time-segmentation a "
        "colour bar underneath shows which key frame stands for each moment of the video.",
        epilog=f"example:\n  {EXAMPLE}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")
    required.add_argument("-i", "--input", required=True, type=Path, help=HELP["-i"])
    required.add_argument("-o", "--output", required=True, type=Path, help=HELP["-o"])
    required.add_argument("-n", "--nframes", required=True, type=int, help=HELP["-n"])
    required.add_argument("-x", "--width", required=True, type=int, help=HELP["-x"])
    required.add_argument("-y", "--height", required=True, type=int, help=HELP["-y"])
    required.add_argument("-a", "--algo", required=True, type=str, help=HELP["-a"])
    optional.add_argument(
        "-t", "--time-segmentation", nargs="?", const=1, default=0, type=int, help=HELP["-t"]
    )
    for name in PIPELINE_COMPONENTS:
        optional.add_argument(f"--{name}", default=None, type=str, help=HELP[f"--{name}"])
    optional.add_argument("-c", "--clustering", default=None, type=str, help=HELP["-c"])
    optional.add_argument("-s", "--time-smoothing", default=0.0, type=float, help=HELP["-s"])
    optional.add_argument("-f", "--fps", default=None, type=float, help=HELP["-f"])
    optional.add_argument("-l", "--log", default=None, type=Path, help=HELP["-l"])
    return parser


def components(args: argparse.Namespace) -> dict[str, str | None]:
    """The component overrides given on the command line."""
    return {name: getattr(args, name) for name in PIPELINE_COMPONENTS}


def validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Exit with a usage error if the arguments are inconsistent."""
    if not args.input.is_file():
        parser.error(f"input video '{args.input}' does not exist or is not a file")
    if args.output.exists():
        parser.error(f"output '{args.output}' already exists, remove it or choose another path")
    if args.output.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        parser.error("output must have a .jpg, .jpeg or .png extension")
    for name in ("nframes", "width", "height"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be >= 1")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be > 0")
    try:
        resolve(args.algo, args.clustering, args.nframes, args.time_smoothing, **components(args))
    except ValueError as error:
        parser.error(str(error))


def save_image(image: torch.Tensor, path: Path) -> None:
    """Write a (3, H, W) uint8 RGB tensor as JPEG (quality 100) or PNG, chosen by extension."""
    from torchcodec.encoders import JpegEncoder, PngEncoder

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".png":
        PngEncoder(image).to_file(path)
    else:
        JpegEncoder(image).to_file(path, quality=100)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate(args, parser)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log is not None:
        handlers.append(logging.FileHandler(args.log))
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(message)s", handlers=handlers, force=True
    )

    from simplevideosum.summarizer import VideoSummarizer

    summarizer = VideoSummarizer(
        args.algo,
        args.nframes,
        args.width,
        args.height,
        fps=args.fps,
        time_segmentation=bool(args.time_segmentation),
        clustering=args.clustering,
        time_smoothing=args.time_smoothing,
        **components(args),
    )
    tic = time.time()
    try:
        storyboard = summarizer.summarize(args.input)
    except ValueError as error:  # e.g. fewer sampled frames than requested segments
        parser.error(str(error))
    save_image(storyboard, args.output)
    height, width = storyboard.shape[1:]
    logging.info("Video summarized in %.2f seconds.", time.time() - tic)
    logging.info("Saved %dx%d storyboard to %s", width, height, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
