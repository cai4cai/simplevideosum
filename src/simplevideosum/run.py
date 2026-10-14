"""Command line interface: summarise a video into a storyboard image.

Run with ``simplevideosum --help`` or ``python -m simplevideosum.run --help``.
"""

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from torchcodec.encoders import JpegEncoder, PngEncoder

from simplevideosum.algorithms import ALGORITHMS, PLANNED_ALGORITHMS
from simplevideosum.summarizer import VideoSummarizer

EXAMPLE = "simplevideosum -i video.mp4 -o storyboard.jpg -n 16 -x 1920 -y 1080 -a time -t 1"

HELP = {
    "-i": "Path to the input video.",
    "-o": "Path of the output image (.jpg, .jpeg or .png). Must not already exist.",
    "-n": "Number of frames in the collage.",
    "-x": "Width of the collage in pixels.",
    "-y": "Height of the collage in pixels (the time segmentation bar is added below it).",
    "-a": "Algorithm for frame selection. Available: "
    + ", ".join(ALGORITHMS)
    + ". Not implemented yet: "
    + ", ".join(PLANNED_ALGORITHMS)
    + ".",
    "-t": "Add a colour bar below the collage showing which key frame represents each "
    "moment of the video (1 to enable, 0 to disable; a bare -t enables it). Default: 0.",
    "-f": "Sampling frequency in frames per second. Key frames are selected among the video "
    "resampled at this rate. Ignored by the time algorithm, which only needs the video "
    "length. Default: use every frame of the video.",
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
    optional.add_argument("-f", "--fps", default=None, type=float, help=HELP["-f"])
    optional.add_argument("-l", "--log", default=None, type=Path, help=HELP["-l"])
    return parser


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
    if args.algo in PLANNED_ALGORITHMS:
        parser.error(
            f"algorithm '{args.algo}' is not implemented yet (available: {', '.join(ALGORITHMS)})"
        )
    if args.algo not in ALGORITHMS:
        parser.error(f"unknown algorithm '{args.algo}' (available: {', '.join(ALGORITHMS)})")


def save_image(image: torch.Tensor, path: Path) -> None:
    """Write a (3, H, W) uint8 RGB tensor as JPEG (quality 100) or PNG, chosen by extension."""
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

    summarizer = VideoSummarizer(
        args.algo,
        args.nframes,
        args.width,
        args.height,
        fps=args.fps,
        time_segmentation=bool(args.time_segmentation),
    )
    tic = time.time()
    storyboard = summarizer.summarize(args.input)
    save_image(storyboard, args.output)
    height, width = storyboard.shape[1:]
    logging.info("Video summarized in %.2f seconds.", time.time() - tic)
    logging.info("Saved %dx%d storyboard to %s", width, height, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
