from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


def pytest_addoption(parser):
    parser.addoption(
        "--pretrained-backbones",
        action="store_true",
        help="also run the test of the curated backbones with their pre-trained weights, which "
        "downloads about 500 MB in total (each file is deleted right after its model is checked)",
    )
    parser.addoption(
        "--backbones-filter",
        default="",
        help="with --pretrained-backbones, only the backbones whose name contains this "
        "(case-insensitive)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "pretrained_backbones: downloads weights, needs --pretrained-backbones"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--pretrained-backbones"):
        return
    skip = pytest.mark.skip(reason="manual test: pass --pretrained-backbones")
    for item in items:
        if "pretrained_backbones" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def video_path() -> Path:
    return DATA_DIR / "video.mp4"
