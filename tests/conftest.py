from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def video_path() -> Path:
    return DATA_DIR / "video.mp4"
