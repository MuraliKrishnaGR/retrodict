import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _private_cache_dir(tmp_path, monkeypatch):
    """Tests never write into the real .cache (a test's deliberate crash once landed in last_error.txt)."""
    from app import config
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / ".cache")


@pytest.fixture(autouse=True)
def _no_model_downloads(monkeypatch):
    """Tests never download the 200 MB eraser model; the eraser falls back to OpenCV unless a test
    supplies a fake model."""
    from app import eraser
    monkeypatch.setattr(eraser, "_model", None)
    monkeypatch.setattr(eraser, "load", lambda: None)
