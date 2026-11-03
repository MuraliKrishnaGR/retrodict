import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _private_cache_dir(tmp_path, monkeypatch):
    """Tests never write into the real .cache (a test's deliberate crash once landed in last_error.txt)."""
    from app import config
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / ".cache")
