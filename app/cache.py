"""Disk cache: the same request is never paid for twice."""
import hashlib
import json
from pathlib import Path


def make_key(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            h.update(bytes(p))
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
        h.update(b"|")
    return h.hexdigest()[:32]


class DiskCache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, kind: str, key: str, ext: str) -> Path:
        return self.root / kind / f"{key}.{ext}"

    def get_json(self, kind: str, key: str):
        p = self._path(kind, key, "json")
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def put_json(self, kind: str, key: str, value) -> None:
        p = self._path(kind, key, "json")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
        tmp.replace(p)

    def path_for(self, kind: str, key: str, ext: str) -> Path:
        p = self._path(kind, key, ext)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
