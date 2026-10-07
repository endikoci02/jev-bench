"""Content-addressed disk cache for API calls. Reruns hit the cache and cost $0."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def cache_key(
    *,
    provider: str,
    model: str,
    payload: Any,
    params: dict[str, Any],
    sample_index: int = 0,
) -> str:
    """Stable sha256 over everything that can change the model's answer.

    `payload` is the prompt/state plus questions; `sample_index` separates
    repeated samples (self-consistency) of an otherwise identical request.
    """
    blob = json.dumps(
        {
            "provider": provider,
            "model": model,
            "payload": payload,
            "params": params,
            "sample_index": sample_index,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        return data

    def put(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        atomic_write_json(path, value)

    def __contains__(self, key: str) -> bool:
        return self._path(key).exists()


def atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON via temp file + os.replace so a crash never leaves a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
