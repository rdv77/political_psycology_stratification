from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional


class FileCache:
    def __init__(self, cache_dir: str = "cache") -> None:
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path_for_key(self, key: str) -> str:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        subdir = os.path.join(self.cache_dir, h[:2])
        os.makedirs(subdir, exist_ok=True)
        return os.path.join(subdir, f"{h}.json")

    def get(self, key: str) -> Optional[Any]:
        path = self._path_for_key(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._path_for_key(key)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)


