"""SegLST and cache I/O helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json_atomic(data: Any, path: str | Path, *, indent: int | None = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)
        file.write("\n")
    temporary.replace(path)


def audio_files(directory: str | Path) -> list[Path]:
    paths = sorted(Path(directory).glob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"音频目录中没有 .wav 文件: {directory}")
    stems = [path.stem for path in paths]
    if len(stems) != len(set(stems)):
        raise ValueError("音频文件名去除扩展名后必须唯一")
    return paths
