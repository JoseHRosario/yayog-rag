"""Strict geometry and private destination guards for opt-in source selections."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pymupdf

from .core import _digest, _json
from .inference_client import private_path

_rect: Any = pymupdf.Rect


def private_artifact(path: str | Path, error: str = "unsafe_review_path") -> Path:
    try:
        return private_path(path)
    except ValueError:
        raise ValueError(error) from None


def selection_box(selector: dict[str, Any], doc: Any, start: int, end: int) -> Any:
    try:
        page, coords = selector["pdf_page"], selector["bbox"]
        if (
            type(page) is not int
            or not start <= page <= end
            or not isinstance(coords, list)
            or len(coords) != 4
            or any(type(v) not in {int, float} or not math.isfinite(v) for v in coords)
        ):
            raise ValueError("invalid_source_selection")
        box = _rect(coords)
        if box.is_empty or box.is_infinite or not doc[page - 1].rect.contains(box):
            raise ValueError("invalid_source_selection")
        return box
    except (KeyError, TypeError, OverflowError):
        raise ValueError("invalid_source_selection") from None


def recovery_artifacts(path: Path, recovered: dict[str, Any]) -> set[Path]:
    return {
        path.resolve(),
        Path(recovered["source_path"]).resolve(),
        *((path.parent / r["crop"]).resolve() for r in recovered["regions"]),
    }


def publish_json(
    path: Path, value: dict[str, Any], previous: bytes | None, error: str = "stale_excerpts"
) -> None:
    """No-clobber creation; exact-byte compare before replacing an owned prior revision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".excerpt-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(_json({**value, "seal": _digest(_json(value).encode())}))
        if previous is None:
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise ValueError(error) from None
        else:
            if path.read_bytes() != previous:
                raise ValueError(error)
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_crop(path: Path, image: bytes) -> None:
    """Publish a complete content-addressed crop without overwriting concurrent owners."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".crop-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(image)
        try:
            os.link(temporary, path)
        except FileExistsError:
            from .core import _hash_file

            if _hash_file(path) != _digest(image):
                raise ValueError("stale_recovery") from None
    finally:
        temporary.unlink(missing_ok=True)
