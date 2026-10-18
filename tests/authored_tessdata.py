"""Synthetic traineddata bytes for the authored subprocess adapter, not a real OCR model."""

from pathlib import Path


def authored_tessdata(root: Path) -> Path:
    directory = root / "tessdata"
    directory.mkdir(exist_ok=True)
    (directory / "eng.traineddata").write_bytes(b"authored language data version one")
    return directory
