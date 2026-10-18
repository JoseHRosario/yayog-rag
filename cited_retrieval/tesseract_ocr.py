"""Local Tesseract subprocess adapter; no downloads, rewriting, or remote inference."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .core import _digest, _hash_file, _json
from .policy import OCR_DPI, OCR_PSM


class TesseractOCR:
    def __init__(
        self,
        *,
        executable: str | Path = "tesseract",
        tessdata: str | Path | None = None,
        language: str = "eng",
        timeout: float = 60,
        env: dict[str, str] | None = None,
    ) -> None:
        if (
            not math.isfinite(timeout)
            or timeout <= 0
            or not language
            or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+"
                for c in language
            )
        ):
            raise ValueError("invalid_ocr_settings")
        resolved = shutil.which(str(executable))
        if resolved is None:
            raise ValueError("ocr_unavailable")
        self.executable = Path(resolved).resolve()
        self.env = {**os.environ, **(env or {})}
        configured_data = tessdata or self.env.get("TESSDATA_PREFIX")
        if not configured_data:
            raise ValueError("ocr_unavailable")
        self.tessdata = Path(configured_data).resolve()
        self.language = language
        self.timeout = timeout
        for name in self.language.split("+"):
            if not (self.tessdata / f"{name}.traineddata").is_file():
                raise ValueError("ocr_unavailable")
        self.version = self._run(["--version"]).splitlines()[0]

    def _run(self, arguments: list[str]) -> str:
        try:
            result = subprocess.run(
                [str(self.executable), *arguments],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=self.timeout,
                check=True,
                env=self.env,
            )
            return result.stdout.decode("utf-8")
        except (OSError, subprocess.SubprocessError, UnicodeError):
            raise ValueError("ocr_failed") from None

    @property
    def identity(self) -> dict[str, Any]:
        try:
            data = _digest(
                _json(
                    [
                        [language, _hash_file(self.tessdata / f"{language}.traineddata")]
                        for language in self.language.split("+")
                    ]
                ).encode()
            )
            return {
                "engine": "tesseract",
                "version": self.version,
                "executable_sha256": _hash_file(self.executable),
                "tessdata_sha256": data,
                "language": self.language,
                "dpi": OCR_DPI,
                "psm": OCR_PSM,
                "timeout_seconds": self.timeout,
            }
        except OSError:
            raise ValueError("ocr_unavailable") from None

    def recognize(self, image: Path) -> str:
        arguments = [
            str(image),
            "stdout",
            "--dpi",
            str(OCR_DPI),
            "--psm",
            str(OCR_PSM),
            "-l",
            self.language,
        ]
        arguments.extend(["--tessdata-dir", str(self.tessdata)])
        return self._run(arguments)
