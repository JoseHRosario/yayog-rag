"""Real CPU ONNX inference over an explicitly provisioned local model directory."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from fastembed import TextEmbedding

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Bound ONNX inference memory independently of the number of source passages.
EMBEDDING_BATCH_SIZE = 16


class LocalEmbedder:
    """No inference substitute and no automatic network/model download."""

    def __init__(self, model_dir: str | Path) -> None:
        self._directory = Path(model_dir).resolve()
        for name in (
            "model.onnx",
            "tokenizer.json",
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ):
            if not (self._directory / name).is_file():
                raise ValueError("model_unavailable")
        self._initial_identity = self.identity
        self._model = TextEmbedding(
            model_name=MODEL,
            specific_model_path=str(self._directory),
            cache_dir=str(self._directory.parent / "cache"),
            local_files_only=True,
            providers=["CPUExecutionProvider"],
            threads=2,
        )
        if self.identity != self._initial_identity:
            raise ValueError("model_changed")

    @property
    def identity(self) -> dict[str, str]:
        files = []
        for path in sorted(self._directory.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                files.append((str(path.relative_to(self._directory)), digest))
        artifact = hashlib.sha256(json.dumps(files, separators=(",", ":")).encode()).hexdigest()
        runtime = ";".join(
            f"{name}={version(name)}"
            for name in ("fastembed", "onnxruntime", "tokenizers", "numpy")
        )
        return {
            "model": MODEL,
            "version": runtime + ";symmetric-embed-v1",
            "artifact_sha256": artifact,
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.identity != self._initial_identity:
            raise ValueError("model_changed")
        vectors = [
            v.astype(float).tolist()
            for v in self._model.embed(texts, batch_size=EMBEDDING_BATCH_SIZE)
        ]
        if self.identity != self._initial_identity:
            raise ValueError("model_changed")
        return vectors
