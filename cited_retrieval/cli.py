"""Safe JSON CLI. Retrieval stdout deliberately contains private evidence."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import pymupdf

from . import ingest, inspect_source, report, search

SAFE_ERRORS = {
    "invalid_section",
    "unsafe_index_path",
    "review_required",
    "no_eligible_passages",
    "invalid_embeddings",
    "invalid_model_identity",
    "source_changed",
    "model_changed",
    "stale_source",
    "stale_corpus",
    "stale_settings",
    "incompatible_model",
    "incompatible_dimensions",
    "invalid_query",
    "model_unavailable",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import reviewed prose and retrieve cited evidence."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "import"):
        sub = commands.add_parser(command)
        sub.add_argument("pdf")
        sub.add_argument("--start", type=int, required=True)
        sub.add_argument("--end", type=int, required=True)
        if command == "import":
            sub.add_argument("--index", required=True)
            sub.add_argument("--approve-sha256", required=True)
            sub.add_argument(
                "--approve-pages", required=True, help="Comma-separated PDF page numbers"
            )
            sub.add_argument("--model-dir", required=True)
    sub = commands.add_parser("report")
    sub.add_argument("--index", required=True)
    sub = commands.add_parser("search")
    sub.add_argument("question")
    sub.add_argument("--index", required=True)
    sub.add_argument("--model-dir", required=True)
    sub.add_argument("--limit", type=int, default=5)
    return parser


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "inspect":
        return inspect_source(args.pdf, args.start, args.end)
    if args.command == "report":
        return report(args.index)
    if args.command == "import":
        preview = inspect_source(args.pdf, args.start, args.end)
        approval = {
            "sha256": args.approve_sha256,
            "pages": [int(n) for n in args.approve_pages.split(",")],
        }
        if approval != {
            "sha256": preview["sha256"],
            "pages": list(range(args.start, args.end + 1)),
        }:
            raise ValueError("review_required")
    from .local_embedder import LocalEmbedder

    embedder = LocalEmbedder(args.model_dir)
    if args.command == "import":
        return ingest(args.pdf, args.index, args.start, args.end, embedder, approval=approval)
    return search(args.index, args.question, embedder, limit=args.limit)


def main(argv: list[str] | None = None) -> int:
    pymupdf.TOOLS.mupdf_display_errors(False)  # type: ignore[no-untyped-call]
    pymupdf.TOOLS.mupdf_display_warnings(False)  # type: ignore[no-untyped-call]
    try:
        with redirect_stderr(io.StringIO()):
            args = _parser().parse_args(argv)
    except SystemExit as exc:
        if exc.code == 0:
            return 0
        print(json.dumps({"status": "error", "error": "invalid_arguments"}))
        return 2
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = _execute(args)
    except Exception as exc:  # noqa: BLE001 - privacy boundary: never leak provider error text
        code = (
            str(exc)
            if isinstance(exc, ValueError) and str(exc) in SAFE_ERRORS
            else "operation_failed"
        )
        print(json.dumps({"status": "error", "error": code}))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0
