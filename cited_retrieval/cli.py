"""Safe JSON CLI. Retrieval stdout deliberately contains private evidence."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import pymupdf

from . import ingest, inspect_review, inspect_source, recover, report, review, search
from .answers import answer, finish_answer, prepare_answer
from .coverage import draft_coverage, inspect_coverage
from .exercises import draft_exercises, inspect_exercises, review_exercise
from .openai_completion import OpenAICompletion
from .programmes import draft_programmes, inspect_programmes, review_programme

SAFE_ERRORS = {
    "unsafe_coverage_path",
    "stale_coverage",
    "invalid_coverage_manifest",
    "invalid_evaluation_location",
    "stale_coverage_evidence",
    "unsafe_answer_path",
    "invalid_inference_config",
    "connection_not_authorized",
    "stale_programmes",
    "invalid_programme_layout",
    "invalid_session_filter",
    "stale_exercises",
    "invalid_exercise_layout",
    "invalid_exercise_image",
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
    "stale_recovery",
    "unsafe_review_path",
    "review_revision_mismatch",
    "invalid_review_state",
    "empty_recovery",
    "invalid_ocr_settings",
    "ocr_unavailable",
    "ocr_failed",
    "ocr_changed",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import reviewed prose and retrieve cited evidence."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser(
        "draft-coverage", help="Draft private source-bound coverage accounting"
    )
    sub.add_argument("pdf")
    sub.add_argument(
        "--audit", required=True, help="New private JSON path outside every Git checkout"
    )
    sub.add_argument(
        "--inventory",
        help="Private source-bound manual-record JSON; changes require a new draft and reconciliation",
    )
    sub.add_argument("--index", action="append", default=[])
    sub.add_argument("--exercise-catalog", action="append", default=[])
    sub.add_argument("--programme-catalog", action="append", default=[])
    sub.add_argument("--recovery", action="append", default=[])
    sub = commands.add_parser(
        "inspect-coverage", help="Revalidate private coverage and registered inventory"
    )
    sub.add_argument(
        "audit", help="Private manifest outside every Git checkout, including after relocation"
    )
    sub = commands.add_parser("draft-exercises")
    sub.add_argument("pdf")
    sub.add_argument("--catalog", required=True)
    sub.add_argument("--layout", required=True)
    sub.add_argument("--start", type=int, required=True)
    sub.add_argument("--end", type=int, required=True)
    sub.add_argument("--recovery")
    sub = commands.add_parser("inspect-exercises")
    sub.add_argument("catalog")
    sub = commands.add_parser("review-exercise")
    sub.add_argument("catalog")
    sub.add_argument("--record", required=True)
    sub.add_argument("--revision", required=True)
    sub.add_argument("--state", choices=["accepted", "rejected", "pending"], required=True)
    sub = commands.add_parser("draft-programmes")
    sub.add_argument("pdf")
    sub.add_argument("--catalog", required=True)
    sub.add_argument("--layout", required=True)
    sub.add_argument("--start", type=int, required=True)
    sub.add_argument("--end", type=int, required=True)
    sub.add_argument("--recovery")
    sub.add_argument("--exercise-catalog", action="append", default=[])
    sub = commands.add_parser("inspect-programmes")
    sub.add_argument("catalog")
    sub = commands.add_parser("review-programme")
    sub.add_argument("catalog")
    sub.add_argument("--record", required=True)
    sub.add_argument("--revision", required=True)
    sub.add_argument("--state", choices=["accepted", "rejected", "pending"], required=True)
    for command in ("inspect", "import", "recover"):
        sub = commands.add_parser(command)
        sub.add_argument("pdf")
        sub.add_argument("--start", type=int, required=True)
        sub.add_argument("--end", type=int, required=True)
        if command == "recover":
            sub.add_argument("--bundle", required=True)
            sub.add_argument("--tesseract", default="tesseract")
            sub.add_argument("--tessdata-dir")
            sub.add_argument("--language", default="eng")
            sub.add_argument("--timeout", type=float, default=60)
        if command == "import":
            sub.add_argument("--index", required=True)
            sub.add_argument("--approve-sha256", required=True)
            sub.add_argument(
                "--approve-pages", required=True, help="Comma-separated PDF page numbers"
            )
            sub.add_argument("--model-dir", required=True)
            sub.add_argument("--recovery", help="Private reviewed recovery bundle")
            sub.add_argument("--exercises", help="Private reviewed exercise catalog")
            sub.add_argument("--programmes", help="Private reviewed programme catalog")
    sub = commands.add_parser("inspect-review", aliases=["inspectreview"])
    sub.add_argument("bundle")
    sub = commands.add_parser("review")
    sub.add_argument("bundle")
    sub.add_argument("--region", required=True)
    sub.add_argument("--revision", required=True)
    sub.add_argument("--state", choices=["accepted", "rejected", "pending"], required=True)
    sub = commands.add_parser("report")
    sub.add_argument("--index", required=True)
    sub = commands.add_parser("search")
    sub.add_argument("question", nargs="?", default="")
    sub.add_argument("--programme", default=argparse.SUPPRESS)
    sub.add_argument("--week", type=int, default=argparse.SUPPRESS)
    sub.add_argument("--day", type=int, default=argparse.SUPPRESS)
    sub.add_argument("--index", required=True)
    sub.add_argument("--model-dir", required=True)
    sub.add_argument("--limit", type=int, default=5)
    for command in ("answer", "prepare-answer"):
        sub = commands.add_parser(command)
        sub.add_argument("question")
        sub.add_argument("--index", required=True)
        sub.add_argument("--model-dir", required=True)
        sub.add_argument("--limit", type=int, default=5)
        sub.add_argument("--programme", default=argparse.SUPPRESS)
        sub.add_argument("--week", type=int, default=argparse.SUPPRESS)
        sub.add_argument("--day", type=int, default=argparse.SUPPRESS)
        sub.add_argument("--base-url", required=True)
        sub.add_argument("--model", required=True, help="Exact advertised completion model ID")
        sub.add_argument("--auth-env", help="Environment variable NAME, never the secret")
        sub.add_argument("--timeout", type=float, default=60.0)
        sub.add_argument("--max-tokens", type=int, default=1024)
        sub.add_argument("--max-response-bytes", type=int, default=2_000_000)
        sub.add_argument("--max-prompt-bytes", type=int, default=120_000)
        sub.add_argument(
            "--output-dir", required=True, help="NEW private directory outside checkout"
        )
        sub.add_argument("--connection-authorized", action="store_true", required=True)
    sub = commands.add_parser("finish-answer")
    sub.add_argument("--index", required=True)
    sub.add_argument("--request-dir", required=True)
    sub.add_argument("--response", required=True)
    return parser


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "draft-coverage":
        return draft_coverage(
            args.pdf,
            args.audit,
            indexes=args.index,
            exercise_catalogs=args.exercise_catalog,
            programme_catalogs=args.programme_catalog,
            recovery_bundles=args.recovery,
            inventory=args.inventory,
        )
    if args.command == "inspect-coverage":
        return inspect_coverage(args.audit)
    if args.command == "finish-answer":
        return finish_answer(args.index, args.request_dir, args.response)
    provider = None
    if args.command in {"answer", "prepare-answer"}:
        provider = OpenAICompletion(
            args.base_url,
            args.model,
            connection_authorized=args.connection_authorized,
            auth_env=args.auth_env,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            max_response_bytes=args.max_response_bytes,
            max_prompt_bytes=args.max_prompt_bytes,
        )
    if args.command == "draft-programmes":
        return draft_programmes(
            args.pdf,
            args.catalog,
            json.loads(Path(args.layout).read_text()),
            args.start,
            args.end,
            recovery=args.recovery,
            exercise_catalogs=args.exercise_catalog,
        )
    if args.command == "inspect-programmes":
        return inspect_programmes(args.catalog)
    if args.command == "review-programme":
        return review_programme(args.catalog, args.record, args.revision, args.state)
    if args.command == "draft-exercises":
        return draft_exercises(
            args.pdf,
            args.catalog,
            json.loads(Path(args.layout).read_text()),
            args.start,
            args.end,
            recovery=args.recovery,
        )
    if args.command == "inspect-exercises":
        return inspect_exercises(args.catalog)
    if args.command == "review-exercise":
        return review_exercise(args.catalog, args.record, args.revision, args.state)
    if args.command == "inspect":
        return inspect_source(args.pdf, args.start, args.end)
    if args.command == "report":
        return report(args.index)
    if args.command in {"inspect-review", "inspectreview"}:
        return inspect_review(args.bundle)
    if args.command == "review":
        return review(args.bundle, args.region, args.revision, args.state)
    if args.command == "recover":
        from .tesseract_ocr import TesseractOCR

        engine = TesseractOCR(
            executable=args.tesseract,
            tessdata=args.tessdata_dir,
            language=args.language,
            timeout=args.timeout,
        )
        return recover(args.pdf, args.bundle, args.start, args.end, engine)
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
        return ingest(
            args.pdf,
            args.index,
            args.start,
            args.end,
            embedder,
            approval=approval,
            recovery=args.recovery,
            exercises=args.exercises,
            programmes=args.programmes,
        )
    filters = {
        key: getattr(args, key) for key in ("programme", "week", "day") if hasattr(args, key)
    }
    if provider is not None:
        operation = answer if args.command == "answer" else prepare_answer
        return operation(
            args.index,
            args.question,
            embedder,
            provider=provider,
            output_dir=args.output_dir,
            limit=args.limit,
            **filters,
        )
    return search(args.index, args.question, embedder, limit=args.limit, **filters)


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
    return 2 if result.get("status") == "error" else 0
