"""Exercise safety context and original dependency trust through public answer seams."""

import json
from pathlib import Path

import httpx
import pytest
from authored_embedder import AuthoredEmbedder
from test_exercises import accept, build, fixture

from cited_retrieval import (
    answer,
    finish_answer,
    inspect_exercises,
    prepare_answer,
    review_exercise,
)
from cited_retrieval.openai_completion import OpenAICompletion


def test_full_exercise_warnings_and_variation_are_forwarded_without_truncation(tmp_path: Path):
    source, catalog, _ = fixture(tmp_path)
    accept(catalog)
    index = build(source, catalog, tmp_path)
    result = prepare_answer(
        index,
        "Cloud press",
        AuthoredEmbedder(),
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1", "authored-model", connection_authorized=True
        ),
        output_dir=tmp_path / "bundle",
    )
    assert result["status"] == "prepared"
    packet = json.loads(Path(result["request_file"]).read_text())
    passage = json.loads(packet["messages"][1]["content"])["retrieval"]["passages"][0]
    exercise = passage["exercise"]
    assert exercise["warnings"][0]["text"] == "Stop if discomfort develops during the movement."
    assert (
        exercise["variations"][0]["instructions"][0]["text"] == "Use the wall to reduce the load."
    )
    assert exercise["warnings"][0]["location"]["pdf_page"] == 2
    assert "UNTRUSTED DATA" in packet["messages"][0]["content"]


def test_ambiguous_identity_abstains_without_inference(tmp_path: Path):
    source, catalog, _ = fixture(tmp_path)
    accept(catalog)
    index = build(source, catalog, tmp_path)

    def forbidden(request: httpx.Request) -> httpx.Response:
        pytest.fail("ambiguous identity must never contact inference")

    result = answer(
        index,
        "Compare Cloud press with Stone curl",
        AuthoredEmbedder(),
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1",
            "authored-model",
            connection_authorized=True,
            transport=httpx.MockTransport(forbidden),
        ),
        output_dir=tmp_path / "bundle",
    )
    assert result["status"] == "abstain"
    assert result["reason"] == "ambiguous"
    assert {c["id"] for c in result["disclosures"]["candidates"]} == {"cloud", "stone"}
    assert not (tmp_path / "bundle" / "inference-request.json").exists()


@pytest.mark.parametrize("wrapper", [{"error": "inference_connection"}, {"response": {}}])
def test_staged_finish_revalidates_review_dependencies_even_for_failed_response(
    tmp_path: Path, wrapper
):
    source, catalog, _ = fixture(tmp_path)
    accept(catalog)
    index = build(source, catalog, tmp_path)
    prepared = prepare_answer(
        index,
        "Cloud press",
        AuthoredEmbedder(),
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1", "authored-model", connection_authorized=True
        ),
        output_dir=tmp_path / "bundle",
    )
    record = inspect_exercises(catalog)["records"][0]
    review_exercise(catalog, record["id"], record["revision"], "rejected")
    response = tmp_path / "returned.json"
    response.write_text(json.dumps({"schema": 1, "request_id": prepared["request_id"], **wrapper}))
    result = finish_answer(index, tmp_path / "bundle", response)
    assert result["error"] == "stale_exercises"
    assert result["phase"] == "revalidation"
    assert "answer" not in result
