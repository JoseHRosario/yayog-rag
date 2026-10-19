"""Public answering regressions for private destinations and exact snapshots."""

import json
import shutil

import pytest
from authored_embedder import AuthoredEmbedder
from test_answers import make_index

from cited_retrieval import finish_answer, ingest, inspect_source, prepare_answer, report
from cited_retrieval.openai_completion import OpenAICompletion


def provider():
    return OpenAICompletion(
        "http://127.0.0.1:8899/v1", "authored-model", connection_authorized=True
    )


def test_prompt_summarizes_unselected_pages_without_dropping_approved_evidence(tmp_path):
    import pymupdf

    source, index = make_index(tmp_path)
    with pymupdf.open(source) as doc:
        for _ in range(2):
            doc.new_page().insert_text((50, 100), "Unselected reference section.")
        data = doc.tobytes()
    source.write_bytes(data)
    ingest(
        source,
        index,
        1,
        1,
        AuthoredEmbedder(),
        approval={"sha256": inspect_source(source, 1, 1)["sha256"], "pages": [1]},
    )
    bundle = tmp_path / "private-bundle"
    prepared = prepare_answer(
        index, "recovery", AuthoredEmbedder(), provider=provider(), output_dir=bundle
    )
    packet = json.loads((bundle / "inference-request.json").read_text())
    user = json.loads(packet["messages"][1]["content"])
    assert prepared["status"] == "prepared"
    assert (
        user["retrieval"]["passages"][0]["text"]
        == "Rest and sleep support recovery after training."
    )
    assert len(user["disclosures"]["source_pages"]) == 1
    assert user["disclosures"]["not_selected_page_count"] == 2
    retained = json.loads((bundle / "request.json").read_text())
    assert len(retained["snapshot"]["pages"]) == 3
    assert len(retained["disclosures"]["source_pages"]) == 3
    assert user["snapshot_sha256"] == retained["snapshot_sha256"]


def response_file(tmp_path, request_id):
    response = tmp_path / "response.json"
    response.write_text(
        json.dumps(
            {
                "schema": 1,
                "request_id": request_id,
                "response": {
                    "model": "authored-model",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {
                                        "status": "answer",
                                        "answer": "The source recommends rest [E1].",
                                    }
                                )
                            },
                        }
                    ],
                },
            }
        )
    )
    return response


@pytest.mark.parametrize("linked_worktree", [False, True])
def test_prepare_rejects_other_checkout_destinations(tmp_path, linked_worktree):
    _, index = make_index(tmp_path)
    checkout = tmp_path / "other-checkout"
    checkout.mkdir()
    if linked_worktree:
        (checkout / ".git").write_text("gitdir: ../main/.git/worktrees/other")
    else:
        (checkout / ".git").mkdir()
    destination = checkout / "answer"
    with pytest.raises(ValueError, match="unsafe_answer_path"):
        prepare_answer(
            index, "recovery", AuthoredEmbedder(), provider=provider(), output_dir=destination
        )
    assert not destination.exists()


def test_finish_rejects_a_bundle_relocated_into_a_checkout(tmp_path):
    _, index = make_index(tmp_path)
    original = tmp_path / "private-bundle"
    prepared = prepare_answer(
        index, "recovery", AuthoredEmbedder(), provider=provider(), output_dir=original
    )
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    moved = checkout / "moved-bundle"
    shutil.copytree(original, moved)
    before = sorted(p.name for p in moved.iterdir())
    response = response_file(tmp_path, prepared["request_id"])
    with pytest.raises(ValueError, match="unsafe_answer_path"):
        finish_answer(index, moved, response)
    assert sorted(p.name for p in moved.iterdir()) == before


def test_replacing_only_snapshot_cannot_rebind_an_old_response(tmp_path):
    source, index = make_index(tmp_path)
    bundle = tmp_path / "private-bundle"
    prepared = prepare_answer(
        index, "recovery", AuthoredEmbedder(), provider=provider(), output_dir=bundle
    )
    retained = json.loads((bundle / "request.json").read_text())
    before = retained["snapshot"]["corpus_sha256"]
    ingest(
        source,
        index,
        1,
        1,
        AuthoredEmbedder(fixed_vector=[0.5, 0.5]),
        approval={"sha256": inspect_source(source, 1, 1)["sha256"], "pages": [1]},
    )
    retained["snapshot"] = report(index)
    assert retained["snapshot"]["corpus_sha256"] != before
    (bundle / "request.json").write_text(json.dumps(retained))
    result = finish_answer(index, bundle, response_file(tmp_path, prepared["request_id"]))
    assert result["status"] == "error"
    assert result["error"] == "response_binding_mismatch"
    assert "answer" not in result
