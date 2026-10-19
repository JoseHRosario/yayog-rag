"""Lossless structured evidence at public prepare/answer/finish boundaries."""

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder
from test_exercises import accept

from cited_retrieval import answer, draft_exercises, ingest, inspect_source, report, search
from cited_retrieval.openai_completion import OpenAICompletion


def dense_index(tmp_path: Path) -> Path:
    source = tmp_path / "authored.pdf"
    records: list[dict[str, Any]] = []
    with pymupdf.open() as doc:
        for i in range(13):
            page = doc.new_page()
            page.insert_text((50, 50), f"Cloud movement {i}")
            for j in range(12):
                page.insert_text(
                    (50, 90 + j * 25), f"Step {i}-{j}: move slowly; stop at discomfort."
                )
            page.insert_text((50, 430), f"Warning {i}: do not force the movement.")
            page.insert_text((50, 460), f"Reference {i}: consult the movement guidance.")
            records.append(
                {
                    "id": f"cloud-{i}",
                    "aliases": [f"sky-{i}"],
                    "boundaries": [{"pdf_page": i + 1, "bbox": [40, 20, 550, 490]}],
                    "fields": {
                        "name": [{"pdf_page": i + 1, "bbox": [40, 30, 550, 55]}],
                        "instructions": [{"pdf_page": i + 1, "bbox": [40, 65, 550, 400]}],
                        "warnings": [{"pdf_page": i + 1, "bbox": [40, 410, 550, 435]}],
                        "references": [{"pdf_page": i + 1, "bbox": [40, 440, 550, 465]}],
                    },
                }
            )
        doc.save(source)
    parent = records[0]
    parent["boundaries"] = [b for r in records for b in r["boundaries"]]
    parent["coverage"] = "complete"
    parent["variations"] = [{**r, "relation": "alternative"} for r in records[1:]]
    catalog = tmp_path / "catalog.json"
    draft_exercises(source, catalog, {"records": [parent]}, 1, 13)
    accept(catalog)
    index = tmp_path / "index.sqlite"
    ingest(
        source,
        index,
        1,
        13,
        AuthoredEmbedder(),
        exercises=catalog,
        approval={"sha256": inspect_source(source, 1, 13)["sha256"], "pages": list(range(1, 14))},
    )
    return index


def test_dense_exercise_reaches_wire_with_every_approved_field_and_bound_snapshot(
    tmp_path: Path,
) -> None:
    index = dense_index(tmp_path)
    original = search(index, "sky-7", AuthoredEmbedder())
    snapshot = report(index)
    captured: list[dict[str, Any]] = []
    wire_sizes: list[int] = []

    def transport(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "authored-model"}]})
        captured.append(json.loads(request.content))
        wire_sizes.append(len(request.content))
        return httpx.Response(
            200,
            json={
                "model": "authored-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"status": "answer", "answer": "Move slowly [E1]."}
                            )
                        },
                    }
                ],
            },
        )

    result = answer(
        index,
        "sky-7",
        AuthoredEmbedder(),
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1",
            "authored-model",
            connection_authorized=True,
            max_prompt_bytes=120000,
            transport=httpx.MockTransport(transport),
        ),
        output_dir=tmp_path / "bundle",
    )
    assert result["status"] == "answer", result
    assert len(captured) == 1
    retained = json.loads((tmp_path / "bundle" / "request.json").read_text())
    user = json.loads(captured[0]["messages"][1]["content"])
    sent = user["retrieval"]["passages"][0]
    raw = original["passages"][0]
    assert sent["text"] == raw["text"]
    assert sent["location"] == raw["location"]
    assert sent["provenance"] == raw["provenance"]
    assert retained["snapshot"] == snapshot
    assert (
        retained["snapshot_sha256"]
        == hashlib.sha256(
            json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
    )
    assert retained["retrieval"]["passages"][0]["exercise"] == raw["exercise"]
    assert 0 < wire_sizes[0] < 120000
    unprojected = {
        **user,
        "retrieval": retained["retrieval"],
        "disclosures": retained["disclosures"],
    }
    old_body = {
        **captured[0],
        "messages": [
            captured[0]["messages"][0],
            {"role": "user", "content": json.dumps(unprojected)},
        ],
    }
    assert len(json.dumps(old_body).encode()) > 120000
    for before, after in zip(
        [raw["exercise"], *raw["exercise"]["variations"]],
        [sent["exercise"], *sent["exercise"]["variations"]],
        strict=True,
    ):
        for key in ("id", "name", "aliases", "boundaries", "images"):
            assert after[key] == before[key]
        for key in ("instructions", "warnings", "references"):
            assert len(after[key]) == len(before[key])
            for original_field, field in zip(before[key], after[key], strict=True):
                assert field == {k: v for k, v in original_field.items() if k != "lines"}
        for key, value in before["citations"].items():
            if key not in ("instructions", "warnings", "references"):
                assert after["citations"][key] == [
                    {k: v for k, v in s.items() if k != "lines"} for s in value
                ]
    assert user["retrieval"]["exercise_coverage"]["record_count"] == 1


def test_structured_overflow_fails_closed_without_trimming_evidence(tmp_path: Path) -> None:
    from test_exercises import build, fixture

    from cited_retrieval import prepare_answer

    source, catalog, _ = fixture(tmp_path)
    accept(catalog)
    index = build(source, catalog, tmp_path)
    original = search(index, "Cloud press", AuthoredEmbedder())
    bundle = tmp_path / "bundle"
    result = prepare_answer(
        index,
        "Cloud press",
        AuthoredEmbedder(),
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1",
            "authored-model",
            connection_authorized=True,
            max_prompt_bytes=1000,
        ),
        output_dir=bundle,
    )
    assert result["error"] == "inference_context_limit"
    assert not (bundle / "inference-request.json").exists()
    retained = json.loads((bundle / "request.json").read_text())
    sent = json.loads(retained["messages"][1]["content"])["retrieval"]["passages"][0]
    assert sent["text"] == original["passages"][0]["text"]
    assert (
        sent["exercise"]["warnings"][0]["text"]
        == "Stop if discomfort develops during the movement."
    )
    assert retained["retrieval"]["passages"][0]["exercise"] == original["passages"][0]["exercise"]


@pytest.mark.parametrize("change", ["raw_trace", "old_projection", "current_source"])
def test_structured_finish_rejects_changed_bound_metadata_and_old_packets(
    tmp_path: Path, change: str
) -> None:
    from test_answer_integrity import response_file

    from cited_retrieval import finish_answer, prepare_answer
    from cited_retrieval.inference_client import packet_id

    index = dense_index(tmp_path)
    bundle = tmp_path / "bundle"
    prepared = prepare_answer(
        index,
        "sky-7",
        AuthoredEmbedder(),
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1", "authored-model", connection_authorized=True
        ),
        output_dir=bundle,
    )
    assert prepared["status"] == "prepared"
    retained = json.loads((bundle / "request.json").read_text())
    if change == "raw_trace":
        retained["retrieval"]["passages"][0]["exercise"]["instructions"][0]["lines"][0]["text"] = (
            "altered trace"
        )
    elif change == "old_projection":
        user = json.loads(retained["messages"][1]["content"])
        del user["structured_snapshot_sha256"]
        user["retrieval"] = retained["retrieval"]
        retained["messages"][1]["content"] = json.dumps(user)
        retained["request_id"] = packet_id(retained["provider"], retained["messages"])
    else:
        with (tmp_path / "authored.pdf").open("ab") as stream:
            stream.write(b"\n% source changed\n")
    (bundle / "request.json").write_text(json.dumps(retained))
    result = finish_answer(index, bundle, response_file(tmp_path, retained["request_id"]))
    assert result["status"] == "error"
    assert result["error"] == (
        "stale_source" if change == "current_source" else "response_binding_mismatch"
    )
    assert "answer" not in result


def test_finish_preserves_future_guidance_and_nonduplicate_citations(tmp_path: Path) -> None:
    """An authored future packet must reconstruct without dropping unknown evidence."""
    from test_answer_integrity import response_file
    from test_exercises import build, fixture

    from cited_retrieval import finish_answer, prepare_answer
    from cited_retrieval.inference_client import packet_id

    source, catalog, _ = fixture(tmp_path)
    accept(catalog)
    index = build(source, catalog, tmp_path)
    bundle = tmp_path / "bundle"
    prepared = prepare_answer(
        index,
        "Cloud press",
        AuthoredEmbedder(),
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1", "authored-model", connection_authorized=True
        ),
        output_dir=bundle,
    )
    assert prepared["status"] == "prepared"
    retained = json.loads((bundle / "request.json").read_text())
    user = json.loads(retained["messages"][1]["content"])
    for target in (retained, user):
        exercise = target["retrieval"]["passages"][0]["exercise"]
        exercise["future_guidance"] = {
            "warnings": ["Do not skip this qualification."],
            "quantity": 17,
        }
        exercise["citations"]["future_policy"] = {"warning": "Keep this unknown citation metadata."}
        # Not equal to the warnings field: this citation must NOT be deduplicated.
        exercise["citations"]["warnings"] = [
            {
                "text": "Additional cited qualification.",
                "source_id": "authored",
                "location": {"pdf_page": 2, "printed_label": "Q"},
            }
        ]
    unknown = {
        "id": "unavailable",
        "state": "rejected",
        "eligible": False,
        "gaps": ["missing warning"],
        "future_warning": "Review pending.",
    }
    for target in (retained["retrieval"], retained["disclosures"]):
        target["exercise_coverage"].append(unknown)
        target["exercise_coverage"][0]["future_warning"] = "Accepted but qualified."
    for target in (user["retrieval"], user["disclosures"]):
        target["exercise_coverage"] = {
            "record_count": 3,
            "accepted_complete_count": 1,
            "unavailable_or_nonroutine": [retained["retrieval"]["exercise_coverage"][0], unknown],
        }
    user["structured_snapshot_sha256"] = hashlib.sha256(
        json.dumps(
            {"retrieval": retained["retrieval"], "disclosures": retained["disclosures"]},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    retained["messages"][1]["content"] = json.dumps(user)
    retained["request_id"] = packet_id(retained["provider"], retained["messages"])
    (bundle / "request.json").write_text(json.dumps(retained))
    result = finish_answer(index, bundle, response_file(tmp_path, retained["request_id"]))
    assert result["status"] == "answer", result


def test_nested_programme_record_compacts_without_changing_session_evidence(tmp_path: Path) -> None:
    from test_programmes import accept as accept_programmes
    from test_programmes import schedule

    from cited_retrieval import draft_programmes, prepare_answer

    support = tmp_path / "support"
    support.mkdir()
    dense_index(support)
    source, catalog, layout, sel = schedule(tmp_path, exercise_name="Cloud movement 0")
    layout["records"][0]["conflicts"] = [{"claims": [[sel(1, 40, 80, 220)], [sel(2, 40, 50)]]}]
    layout["records"].append(
        {
            "id": "alternating",
            "kind": "method",
            "coverage": "complete",
            "aliases": ["Alternate movements"],
            "boundaries": [{"pdf_page": 2, "bbox": [30, 260, 550, 330]}],
            "fields": {"name": [sel(2, 40, 280)], "definition": [sel(2, 40, 310, 450)]},
        }
    )
    draft_programmes(source, catalog, layout, 1, 2, exercise_catalogs=[support / "catalog.json"])
    accept_programmes(catalog)
    index = tmp_path / "sessions.sqlite"
    ingest(
        source,
        index,
        1,
        2,
        AuthoredEmbedder(),
        programmes=catalog,
        approval={"sha256": inspect_source(source, 1, 2)["sha256"], "pages": [1, 2]},
    )
    bundle = tmp_path / "bundle"
    prepared = prepare_answer(
        index,
        "",
        AuthoredEmbedder(),
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1", "authored-model", connection_authorized=True
        ),
        output_dir=bundle,
        programme="Cloud programme",
        week=1,
        day=1,
    )
    assert prepared["status"] == "prepared", prepared
    retained = json.loads((bundle / "request.json").read_text())
    raw = retained["retrieval"]["passages"][0]
    sent = json.loads(retained["messages"][1]["content"])["retrieval"]["passages"][0]
    assert sent["text"] == raw["text"]
    for key, value in raw["session"].items():
        if key != "entries":
            assert sent["session"][key] == value
    for original, projected in zip(
        raw["session"]["entries"], sent["session"]["entries"], strict=True
    ):
        assert {k: v for k, v in projected.items() if k != "exercise_reference"} == {
            k: v for k, v in original.items() if k != "exercise_reference"
        }
        ref = projected["exercise_reference"]
        if ref["status"] == "resolved":
            assert (
                ref["record"]["instructions"][0]["text"]
                == original["exercise_reference"]["record"]["instructions"][0]["text"]
            )
            assert "lines" not in ref["record"]["instructions"][0]
            assert ref["record"]["warnings"][0]["text"] == "Warning 0: do not force the movement."
        else:
            assert ref == original["exercise_reference"]
