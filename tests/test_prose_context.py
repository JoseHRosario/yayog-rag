"""Same-page evidence through the agreed public search/answer seams."""

import json
from pathlib import Path

import httpx
import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder
from authored_ocr import AuthoredOCR

from cited_retrieval import ingest, inspect_source, prepare_answer, search
from cited_retrieval.openai_completion import OpenAICompletion


def test_limit_one_keeps_matching_window_and_complete_eligible_page_context(tmp_path: Path) -> None:
    source = tmp_path / "prose.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 70), "Rest and sleep support recovery after training.")
        page.insert_text(
            (50, 170),
            "The six principles are patience, pacing, balance, care, variety and consistency.",
        )
        page.insert_text(
            (50, 270), "However, these principles apply only when comfortable movement is possible."
        )
        doc.new_page().insert_text((50, 70), "Food supplies energy on the following page.")
        doc.new_page().insert_text((50, 70), "This unselected page must never supply evidence.")
        doc.save(source)
    index = tmp_path / "index.sqlite"
    ingest(
        source,
        index,
        1,
        2,
        AuthoredEmbedder(),
        approval={"sha256": inspect_source(source, 1, 2)["sha256"], "pages": [1, 2]},
    )
    evidence = search(index, "rest", AuthoredEmbedder(), limit=1)
    assert len(evidence["passages"]) == 1
    match = evidence["passages"][0]
    assert match["text"] == "Rest and sleep support recovery after training."
    context = evidence["page_context"]
    assert [p["text"] for p in context] == [
        "The six principles are patience, pacing, balance, care, variety and consistency.",
        "However, these principles apply only when comfortable movement is possible.",
    ]
    assert [p["location"]["ordinal"] for p in context] == [1, 2]
    assert all(p["location"]["pdf_page"] == 1 and p["eligible"] for p in context)
    assert all(p["location"]["bbox"][1] > match["location"]["bbox"][3] for p in context)
    assert all("score" not in p for p in context)
    assert len({p["passage_id"] for p in [match, *context]}) == 3
    prepared = prepare_answer(
        index,
        "rest",
        AuthoredEmbedder(),
        limit=1,
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1", "authored-model", connection_authorized=True
        ),
        output_dir=tmp_path / "answer",
    )
    assert prepared["status"] == "prepared"
    packet = json.loads(Path(prepared["request_file"]).read_text())
    sent = json.loads(packet["messages"][1]["content"])["retrieval"]
    assert [p["citation_id"] for p in sent["passages"]] == ["E1"]
    assert [p["citation_id"] for p in sent["page_context"]] == ["E2", "E3"]
    assert [p["text"] for p in sent["page_context"]] == [p["text"] for p in context]
    from cited_retrieval import finish_answer

    response = tmp_path / "response.json"
    response.write_text(
        json.dumps(
            {
                "schema": 1,
                "request_id": prepared["request_id"],
                "response": {
                    "model": "authored-model",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {
                                        "status": "answer",
                                        "answer": "These principles have a qualification [E3].",
                                    }
                                )
                            },
                        }
                    ],
                },
            }
        )
    )
    finished = finish_answer(index, tmp_path / "answer", response)
    assert finished["status"] == "answer"
    assert finished["citations"] == ["E3"]
    retained_path = tmp_path / "answer" / "request.json"
    retained = json.loads(retained_path.read_text())
    assert retained["retrieval"]["page_context"] == sent["page_context"]
    retained["retrieval"]["page_context"][0]["text"] = "Changed context must not bind."
    retained_path.write_text(json.dumps(retained))
    assert (
        finish_answer(index, tmp_path / "answer", response)["error"] == "response_binding_mismatch"
    )


@pytest.mark.parametrize("mutate_source", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_myth_heading_keeps_refutation_and_context_citations_revalidate(
    tmp_path: Path, mutate_source: bool, grouped: bool
) -> None:
    from cited_retrieval import answer

    source = tmp_path / "myths.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 70), "Myth: rest always makes everyone stronger overnight.")
        page.insert_text(
            (50, 170), "This claim is false; adaptation requires repeated appropriate activity."
        )
        doc.save(source)
    index = tmp_path / "index.sqlite"
    ingest(
        source,
        index,
        1,
        1,
        AuthoredEmbedder(),
        approval={"sha256": inspect_source(source, 1, 1)["sha256"], "pages": [1]},
    )

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "authored-model"}]})
        sent = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert sent["retrieval"]["passages"][0]["text"].startswith("Myth:")
        refutation = sent["retrieval"]["page_context"][0]
        assert (
            refutation["text"]
            == "This claim is false; adaptation requires repeated appropriate activity."
        )
        assert refutation["citation_id"] == "E2"
        if mutate_source:
            source.write_bytes(source.read_bytes() + b"\n% changed source")
        return httpx.Response(
            200,
            json={
                "model": "authored-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "status": "answer",
                                    "answer": (
                                        "The heading is a myth; the body rejects it [E1, E2]."
                                        if grouped
                                        else "The heading is a myth [E1]; the body rejects it [E2]."
                                    ),
                                }
                            )
                        },
                    }
                ],
            },
        )

    result = answer(
        index,
        "rest",
        AuthoredEmbedder(),
        limit=1,
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1",
            "authored-model",
            connection_authorized=True,
            transport=httpx.MockTransport(respond),
        ),
        output_dir=tmp_path / "answer",
    )
    if mutate_source:
        assert result["status"] == "error"
        assert result["error"] == "stale_source"
        assert result["phase"] == "revalidation"
    else:
        assert result["status"] == "answer"
        assert result["citations"] == ["E1", "E2"]


@pytest.mark.parametrize("state", ["pending", "rejected"])
def test_page_context_never_recovers_withheld_mixed_blocks(tmp_path: Path, state: str) -> None:
    from cited_retrieval import recover, review

    source = tmp_path / "recovery.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 70), "Rest and sleep support recovery after training.")
        page.insert_text((50, 170), "The principles apply only when movement is comfortable.")
        page.insert_text((50, 270), "Withheld native instruction accompanies damaged warning.")
        font = page.insert_font(fontname="damaged", fontbuffer=pymupdf.Font("cjk").buffer)
        page.insert_text(
            (50, 283), "Withheld damaged warning must not be evidence.", fontname="damaged"
        )
        doc.xref_set_key(font, "ToUnicode", "null")
        doc.save(source)
    bundle, index = tmp_path / "review.json", tmp_path / "index.sqlite"
    recovered = recover(
        source, bundle, 1, 1, AuthoredOCR("Withheld damaged warning must not be evidence.")
    )
    region = recovered["regions"][0]
    if state != "pending":
        review(bundle, region["id"], region["revision"], state)
    ingest(
        source,
        index,
        1,
        1,
        AuthoredEmbedder(),
        recovery=bundle,
        approval={"sha256": recovered["source_sha256"], "pages": [1]},
    )
    evidence = search(index, "rest", AuthoredEmbedder(), limit=1)
    assert [p["text"] for p in evidence["page_context"]] == [
        "The principles apply only when movement is comfortable."
    ]
    assert "Withheld" not in json.dumps(evidence)
    assert evidence["coverage"]["pages"][0]["status"] == "partial"
    assert evidence["coverage"]["gaps"][0]["regions"][0]["state"] == state
    prepared = prepare_answer(
        index,
        "rest",
        AuthoredEmbedder(),
        limit=1,
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1", "authored-model", connection_authorized=True
        ),
        output_dir=tmp_path / "answer",
    )
    assert prepared["status"] == "prepared"
    assert "Withheld" not in Path(prepared["request_file"]).read_text()


def test_exercise_search_does_not_expand_to_same_page_siblings(tmp_path: Path) -> None:
    from test_exercises import build, fixture

    from cited_retrieval import inspect_exercises, review_exercise

    source, catalog, _ = fixture(tmp_path)
    cloud = next(r for r in inspect_exercises(catalog)["records"] if r["id"] == "cloud")
    review_exercise(catalog, cloud["id"], cloud["revision"], "accepted")
    index = build(source, catalog, tmp_path)
    evidence = search(index, "Cloud press", AuthoredEmbedder(), limit=1)
    assert evidence["status"] == "evidence"
    assert "page_context" not in evidence
    assert [p["exercise"]["id"] for p in evidence["passages"]] == ["cloud"]
    assert "Stone curl" not in json.dumps(evidence["passages"])


def test_programme_search_does_not_expand_to_page_prose(tmp_path: Path) -> None:
    from test_programmes import accept, build, schedule

    from cited_retrieval import draft_programmes

    source, catalog, layout, _ = schedule(tmp_path)
    draft_programmes(source, catalog, layout, 1, 2)
    accept(catalog)
    index = build(source, catalog, tmp_path)
    evidence = search(
        index, "", AuthoredEmbedder(), limit=1, programme="Cloud programme", week=1, day=1
    )
    assert evidence["status"] == "evidence"
    assert "page_context" not in evidence
    assert len(evidence["passages"]) == 1
    assert "aaaaaaaaaaaaaaaaaaaa" not in json.dumps(evidence["passages"])
