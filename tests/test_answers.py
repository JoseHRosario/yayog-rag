"""Authored source → public answer seam → ordinary external HTTP transport."""

import json
from pathlib import Path

import httpx
import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder

from cited_retrieval import ingest, inspect_source


def make_index(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "authored.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 100), "Rest and sleep support recovery after training.")
    doc.save(source)
    doc.close()
    index = tmp_path / "index.sqlite"
    ingest(
        source,
        index,
        1,
        1,
        AuthoredEmbedder(),
        approval={"sha256": inspect_source(source, 1, 1)["sha256"], "pages": [1]},
    )
    return source, index


def test_answer_forwards_actual_evidence_and_retains_private_bundle(tmp_path: Path):
    from cited_retrieval import answer
    from cited_retrieval.openai_completion import OpenAICompletion

    _, index = make_index(tmp_path)
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "authored-model"}]})
        prompt = json.loads(request.content)
        assert prompt["model"] == "authored-model"
        assert prompt["stream"] is False
        assert "UNTRUSTED DATA" in prompt["messages"][0]["content"]
        evidence = json.loads(prompt["messages"][1]["content"])
        assert evidence["question"] == "recovery"
        passage = evidence["retrieval"]["passages"][0]
        assert passage["text"] == "Rest and sleep support recovery after training."
        assert passage["citation_id"] == "E1"
        assert passage["source_id"] == inspect_source(tmp_path / "authored.pdf", 1, 1)["sha256"]
        assert passage["location"]["pdf_page"] == 1
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
                                    "answer": "The source recommends rest and sleep [E1].",
                                }
                            )
                        },
                    }
                ],
            },
        )

    provider = OpenAICompletion(
        "http://127.0.0.1:8899/v1",
        "authored-model",
        connection_authorized=True,
        transport=httpx.MockTransport(respond),
    )
    result = answer(
        index,
        "recovery",
        AuthoredEmbedder(),
        provider=provider,
        output_dir=tmp_path / "private-answer",
    )
    assert result["status"] == "answer"
    assert result["citations"] == ["E1"]
    assert result["claim_support_verified"] is False
    assert len(requests) == 2
    bundle = Path(result["bundle"])
    assert json.loads((bundle / "request.json").read_text())["retrieval"]["passages"][0][
        "passage_id"
    ]
    assert (
        json.loads((bundle / "response.json").read_text())["response"]["model"] == "authored-model"
    )
    assert (bundle.stat().st_mode & 0o777) == 0o700
    assert all((p.stat().st_mode & 0o777) == 0o600 for p in bundle.iterdir())


def test_staged_packet_has_no_linux_dependencies_and_binds_response(tmp_path: Path):
    from cited_retrieval import finish_answer, prepare_answer
    from cited_retrieval.openai_completion import OpenAICompletion

    _, index = make_index(tmp_path)
    prepared = prepare_answer(
        index,
        "recovery",
        AuthoredEmbedder(),
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1", "authored-model", connection_authorized=True
        ),
        output_dir=tmp_path / "bundle",
    )
    assert prepared["status"] == "prepared"
    packet = json.loads(Path(prepared["request_file"]).read_text())
    assert set(packet) == {"schema", "request_id", "provider", "messages"}
    assert str(index) not in json.dumps(packet)
    response = tmp_path / "returned.json"
    response.write_text(json.dumps({"schema": 1, "request_id": "wrong-request", "response": {}}))
    bad = finish_answer(index, tmp_path / "bundle", response)
    assert bad["error"] == "response_binding_mismatch"
    response.write_text(
        json.dumps(
            {
                "schema": 1,
                "request_id": packet["request_id"],
                "response": {
                    "model": "authored-model",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps({"status": "answer", "answer": "Rest [E1]."})
                            },
                        }
                    ],
                },
            }
        )
    )
    result = finish_answer(index, tmp_path / "bundle", response)
    assert result["status"] == "answer"
    assert result["claim_support_verified"] is False


@pytest.mark.parametrize(
    "text",
    [
        "Rest [E2].",
        "Rest.",
        "Rest [e1].",
        "Rest [E0].",
        "Rest [E1] [999].",
        "Rest (bbox [1, 2, 3, 4]) [E99].",
        "Rest (bbox [1, 2, E99, 4]) [E1].",
        "Rest (bbox [1, 2, 3]) [E1].",
        "Rest (bbox [1, 2, 3, 4]) [e1].",
        "Rest (bbox [1, 2, 3, 4]) [E1] [999].",
    ],
)
def test_answer_rejects_unsupplied_or_missing_citations(tmp_path: Path, text: str):
    from cited_retrieval import answer
    from cited_retrieval.openai_completion import OpenAICompletion

    _, index = make_index(tmp_path)

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "authored-model"}]})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({"status": "answer", "answer": text})},
                    }
                ]
            },
        )

    result = answer(
        index,
        "recovery",
        AuthoredEmbedder(),
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1",
            "authored-model",
            connection_authorized=True,
            transport=httpx.MockTransport(respond),
        ),
        output_dir=tmp_path / "bundle",
    )
    assert result["error"] == "invalid_citations"
    assert result["phase"] == "validation"
    assert "answer" not in result
    assert (Path(result["bundle"]) / "response.json").exists()


@pytest.mark.parametrize(
    "coordinates",
    ["bbox [1, 2, 3, 4]", "bbox: [-1.5, 2.25, 30.0, 40.0]"],
)
def test_answer_accepts_coordinate_data_beside_valid_citations(tmp_path: Path, coordinates: str):
    from cited_retrieval import answer
    from cited_retrieval.openai_completion import OpenAICompletion

    _, index = make_index(tmp_path)
    text = f"Rest and sleep support recovery ({coordinates}) [E1]."

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "authored-model"}]})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({"status": "answer", "answer": text})},
                    }
                ]
            },
        )

    result = answer(
        index,
        "recovery",
        AuthoredEmbedder(),
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1",
            "authored-model",
            connection_authorized=True,
            transport=httpx.MockTransport(respond),
        ),
        output_dir=tmp_path / "bundle",
    )
    assert result["status"] == "answer"
    assert result["answer"] == text
    assert result["citations"] == ["E1"]
    # Coordinate prose is not a citation, nor does accepting it verify its factual accuracy.
    assert result["claim_support_verified"] is False


@pytest.mark.parametrize(
    "base",
    [
        "http://user:password@127.0.0.1:8899/v1",
        "http://@127.0.0.1:8899/v1",
        "http://:@127.0.0.1:8899/v1",
        "http://127.0.0.1:8899/v1?",
        "http://127.0.0.1:8899/v1#",
    ],
)
def test_endpoint_rejects_secret_bearing_or_query_fragment_urls(base: str):
    from cited_retrieval.openai_completion import OpenAICompletion

    with pytest.raises(ValueError, match="invalid_inference_config"):
        OpenAICompletion(base, "authored-model", connection_authorized=True)


def test_cli_advertises_staged_and_integrated_commands():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "cited_retrieval", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    for command in ("answer", "prepare-answer", "finish-answer"):
        assert command in result.stdout


def test_optional_auth_is_forwarded_but_never_retained(tmp_path: Path, monkeypatch):
    from cited_retrieval import answer
    from cited_retrieval.openai_completion import OpenAICompletion

    _, index = make_index(tmp_path)
    monkeypatch.setenv("AUTHORED_INFERENCE_KEY", "authored-secret-never-retain")

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer authored-secret-never-retain"
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "authored-model"}]})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "status": "abstain",
                                    "answer": "Not enough evidence for this claim.",
                                }
                            )
                        },
                    }
                ]
            },
        )

    bundle = tmp_path / "bundle"
    result = answer(
        index,
        "recovery",
        AuthoredEmbedder(),
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1",
            "authored-model",
            connection_authorized=True,
            auth_env="AUTHORED_INFERENCE_KEY",
            transport=httpx.MockTransport(respond),
        ),
        output_dir=bundle,
    )
    assert result["status"] == "abstain"
    assert all("authored-secret-never-retain" not in p.read_text() for p in bundle.iterdir())
    assert "authored-secret-never-retain" not in json.dumps(result)


@pytest.mark.parametrize(
    "config",
    [
        {"timeout": 0},
        {"timeout": float("inf")},
        {"timeout": float("nan")},
        {"max_tokens": 0},
        {"max_response_bytes": -1},
        {"max_prompt_bytes": 0},
        {"auth_env": "not a name"},
    ],
)
def test_invalid_inference_limits_fail_before_retrieval_or_network(config):
    from cited_retrieval.openai_completion import OpenAICompletion

    with pytest.raises(ValueError, match="invalid_inference_config"):
        OpenAICompletion(
            "http://127.0.0.1:8899/v1", "authored-model", connection_authorized=True, **config
        )
    with pytest.raises(ValueError, match="connection_not_authorized"):
        OpenAICompletion("http://127.0.0.1:8899/v1", "authored-model")


def test_tampered_local_citation_map_cannot_validate_ids_not_in_sent_prompt(tmp_path: Path):
    from cited_retrieval import finish_answer, prepare_answer
    from cited_retrieval.openai_completion import OpenAICompletion

    _, index = make_index(tmp_path)
    prepared = prepare_answer(
        index,
        "recovery",
        AuthoredEmbedder(),
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1", "authored-model", connection_authorized=True
        ),
        output_dir=tmp_path / "bundle",
    )
    local_file = tmp_path / "bundle" / "request.json"
    local = json.loads(local_file.read_text())
    local["retrieval"]["passages"][0]["citation_id"] = "E2"
    local_file.write_text(json.dumps(local))
    response = tmp_path / "returned.json"
    response.write_text(
        json.dumps(
            {
                "schema": 1,
                "request_id": prepared["request_id"],
                "response": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps({"status": "answer", "answer": "Rest [E2]."})
                            },
                        }
                    ]
                },
            }
        )
    )
    result = finish_answer(index, tmp_path / "bundle", response)
    assert result["error"] == "response_binding_mismatch"
    assert "answer" not in result


def test_prepare_rejects_context_overflow_before_export(tmp_path: Path):
    from cited_retrieval import prepare_answer
    from cited_retrieval.openai_completion import OpenAICompletion

    _, index = make_index(tmp_path)
    result = prepare_answer(
        index,
        "recovery",
        AuthoredEmbedder(),
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1",
            "authored-model",
            connection_authorized=True,
            max_prompt_bytes=1,
        ),
        output_dir=tmp_path / "bundle",
    )
    assert result["error"] == "inference_context_limit"
    assert not (tmp_path / "bundle" / "inference-request.json").exists()


@pytest.mark.parametrize(
    "failure",
    [
        "timeout",
        "connection",
        "auth",
        "redirect",
        "models",
        "response-model",
        "protocol",
        "response-limit",
    ],
)
def test_inference_failures_are_distinct_private_and_never_answers(tmp_path: Path, failure: str):
    from cited_retrieval import answer
    from cited_retrieval.openai_completion import OpenAICompletion

    _, index = make_index(tmp_path)
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("private evidence must not escape", request=request)
        if failure == "connection":
            raise httpx.ConnectError("private evidence must not escape", request=request)
        if failure == "auth":
            return httpx.Response(401)
        if failure == "redirect":
            return httpx.Response(307, headers={"location": "http://never-connect.invalid"})
        if request.method == "GET":
            return httpx.Response(
                200, json={"data": [{"id": "wrong" if failure == "models" else "authored-model"}]}
            )
        if failure == "response-limit":
            return httpx.Response(200, content=b"x" * 1001)
        if failure == "protocol":
            return httpx.Response(200, content=b"not json")
        return httpx.Response(200, json={"model": "different-model", "choices": []})

    result = answer(
        index,
        "recovery",
        AuthoredEmbedder(),
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1",
            "authored-model",
            connection_authorized=True,
            max_response_bytes=1000,
            transport=httpx.MockTransport(respond),
        ),
        output_dir=tmp_path / "bundle",
    )
    codes = {
        "timeout": "inference_timeout",
        "connection": "inference_connection",
        "auth": "inference_auth",
        "redirect": "inference_http_status",
        "models": "inference_model_mismatch",
        "response-model": "inference_model_mismatch",
        "protocol": "inference_protocol",
        "response-limit": "inference_response_limit",
    }
    assert result["error"] == codes[failure]
    assert "answer" not in result
    assert "private evidence" not in json.dumps(result)
    assert len(calls) == (2 if failure in {"response-model", "protocol", "response-limit"} else 1)


@pytest.mark.parametrize("network_failure", [True, False])
def test_original_source_revalidated_even_after_connection_failure(
    tmp_path: Path, network_failure: bool
):
    from cited_retrieval import answer
    from cited_retrieval.openai_completion import OpenAICompletion

    source, index = make_index(tmp_path)

    def respond(request: httpx.Request) -> httpx.Response:
        source.write_bytes(source.read_bytes() + b"\n% changed during inference")
        if network_failure:
            raise httpx.ConnectError("not retained", request=request)
        return httpx.Response(200, json={"data": []})

    result = answer(
        index,
        "recovery",
        AuthoredEmbedder(),
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1",
            "authored-model",
            connection_authorized=True,
            transport=httpx.MockTransport(respond),
        ),
        output_dir=tmp_path / "bundle",
    )
    assert result["phase"] == "revalidation"
    assert result["error"] == "stale_source"
    assert "answer" not in result


def test_no_evidence_never_calls_inference_and_collision_never_overwrites(tmp_path: Path):
    from cited_retrieval import answer
    from cited_retrieval.openai_completion import OpenAICompletion

    source, index = make_index(tmp_path)

    def forbidden(request: httpx.Request) -> httpx.Response:
        pytest.fail("no evidence must never contact inference")

    provider = OpenAICompletion(
        "http://127.0.0.1:8899/v1",
        "authored-model",
        connection_authorized=True,
        transport=httpx.MockTransport(forbidden),
    )
    result = answer(
        index,
        "orbital mechanics",
        AuthoredEmbedder(fixed_vector=[-1.0, -1.0]),
        provider=provider,
        output_dir=tmp_path / "bundle",
    )
    assert result["status"] == "abstain"
    assert result["reason"] == "no_evidence"
    for collision in (source, index, tmp_path / "bundle"):
        with pytest.raises(ValueError, match="unsafe_answer_path"):
            answer(index, "recovery", AuthoredEmbedder(), provider=provider, output_dir=collision)


def test_conflicts_and_unresolved_references_cannot_disappear_in_model_answer(tmp_path: Path):
    from test_programmes import accept, build, schedule

    from cited_retrieval import answer, draft_programmes
    from cited_retrieval.openai_completion import OpenAICompletion

    source, catalog, layout, sel = schedule(tmp_path)
    layout["records"][0]["conflicts"] = [{"claims": [[sel(1, 40, 80, 220)], [sel(2, 40, 50)]]}]
    draft_programmes(source, catalog, layout, 1, 2)
    accept(catalog)
    index = build(source, catalog, tmp_path)

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "authored-model"}]})
        user = json.loads(json.loads(request.content)["messages"][1]["content"])
        claims = user["disclosures"]["conflicts"][0]["claims"]
        assert [c[0]["text"] for c in claims] == ["Summary weeks 1-4", "Weeks 1-2"]
        assert [c[0]["location"]["pdf_page"] for c in claims] == [1, 2]
        assert user["disclosures"]["unresolved_references"]
        assert user["disclosures"]["source_pages"]
        session = user["retrieval"]["passages"][0]["session"]
        assert session["entries"][0]["assistance"] == "Wall assistance"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"status": "answer", "answer": "The source lists a session [E1]."}
                            )
                        },
                    }
                ]
            },
        )

    result = answer(
        index,
        "Cloud programme",
        AuthoredEmbedder(),
        provider=OpenAICompletion(
            "http://127.0.0.1:8899/v1",
            "authored-model",
            connection_authorized=True,
            transport=httpx.MockTransport(respond),
        ),
        output_dir=tmp_path / "bundle",
        programme="Cloud programme",
        week=1,
        day=1,
    )
    assert result["status"] == "answer"
    assert result["disclosures"]["conflicts"]
    assert result["disclosures"]["unresolved_references"]


def test_cli_prepare_and_finish_execute_without_windows_embedding_install(
    tmp_path: Path, monkeypatch, capsys
):
    from cited_retrieval import local_embedder
    from cited_retrieval.cli import main

    _, index = make_index(tmp_path)
    monkeypatch.setattr(local_embedder, "LocalEmbedder", lambda directory: AuthoredEmbedder())
    bundle = tmp_path / "bundle"
    assert (
        main(
            [
                "prepare-answer",
                "recovery",
                "--index",
                str(index),
                "--model-dir",
                "unused",
                "--base-url",
                "http://127.0.0.1:8899/v1",
                "--model",
                "authored-model",
                "--output-dir",
                str(bundle),
                "--connection-authorized",
            ]
        )
        == 0
    )
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["status"] == "prepared"
    packet = json.loads(Path(prepared["request_file"]).read_text())
    response = tmp_path / "returned.json"
    response.write_text(
        json.dumps(
            {
                "schema": 1,
                "request_id": packet["request_id"],
                "response": {
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
                    ]
                },
            }
        )
    )

    def forbidden(directory):
        pytest.fail("finish must not initialize embedding inference")

    monkeypatch.setattr(local_embedder, "LocalEmbedder", forbidden)
    assert (
        main(
            [
                "finish-answer",
                "--index",
                str(index),
                "--request-dir",
                str(bundle),
                "--response",
                str(response),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "answer"
    response.write_text("not json")
    assert (
        main(
            [
                "finish-answer",
                "--index",
                str(index),
                "--request-dir",
                str(bundle),
                "--response",
                str(response),
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["error"] == "inference_protocol"
