"""The portable client's public file/command interface; HTTP I/O is the only double."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from authored_embedder import AuthoredEmbedder
from test_answers import make_index

from cited_retrieval import finish_answer, prepare_answer
from cited_retrieval.inference_client import main
from cited_retrieval.openai_completion import OpenAICompletion


def test_standalone_script_imports_only_stdlib_even_without_site_packages(tmp_path: Path):
    from cited_retrieval import inference_client

    copied = tmp_path / "inference_client.py"
    copied.write_bytes(Path(inference_client.__file__).read_bytes())
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(copied), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--connection-authorized" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize("failure", [None, "timeout", "auth", "redirect", "model"])
def test_windows_file_exchange_with_real_preparation_and_finish(
    tmp_path: Path, monkeypatch, capsys, failure
):
    import http.client

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
    requests = []

    def connect(host, port, timeout):
        assert (host, port, timeout) == ("127.0.0.1", 8899, 60.0)
        connection = MagicMock()

        def request(method, path, body, headers):
            requests.append((method, path, body, headers))
            if failure == "timeout":
                raise TimeoutError("never print private error text")

        connection.request.side_effect = request
        response = MagicMock()
        response.status = 401 if failure == "auth" else 307 if failure == "redirect" else 200
        if not requests:
            payload = {"data": [{"id": "different" if failure == "model" else "authored-model"}]}
        else:
            payload = {
                "model": "authored-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"status": "answer", "answer": "Rest and sleep [E1]."}
                            )
                        },
                    }
                ],
            }
        response.read.return_value = json.dumps(payload).encode()
        connection.getresponse.return_value = response
        return connection

    monkeypatch.setattr(http.client, "HTTPConnection", connect)
    returned = tmp_path / "returned.json"
    code = main([prepared["request_file"], "--response", str(returned), "--connection-authorized"])
    assert code == (2 if failure else 0)
    assert "never print" not in capsys.readouterr().out
    result = finish_answer(index, tmp_path / "bundle", returned)
    if failure:
        codes = {
            "timeout": "inference_timeout",
            "auth": "inference_auth",
            "redirect": "inference_http_status",
            "model": "inference_model_mismatch",
        }
        assert result["error"] == codes[failure]
        assert "answer" not in result
        assert len(requests) == 1
    else:
        assert result["status"] == "answer"
        assert [r[:2] for r in requests] == [
            ("GET", "/v1/models"),
            ("POST", "/v1/chat/completions"),
        ]
        body = json.loads(requests[1][2])
        assert "Rest and sleep support recovery" in body["messages"][1]["content"]
    before = returned.read_bytes()
    assert (
        main([prepared["request_file"], "--response", str(returned), "--connection-authorized"])
        == 2
    )
    assert returned.read_bytes() == before
