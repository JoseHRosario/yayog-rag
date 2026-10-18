import json
import subprocess
import sys
from pathlib import Path


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cited_retrieval", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_errors_are_json_without_echoing_paths_or_inputs(tmp_path: Path):
    sensitive = str(tmp_path / "secret-source.pdf")
    result = run("inspect", sensitive, "--start", "1", "--end", "2")
    assert result.returncode == 2
    assert json.loads(result.stdout) == {"status": "error", "error": "operation_failed"}
    assert result.stderr == ""
    assert sensitive not in result.stdout
    result = run("--secret-unrecognised-value")
    assert result.returncode == 2
    assert json.loads(result.stdout)["error"] == "invalid_arguments"
    assert result.stderr == ""


def test_cli_help_documents_public_commands():
    result = run("--help")
    assert result.returncode == 0
    for command in ("inspect", "import", "report", "search"):
        assert command in result.stdout
