import json
from pathlib import Path

from authored_tessdata import authored_tessdata
from test_cli import run
from test_recovery import damaged_source


def test_recover_inspect_and_explicit_review_are_runnable_commands(tmp_path: Path):
    source = damaged_source(tmp_path)
    bundle = tmp_path / "bundle.json"
    executable = tmp_path / "ocr"
    executable.write_text(
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then printf 'authored engine 1\\n'; else printf 'Hold for 12 seconds. Stop if pain develops.\\n'; fi\n"
    )
    executable.chmod(0o700)
    result = run(
        "recover",
        str(source),
        "--start",
        "1",
        "--end",
        "1",
        "--bundle",
        str(bundle),
        "--tesseract",
        str(executable),
        "--tessdata-dir",
        str(authored_tessdata(tmp_path)),
    )
    assert result.returncode == 0, result.stdout
    region = json.loads(result.stdout)["regions"][0]
    inspected = run("inspect-review", str(bundle))
    assert inspected.returncode == 0
    assert json.loads(inspected.stdout)["regions"][0]["state"] == "pending"
    reviewed = run(
        "review",
        str(bundle),
        "--region",
        region["id"],
        "--revision",
        region["revision"],
        "--state",
        "accepted",
    )
    assert reviewed.returncode == 0
    assert (
        json.loads(run("inspect-review", str(bundle)).stdout)["regions"][0]["state"] == "accepted"
    )
    bad = run(
        "review",
        str(bundle),
        "--region",
        region["id"],
        "--revision",
        "private-wrong-revision",
        "--state",
        "accepted",
    )
    assert bad.returncode == 2
    assert json.loads(bad.stdout) == {"status": "error", "error": "review_revision_mismatch"}
    assert bad.stderr == ""
