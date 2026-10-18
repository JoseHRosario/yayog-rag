import json
from pathlib import Path

from authored_embedder import AuthoredEmbedder
from test_exercises import fixture

from cited_retrieval import local_embedder
from cited_retrieval.cli import main


def test_private_catalog_cli_draft_review_import_and_search(tmp_path, capsys, monkeypatch):
    source, catalog, layout = fixture(tmp_path)
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(layout))
    assert (
        main(
            [
                "draft-exercises",
                str(source),
                "--catalog",
                str(catalog),
                "--layout",
                str(layout_path),
                "--start",
                "1",
                "--end",
                "2",
            ]
        )
        == 0
    )
    draft = json.loads(capsys.readouterr().out)
    assert main(["inspect-exercises", str(catalog)]) == 0
    assert json.loads(capsys.readouterr().out)["records"][0]["state"] == "pending"
    for record in draft["records"]:
        assert (
            main(
                [
                    "review-exercise",
                    str(catalog),
                    "--record",
                    record["id"],
                    "--revision",
                    record["revision"],
                    "--state",
                    "accepted",
                ]
            )
            == 0
        )
        assert json.loads(capsys.readouterr().out)["records"]
    monkeypatch.setattr(local_embedder, "LocalEmbedder", lambda model_dir: AuthoredEmbedder())
    index = tmp_path / "index.sqlite"
    assert (
        main(
            [
                "import",
                str(source),
                "--start",
                "1",
                "--end",
                "2",
                "--index",
                str(index),
                "--model-dir",
                str(tmp_path / "model"),
                "--approve-sha256",
                draft["binding"]["source_sha256"],
                "--approve-pages",
                "1,2",
                "--exercises",
                str(catalog),
            ]
        )
        == 0
    )
    imported = json.loads(capsys.readouterr().out)
    assert imported["passage_count"] == 3  # Two parent windows and one variation window.
    assert len(imported["exercises"]["records"]) == 2
    assert (
        main(["search", "sky push", "--index", str(index), "--model-dir", str(tmp_path / "model")])
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["passages"][0]["exercise"]["name"] == "Cloud press"


def test_exercise_cli_has_offline_draft_inspect_and_review_commands(tmp_path: Path, capsys):
    assert (
        main(
            [
                "draft-exercises",
                str(tmp_path / "absent.pdf"),
                "--catalog",
                str(tmp_path / "catalog.json"),
                "--layout",
                str(tmp_path / "layout.json"),
                "--start",
                "1",
                "--end",
                "1",
            ]
        )
        == 2
    )
    assert "invalid_arguments" not in capsys.readouterr().out
    assert main(["inspect-exercises", str(tmp_path / "absent.json")]) == 2
    assert "stale_exercises" in capsys.readouterr().out
