import json

from authored_embedder import AuthoredEmbedder
from test_programmes import schedule

from cited_retrieval.cli import main


def invoke(capsys, *args):
    code = main(list(args))
    return code, json.loads(capsys.readouterr().out)


def test_programme_cli_draft_review_import_and_filtered_search(tmp_path, capsys, monkeypatch):
    from cited_retrieval import inspect_source
    from cited_retrieval.local_embedder import LocalEmbedder

    monkeypatch.setattr(LocalEmbedder, "__new__", lambda *args, **kwargs: AuthoredEmbedder())
    source, catalog, layout, _ = schedule(tmp_path)
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(layout))
    code, result = invoke(
        capsys,
        "draft-programmes",
        str(source),
        "--catalog",
        str(catalog),
        "--layout",
        str(layout_path),
        "--start",
        "1",
        "--end",
        "2",
    )
    assert code == 0
    revision = result["records"][0]["revision"]
    code, _ = invoke(
        capsys,
        "review-programme",
        str(catalog),
        "--record",
        "cloud-1",
        "--revision",
        revision,
        "--state",
        "accepted",
    )
    assert code == 0
    index = tmp_path / "index.sqlite"
    code, _ = invoke(
        capsys,
        "import",
        str(source),
        "--index",
        str(index),
        "--start",
        "1",
        "--end",
        "2",
        "--approve-pages",
        "1,2",
        "--approve-sha256",
        inspect_source(source, 1, 2)["sha256"],
        "--model-dir",
        "unused",
        "--programmes",
        str(catalog),
    )
    assert code == 0
    code, result = invoke(
        capsys,
        "search",
        "--index",
        str(index),
        "--model-dir",
        "unused",
        "--programme",
        "Cloud programme",
        "--week",
        "2",
        "--day",
        "1",
    )
    assert code == 0
    assert result["passages"][0]["session"]["id"] == "cloud-1"
    code, result = invoke(
        capsys, "search", "Cloud", "--index", str(index), "--model-dir", "unused", "--week", "1"
    )
    assert code == 2
    assert result["error"] == "invalid_session_filter"
