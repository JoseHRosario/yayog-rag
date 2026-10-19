from pathlib import Path

import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder
from input_mutating_embedder import InputMutatingEmbedder

from cited_retrieval import draft_excerpts, ingest, report, review_excerpt


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("mutation", ["write", "replace", "symlink", "directory"])
def test_ingest_preserves_concurrent_destination_owner(
    tmp_path: Path, existing: bool, mutation: str
) -> None:
    source, catalog, index = (tmp_path / n for n in ("source.pdf", "excerpts.json", "index.sqlite"))
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((30, 70), "Rest the widget for twelve seconds before continuing safely.")
        doc.save(source)
    selector = {"pdf_page": 1, "bbox": [20, 50, 500, 90]}
    result = draft_excerpts(
        source,
        catalog,
        {
            "records": [
                {
                    "id": "p",
                    "coverage": "complete",
                    "boundaries": [selector],
                    "selectors": [selector],
                }
            ]
        },
        1,
        1,
    )
    record = result["records"][0]
    review_excerpt(catalog, "p", record["revision"], "accepted")
    approval = {"sha256": result["binding"]["source_sha256"], "pages": [1]}
    if existing:
        ingest(source, index, 1, 1, AuthoredEmbedder(), approval=approval, excerpts=catalog)
    other = tmp_path / "other"
    other.write_bytes(b"OTHER OWNER")

    def mutate() -> None:
        if mutation == "write":
            index.write_bytes(b"CONCURRENT OWNER")
        else:
            index.unlink(missing_ok=True)
            if mutation == "replace":
                replacement = tmp_path / "replacement"
                replacement.write_bytes(b"CONCURRENT OWNER")
                replacement.replace(index)
            elif mutation == "symlink":
                index.symlink_to(other)
            else:
                index.mkdir()

    with pytest.raises(ValueError, match="index_changed"):
        ingest(
            source, index, 1, 1, InputMutatingEmbedder(mutate), approval=approval, excerpts=catalog
        )
    if mutation == "directory":
        assert index.is_dir()
    elif mutation == "symlink":
        assert index.is_symlink() and other.read_bytes() == b"OTHER OWNER"
    else:
        assert index.read_bytes() == b"CONCURRENT OWNER"


def test_unchanged_index_rebuild_and_failed_embedding_preserve_prior_index(tmp_path: Path) -> None:
    source, index = tmp_path / "source.pdf", tmp_path / "index.sqlite"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((30, 70), "Rest the widget for twelve seconds before continuing safely.")
        doc.save(source)
    from cited_retrieval import inspect_source

    approval = {"sha256": inspect_source(source, 1, 1)["sha256"], "pages": [1]}
    first = ingest(source, index, 1, 1, AuthoredEmbedder(), approval=approval)
    assert ingest(source, index, 1, 1, AuthoredEmbedder(), approval=approval) == first
    before = index.read_bytes()
    with pytest.raises(RuntimeError, match="authored external failure"):
        ingest(
            source, index, 1, 1, InputMutatingEmbedder(lambda: None, fail=True), approval=approval
        )
    assert index.read_bytes() == before
    assert report(index) == first


@pytest.mark.parametrize("existing", [False, True])
def test_cooperative_builders_cannot_both_publish_one_destination(
    tmp_path: Path, existing: bool
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from cited_retrieval import inspect_source

    source, index = tmp_path / "source.pdf", tmp_path / "index.sqlite"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((30, 70), "Rest the widget for twelve seconds before continuing safely.")
        doc.save(source)
    approval = {"sha256": inspect_source(source, 1, 1)["sha256"], "pages": [1]}
    if existing:
        ingest(source, index, 1, 1, AuthoredEmbedder(), approval=approval)
    barrier = Barrier(2, timeout=10)

    def synchronize() -> None:
        barrier.wait()

    def build() -> str:
        try:
            ingest(source, index, 1, 1, InputMutatingEmbedder(synchronize), approval=approval)
            return "published"
        except ValueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: build(), range(2)))
    assert sorted(results) == ["index_changed", "published"]
    assert report(index)["passage_count"] == 1
