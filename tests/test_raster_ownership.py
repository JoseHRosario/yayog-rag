from pathlib import Path

import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder
from authored_ocr import AuthoredOCR

from cited_retrieval import draft_excerpts, ingest, recover, review_excerpt
from cited_retrieval.exercises import draft_exercises


@pytest.mark.parametrize("kind", ["excerpt", "exercise"])
@pytest.mark.parametrize("warning", [True, False])
@pytest.mark.parametrize("inline", [True, False])
def test_native_rectangle_does_not_own_raster_warning(
    tmp_path: Path, kind: str, warning: bool, inline: bool
) -> None:
    source = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=300)
        page.insert_text(
            (20, 70), "Rest the widget                              for twelve seconds."
        )
        page.insert_text((20, 88), "Continue only if the lamp stays green.")
        if warning:
            with pymupdf.open() as raster:
                label = raster.new_page(width=180, height=8)
                label.insert_text((0, 6), "STOP IF PAIN DEVELOPS", fontsize=6)
                pixels = label.get_pixmap(matrix=pymupdf.Matrix(3, 3))
                image = pixels.tobytes("png")
            if inline:
                native_bytes = b"\n".join(doc.xref_stream(x) for x in page.get_contents())
                content = doc.get_new_xref()
                doc.update_object(content, "<<>>")
                header = f"q 80 0 0 12 100 229 cm BI /W {pixels.width} /H {pixels.height} /CS /RGB /BPC 8 /F /AHx ID\n"
                doc.update_stream(
                    content,
                    native_bytes + header.encode() + pixels.samples.hex().encode() + b">\nEI Q",
                )
                page.set_contents(content)
            else:
                page.insert_image(
                    pymupdf.Rect(100, 59, 180, 71), stream=image, keep_proportion=False
                )
        doc.save(source)
    bundle = tmp_path / "bundle.json"
    recovered = recover(
        source,
        bundle,
        1,
        1,
        AuthoredOCR(),
        image_regions=[{"id": "other", "pdf_page": 1, "bbox": [20, 200, 200, 230]}],
    )
    native = next(b for b in recovered["blocks"] if not b.get("image_region_id"))
    boundary = {"pdf_page": 1, "bbox": native["bbox"]}
    with pymupdf.open(source) as doc:
        assert "STOP" not in doc[0].get_text()
        if warning:
            assert pymupdf.Rect(native["bbox"]).contains(doc[0].get_image_info()[0]["bbox"])
            if inline:
                assert doc[0].get_image_info(xrefs=True)[0]["xref"] == 0
    spec = {"id": "widget", "coverage": "complete", "boundaries": [boundary]}
    catalog = tmp_path / "catalog.json"
    if kind == "excerpt":
        spec["selectors"] = [boundary]
        result = draft_excerpts(source, catalog, {"records": [spec]}, 1, 1, recovery=bundle)
    else:
        spec["fields"] = {
            "name": [{**boundary, "quote": "Rest the widget"}],
            "instructions": [boundary],
        }
        result = draft_exercises(source, catalog, {"records": [spec]}, 1, 1, recovery=bundle)
    record = result["records"][0]
    assert record["coverage"] == ("partial" if warning else "complete")
    assert ("unresolved_recovery" in record["gaps"]) == warning
    if kind == "excerpt" and warning:
        review_excerpt(catalog, "widget", record["revision"], "accepted")
        with pytest.raises(ValueError, match="no_eligible_passages"):
            ingest(
                source,
                tmp_path / "index.sqlite",
                1,
                1,
                AuthoredEmbedder(),
                approval={"sha256": recovered["source_sha256"], "pages": [1]},
                recovery=bundle,
                excerpts=catalog,
            )


def test_ordered_native_and_owned_raster_boundaries_preserve_complete_context(
    tmp_path: Path,
) -> None:
    from cited_retrieval import review, search

    source, bundle, catalog = (tmp_path / n for n in ("source.pdf", "bundle.json", "catalog.json"))
    with pymupdf.open() as raster:
        page = raster.new_page(width=300, height=30)
        page.insert_text((5, 20), "Stop if the warning lamp flashes.")
        image = page.get_pixmap().tobytes("png")
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=300)
        page.insert_text((20, 70), "Rest the widget for twelve seconds before resuming.")
        page.insert_image(pymupdf.Rect(20, 120, 320, 150), stream=image)
        doc.save(source)
    roi = {"pdf_page": 1, "bbox": [20, 120, 320, 150]}
    recovered = recover(
        source,
        bundle,
        1,
        1,
        AuthoredOCR("Stop if the warning lamp flashes."),
        image_regions=[{"id": "warning", **roi}],
    )
    region = recovered["regions"][0]
    review(bundle, "warning", region["revision"], "accepted")
    native = next(b for b in recovered["blocks"] if not b.get("image_region_id"))
    boundaries = [{"pdf_page": 1, "bbox": native["bbox"]}, roi]
    draft = draft_excerpts(
        source,
        catalog,
        {
            "records": [
                {
                    "id": "p",
                    "coverage": "complete",
                    "boundaries": boundaries,
                    "selectors": boundaries,
                }
            ]
        },
        1,
        1,
        recovery=bundle,
    )
    record = draft["records"][0]
    assert record["coverage"] == "complete"
    review_excerpt(catalog, "p", record["revision"], "accepted")
    index = tmp_path / "index.sqlite"
    result = ingest(
        source,
        index,
        1,
        1,
        AuthoredEmbedder(),
        approval={"sha256": recovered["source_sha256"], "pages": [1]},
        recovery=bundle,
        excerpts=catalog,
    )
    assert result["pages"][0]["status"] == "partial"
    assert result["recovery"]["gaps"]
    assert (
        search(index, "widget warning", AuthoredEmbedder())["passages"][0]["text"]
        == "Rest the widget for twelve seconds before resuming.\nStop if the warning lamp flashes."
    )
