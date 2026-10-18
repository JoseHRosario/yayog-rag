from pathlib import Path

import pymupdf
import pytest
from authored_embedder import AuthoredEmbedder
from authored_ocr import AuthoredOCR

from cited_retrieval import ingest, inspect_source, recover


def test_shared_font_family_does_not_ocr_healthy_native_lines(tmp_path: Path):
    source = tmp_path / "shared-family.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        good = page.insert_font(
            fontname="good", fontbuffer=pymupdf.Font("helv").buffer, set_simple=True
        )
        bad = page.insert_font(fontname="bad", fontbuffer=pymupdf.Font("helv").buffer)
        doc.xref_set_key(bad, "BaseFont", "/NimbusSans-Regular")
        page.insert_text(
            (50, 100), "Rest supports recovery after demanding training.", fontname="good"
        )
        page.insert_text((50, 200), "Hold for twelve seconds before resting.", fontname="bad")
        doc.xref_set_key(bad, "ToUnicode", "null")
        assert good != bad
        doc.save(source)
    result = recover(source, tmp_path / "bundle.json", 1, 1, AuthoredOCR())
    assert len(result["regions"]) == 1
    assert result["blocks"][0]["lines"][0]["region_id"] is None
    assert (
        result["blocks"][0]["lines"][0]["text"]
        == "Rest supports recovery after demanding training."
    )


def test_blank_rendered_font_spacing_is_not_invented_as_ocr_text(tmp_path: Path):
    source = tmp_path / "spacing.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 100), "Rest supports recovery after demanding training.")
        font = page.insert_font(fontname="damaged", fontbuffer=pymupdf.Font("helv").buffer)
        page.insert_text((50, 150), " ", fontname="damaged")
        page.insert_text((50, 200), "Hold for twelve seconds before resting.", fontname="damaged")
        doc.xref_set_key(font, "ToUnicode", "null")
        doc.save(source)
    result = recover(source, tmp_path / "bundle.json", 1, 1, AuthoredOCR())
    assert len(result["regions"]) == 1
    blank = next(
        line for block in result["blocks"] for line in block["lines"] if line.get("rendered_blank")
    )
    assert blank["original_text"]
    assert blank["text"] == ""
    assert blank["region_id"] is None


@pytest.mark.parametrize(
    "replacement,reason",
    [("0001", "invalid_unicode"), ("fffd", "invalid_unicode"), (None, "suspicious_native")],
)
def test_raw_span_damage_and_suspicious_native_cannot_bypass_recovery(
    tmp_path: Path, replacement: str | None, reason: str
):
    source = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        text = (
            "Rest and sleep support recovery after training."
            if replacement
            else "aaaaaa aaaaaa aaaaaa aaaaaa aaaaaa aaaaaa"
        )
        if replacement == "0001":
            font = page.insert_font(fontname="fixture", fontbuffer=pymupdf.Font("cjk").buffer)
            page.insert_text((50, 100), text, fontname="fixture")
            cmap = int(doc.xref_get_key(font, "ToUnicode")[1].split()[0])
            doc.update_stream(
                cmap,
                doc.xref_stream(cmap).replace(b"<0001> <005f> <0020>", b"<0001> <005f> <0001>"),
            )
        else:
            page.insert_text((50, 100), text)
        if replacement == "fffd":
            font = page.get_fonts()[0][0]
            cmap = doc.get_new_xref()
            doc.update_object(cmap, "<<>>")
            doc.update_stream(
                cmap,
                (
                    "/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CMapType 2 def 1 begincodespacerange <00> <ff> endcodespacerange 1 beginbfchar <52> <"
                    + replacement
                    + "> endbfchar endcmap end end"
                ).encode(),
            )
            doc.xref_set_key(font, "ToUnicode", f"{cmap} 0 R")
        doc.save(source)
    preview = inspect_source(source, 1, 1)
    assert preview["pages"][0]["status"] == "excluded"
    approval = {"sha256": preview["sha256"], "pages": [1]}
    with pytest.raises(ValueError, match="no_eligible_passages"):
        ingest(source, tmp_path / "index.sqlite", 1, 1, AuthoredEmbedder(), approval=approval)
    result = recover(source, tmp_path / "bundle.json", 1, 1, AuthoredOCR())
    assert result["regions"][0]["reason"] == reason
