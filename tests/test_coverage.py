"""Full-source audit uses authored PDFs through public boundaries only."""

import json
from pathlib import Path
from typing import Any

import pymupdf
import pytest

import cited_retrieval as api


def source(tmp_path: Path) -> Path:
    pdf = tmp_path / "authored.pdf"
    with pymupdf.open() as doc:
        doc.new_page()  # Deliberately textless front matter.
        doc.new_page().insert_text((40, 50), "A synthetic instruction with enough words to read.")
        doc.set_page_labels([{"startpage": 0, "prefix": "", "style": "r", "firstpagenum": 1}])
        doc.save(pdf)
    return pdf


def test_draft_accounts_for_every_page_without_implying_review(tmp_path: Path) -> None:
    pdf = source(tmp_path)
    audit = tmp_path / "audit.json"
    result = api.draft_coverage(pdf, audit)
    assert [p["pdf_page"] for p in result["pages"]] == [1, 2]
    assert [p["printed_label"] for p in result["pages"]] == ["i", "ii"]
    assert result["pages"][0]["native_reason"] == "no_native_text"
    assert all(p["status"] == "unresolved" for p in result["pages"])
    assert result["coverage_complete"] is False
    assert result["gates"] == {"offline": "not_run", "private": "not_run", "live": "not_run"}
    assert result["binding"] == json.loads(audit.read_text())["binding"]
    assert api.inspect_coverage(audit) == result
    assert audit.stat().st_mode & 0o777 == 0o600
    assert json.loads(audit.read_text())["binding"]["source_sha256"] == result["source_sha256"]


def reviewed_pages(audit: Path) -> dict[str, Any]:
    manifest = json.loads(audit.read_text())
    for page in manifest["pages"]:
        page.update(
            status="reviewed_exclusion",
            reason="Front matter visually reviewed",
            whole_page_reviewed=True,
            inventory_complete=True,
        )
    audit.write_text(json.dumps(manifest))
    return manifest


def test_reviewed_exclusions_are_deliberate_not_automatic(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    api.draft_coverage(source(tmp_path), audit)
    reviewed_pages(audit)
    result = api.inspect_coverage(audit)
    assert result["coverage_complete"] is True
    assert result["delivery_complete"] is False
    assert result["counts"]["pages"] == {"trusted": 0, "reviewed_exclusion": 2, "unresolved": 0}


@pytest.mark.parametrize(
    "mutation", ["fragment", "inventory", "duplicate", "missing", "label", "boolean"]
)
def test_rejects_incomplete_or_corrupt_whole_page_accounting(tmp_path: Path, mutation: str) -> None:
    audit = tmp_path / "audit.json"
    api.draft_coverage(source(tmp_path), audit)
    manifest = reviewed_pages(audit)
    if mutation == "fragment":
        manifest["pages"][0]["whole_page_reviewed"] = False
    elif mutation == "inventory":
        manifest["pages"][0]["inventory_complete"] = False
    elif mutation == "duplicate":
        manifest["pages"].append(manifest["pages"][0])
    elif mutation == "missing":
        manifest["pages"].pop()
    elif mutation == "label":
        manifest["pages"][0]["printed_label"] = "invented"
    else:
        manifest["pages"][0]["pdf_page"] = True
    audit.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid_coverage_manifest"):
        api.inspect_coverage(audit)


def test_changed_source_invalidates_review(tmp_path: Path) -> None:
    pdf = source(tmp_path)
    audit = tmp_path / "audit.json"
    api.draft_coverage(pdf, audit)
    reviewed_pages(audit)
    pdf.write_bytes(pdf.read_bytes() + b"\n% changed artifact\n")
    with pytest.raises(ValueError, match="stale_coverage"):
        api.inspect_coverage(audit)


def test_catalog_records_and_index_do_not_imply_whole_page_review(tmp_path: Path) -> None:
    from test_exercises import accept, build, fixture

    pdf, catalog, _ = fixture(tmp_path)
    accept(catalog)
    index = build(pdf, catalog, tmp_path)
    audit = tmp_path / "audit.json"
    result = api.draft_coverage(pdf, audit, indexes=[index])
    assert len(result["records"]) == 2
    assert all(r["status"] == "trusted" for r in result["records"])
    assert all(p["status"] == "unresolved" for p in result["pages"])
    assert result["coverage_complete"] is False
    first = api.inspect_exercises(catalog)["records"][0]
    api.review_exercise(catalog, first["id"], first["revision"], "rejected")
    with pytest.raises(ValueError, match="stale_coverage|stale_exercises"):
        api.inspect_coverage(audit)


@pytest.mark.parametrize("mutation", ["delete", "identity", "append", "duplicate"])
def test_missing_relevant_record_is_a_visible_coverage_blocker(
    tmp_path: Path, mutation: str
) -> None:
    audit = tmp_path / "audit.json"
    pdf = source(tmp_path)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "source_sha256": api.inspect_source(pdf, 1, 2)["sha256"],
                "records": [
                    {
                        "key": "missing-principle",
                        "kind": "principle",
                        "pdf_pages": [2],
                        "status": "unresolved",
                        "reason": "Not catalogued",
                        "dependencies": ["missing-continuation"],
                    }
                ],
            }
        )
    )
    api.draft_coverage(pdf, audit, inventory=inventory)
    manifest = reviewed_pages(audit)
    result = api.inspect_coverage(audit)
    assert result["coverage_complete"] is False
    assert result["counts"]["records"]["unresolved"] == 1
    assert "missing-continuation" in str(result["records"])
    manifest["records"][0].update(status="reviewed_exclusion", reason="Explicit owner exclusion")
    audit.write_text(json.dumps(manifest))
    assert api.inspect_coverage(audit)["coverage_complete"] is True
    manifest["records"][0]["status"] = "unresolved"
    if mutation == "delete":
        manifest["records"].clear()
    elif mutation == "identity":
        manifest["records"][0]["dependencies"] = []
    else:
        record = dict(manifest["records"][0])
        if mutation == "append":
            record["key"] = "unregistered"
        manifest["records"].append(record)
    audit.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid_coverage_manifest"):
        api.inspect_coverage(audit)


def test_draft_protects_existing_input_and_prior_audit(tmp_path: Path) -> None:
    from test_exercises import fixture

    pdf, catalog, _ = fixture(tmp_path)
    original = catalog.read_bytes()
    with pytest.raises(ValueError, match="unsafe_coverage_path"):
        api.draft_coverage(pdf, catalog, exercise_catalogs=[catalog])
    assert catalog.read_bytes() == original
    audit = tmp_path / "audit.json"
    api.draft_coverage(pdf, audit)
    original = audit.read_bytes()
    with pytest.raises(ValueError, match="unsafe_coverage_path"):
        api.draft_coverage(pdf, audit)
    assert audit.read_bytes() == original


def test_evaluation_manifest_separates_gates_and_checks_source_locations(tmp_path: Path) -> None:
    import hashlib

    audit = tmp_path / "audit.json"
    api.draft_coverage(source(tmp_path), audit)
    manifest = reviewed_pages(audit)
    evidence = tmp_path / "owner-results.txt"
    evidence.write_text("Authored offline owner review, not real model acceptance.")
    reference = {"path": str(evidence), "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}
    manifest["gates"]["offline"] = {"status": "passed", "evidence": reference}
    categories = [
        "exact_name",
        "paraphrase",
        "variation",
        "session_filter",
        "unsupported",
        "conflict",
    ]
    for category in categories:
        manifest["evaluations"].append(
            {
                "id": category,
                "category": category,
                "question": "Authored question?",
                "status": "passed",
                "claim_support": "Compared claims with original source",
                "expected_behavior": "abstain"
                if category == "unsupported"
                else "disclose_conflict"
                if category == "conflict"
                else "answer",
                "locations": []
                if category == "unsupported"
                else [{"pdf_page": 2, "printed_label": "ii", "quote": "A synthetic instruction"}],
                "evidence": reference,
            }
        )
    # A conflict requires two separately checkable source claims.
    manifest["evaluations"][-1]["locations"].append(
        {"pdf_page": 2, "printed_label": "ii", "quote": "enough words"}
    )
    for case in manifest["evaluations"]:
        case_file = tmp_path / (case["id"] + "-result.json")
        case_file.write_text(
            json.dumps(
                {
                    "case": {
                        "source_sha256": manifest["binding"]["source_sha256"],
                        **{
                            k: case[k]
                            for k in (
                                "id",
                                "category",
                                "question",
                                "expected_behavior",
                                "locations",
                            )
                        },
                    },
                    "result": "Authored reviewed result; no live model",
                }
            )
        )
        case["evidence"] = {
            "path": str(case_file),
            "sha256": hashlib.sha256(case_file.read_bytes()).hexdigest(),
        }
    audit.write_text(json.dumps(manifest))
    result = api.inspect_coverage(audit)
    assert result["evaluation_complete"] is True
    assert result["gates"] == {"offline": "passed", "private": "not_run", "live": "not_run"}
    assert result["delivery_complete"] is False
    manifest["evaluations"][0]["locations"][0]["quote"] = "Invented unsupported instruction"
    audit.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid_evaluation_location"):
        api.inspect_coverage(audit)


@pytest.mark.parametrize("changed_field", ["question", "locations", "id", "category", "filters"])
def test_evaluation_result_cannot_be_reused_for_a_changed_case(
    tmp_path: Path, changed_field: str
) -> None:
    import hashlib

    audit = tmp_path / "audit.json"
    api.draft_coverage(source(tmp_path), audit)
    manifest = reviewed_pages(audit)
    case = {
        "id": "literal-question",
        "category": "exact_name",
        "question": "What is the synthetic instruction?",
        "expected_behavior": "answer",
        "locations": [{"pdf_page": 2, "printed_label": "ii", "quote": "A synthetic instruction"}],
        "status": "passed",
        "claim_support": "Literal authored source comparison, not actual model acceptance",
        "filters": {"programme": "Synthetic Program", "week": 1, "day": 1},
    }
    evidence = tmp_path / "case-result.json"
    evidence.write_text(
        json.dumps(
            {
                "case": {
                    "source_sha256": manifest["binding"]["source_sha256"],
                    "filters": {"programme": "Synthetic Program", "week": 1, "day": 1},
                    **{
                        k: case[k]
                        for k in ("id", "category", "question", "expected_behavior", "locations")
                    },
                },
                "result": "Authored synthetic instruction",
            }
        )
    )
    case["evidence"] = {
        "path": str(evidence),
        "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
    }
    manifest["evaluations"] = [case]
    audit.write_text(json.dumps(manifest))
    assert api.inspect_coverage(audit)["evaluations"][0]["status"] == "passed"
    changes = {
        "question": "A different question using the same old result?",
        "locations": [{"pdf_page": 2, "printed_label": "ii", "quote": "enough words"}],
        "id": "a-different-case",
        "category": "paraphrase",
        "filters": {"programme": "Synthetic Program", "week": True, "day": 1},
    }
    case[changed_field] = changes[changed_field]
    audit.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="coverage_case_binding_mismatch"):
        api.inspect_coverage(audit)


def test_operator_evidence_changes_invalidate_recorded_gate(tmp_path: Path) -> None:
    import hashlib

    audit = tmp_path / "audit.json"
    api.draft_coverage(source(tmp_path), audit)
    manifest = json.loads(audit.read_text())
    evidence = tmp_path / "results.txt"
    evidence.write_text("First reviewed result")
    manifest["gates"]["live"] = {
        "status": "passed",
        "evidence": {
            "path": str(evidence),
            "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        },
    }
    audit.write_text(json.dumps(manifest))
    evidence.write_text("Changed result")
    with pytest.raises(ValueError, match="stale_coverage_evidence"):
        api.inspect_coverage(audit)


def test_bound_region_exclusion_is_distinct_from_pending_body_without_granting_trust(
    tmp_path: Path,
) -> None:
    from authored_ocr import AuthoredOCR
    from test_recovery import damaged_source

    pdf = damaged_source(tmp_path)
    bundle = tmp_path / "review" / "bundle.json"
    recovered = api.recover(pdf, bundle, 1, 1, AuthoredOCR())
    region = recovered["regions"][0]
    api.review(bundle, region["id"], region["revision"], "rejected")
    audit = tmp_path / "audit.json"
    result = api.draft_coverage(pdf, audit, recovery_bundles=[bundle])
    assert result["regions"][0]["status"] == "unresolved"
    manifest = reviewed_pages(audit)
    manifest["pages"][0].update(
        status="trusted", reason="Entire page reviewed; running footer excluded"
    )
    assert api.inspect_coverage(audit)["coverage_complete"] is False
    manifest["regions"][0].update(
        status="reviewed_exclusion", reason="Visually verified running footer, not body"
    )
    audit.write_text(json.dumps(manifest))
    result = api.inspect_coverage(audit)
    assert result["coverage_complete"] is True
    assert result["regions"][0]["revision"] == region["revision"]
    assert result["regions"][0]["bbox"] == region["bbox"]
    assert api.inspect_review(bundle)["regions"][0]["state"] == "rejected"
    manifest["regions"][0]["bbox"][0] += 1
    audit.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid_coverage_manifest"):
        api.inspect_coverage(audit)


def test_explicit_missing_body_gap_prevents_whole_source_completeness(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    api.draft_coverage(source(tmp_path), audit)
    manifest = reviewed_pages(audit)
    manifest["pages"][1].update(status="trusted", gaps=["missing_body_continuation"])
    audit.write_text(json.dumps(manifest))
    result = api.inspect_coverage(audit)
    assert result["coverage_complete"] is False
    assert result["pages"][1]["status"] == "unresolved"
    assert result["pages"][1]["declared_status"] == "trusted"


def test_coverage_cli_runs_offline_and_redacts_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cited_retrieval.cli import main

    pdf = source(tmp_path)
    audit = tmp_path / "audit.json"
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "source_sha256": api.inspect_source(pdf, 1, 2)["sha256"],
                "records": [
                    {
                        "key": "missing",
                        "kind": "principle",
                        "pdf_pages": [2],
                        "status": "unresolved",
                        "reason": "Awaiting mapping",
                    }
                ],
            }
        )
    )
    assert (
        main(["draft-coverage", str(pdf), "--audit", str(audit), "--inventory", str(inventory)])
        == 0
    )
    drafted = json.loads(capsys.readouterr().out)
    assert drafted["records"][0]["key"] == "missing"
    assert drafted["binding"]["inventory"]["path"] == str(inventory)
    assert drafted["coverage_complete"] is False
    assert main(["inspect-coverage", str(audit)]) == 0
    assert json.loads(capsys.readouterr().out) == drafted
    assert main(["draft-coverage", str(pdf), "--audit", str(pdf)]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "error",
        "error": "unsafe_coverage_path",
    }


def test_identical_drafts_are_reproducible_and_catalog_omission_is_rejected(tmp_path: Path) -> None:
    from test_exercises import fixture

    pdf, catalog, _ = fixture(tmp_path)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    api.draft_coverage(pdf, first, exercise_catalogs=[catalog])
    api.draft_coverage(pdf, second, exercise_catalogs=[catalog])
    assert first.read_bytes() == second.read_bytes()
    manifest = json.loads(first.read_text())
    manifest["records"].pop()
    first.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid_coverage_manifest"):
        api.inspect_coverage(first)


@pytest.mark.parametrize("evidence_status", ["trusted", "reviewed_exclusion", "unresolved"])
def test_explicit_principle_review_can_reference_bound_prose_without_inferring_page_review(
    tmp_path: Path,
    evidence_status: str,
) -> None:
    from authored_embedder import AuthoredEmbedder

    pdf = source(tmp_path)
    index = tmp_path / "prose.sqlite"
    api.ingest(
        pdf,
        index,
        1,
        2,
        AuthoredEmbedder(),
        approval={"sha256": api.inspect_source(pdf, 1, 2)["sha256"], "pages": [1, 2]},
    )
    audit = tmp_path / "audit.json"
    result = api.draft_coverage(pdf, audit, indexes=[index])
    assert len(result["records"]) == 1
    passage = result["records"][0]
    assert passage["kind"] == "passage"
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "source_sha256": result["source_sha256"],
                "records": [
                    {
                        "key": "principle-1",
                        "kind": "principle",
                        "pdf_pages": [2],
                        "status": "trusted",
                        "reason": "Whole principle reviewed including qualifications",
                        "whole_record_reviewed": True,
                        "evidence_keys": [passage["key"]],
                        "dependencies": [],
                    }
                ],
            }
        )
    )
    audit = tmp_path / "registered.json"
    api.draft_coverage(pdf, audit, indexes=[index], inventory=inventory)
    manifest = json.loads(audit.read_text())
    manifest["records"][0]["status"] = evidence_status
    audit.write_text(json.dumps(manifest))
    if evidence_status != "trusted":
        with pytest.raises(ValueError, match="invalid_coverage_manifest"):
            api.inspect_coverage(audit)
        return
    result = api.inspect_coverage(audit)
    assert result["records"][-1]["status"] == "trusted"
    assert result["coverage_complete"] is False
    for number, (field, value) in enumerate(
        [
            ("evidence_keys", ["invented-passage"]),
            ("evidence_keys", ["principle-1"]),
            ("evidence_keys", []),
            ("whole_record_reviewed", False),
            ("pdf_pages", [1, 2]),
            ("dependencies", ["missing-qualification"]),
            ("gaps", ["missing-body"]),
            ("source_sha256", "another-source"),
        ]
    ):
        invalid = json.loads(inventory.read_text())
        invalid["records"][0][field] = value
        invalid_inventory = tmp_path / f"invalid-inventory-{number}.json"
        invalid_inventory.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="invalid_coverage_manifest"):
            api.draft_coverage(
                pdf,
                tmp_path / f"invalid-audit-{number}.json",
                indexes=[index],
                inventory=invalid_inventory,
            )
    manifest["records"][-1]["evidence_keys"] = ["invented-passage"]
    audit.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid_coverage_manifest"):
        api.inspect_coverage(audit)


def test_accepted_session_with_missing_method_and_exercises_remains_audit_unresolved(
    tmp_path: Path,
) -> None:
    from test_programmes import accept, schedule

    pdf, catalog, layout, _ = schedule(tmp_path)
    api.draft_programmes(pdf, catalog, layout, 1, 2)
    accept(catalog)
    audit = tmp_path / "audit.json"
    result = api.draft_coverage(pdf, audit, programme_catalogs=[catalog])
    session = result["records"][0]
    assert session["eligible"] is True  # Existing retrieval policy is unchanged.
    assert session["status"] == "unresolved"
    assert any("method_reference" in d for d in session["dependencies"])
    assert any("exercise_reference" in d for d in session["dependencies"])


@pytest.mark.parametrize("payload", ["{}", "[]", '{"schema": 1}', "not json"])
def test_malformed_operator_manifest_has_safe_public_error(tmp_path: Path, payload: str) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(payload)
    with pytest.raises(ValueError, match="invalid_coverage_manifest"):
        api.inspect_coverage(audit)


def test_missing_source_is_stale_not_an_unbound_report(tmp_path: Path) -> None:
    pdf = source(tmp_path)
    audit = tmp_path / "audit.json"
    api.draft_coverage(pdf, audit)
    pdf.unlink()
    with pytest.raises(ValueError, match="stale_coverage"):
        api.inspect_coverage(audit)


@pytest.mark.parametrize("linked", [False, True])
def test_private_coverage_rejects_any_git_ancestor_on_draft_and_relocation(
    tmp_path: Path, linked: bool
) -> None:
    pdf = source(tmp_path)
    audit = tmp_path / "private.json"
    api.draft_coverage(pdf, audit)
    checkout = tmp_path / "other-checkout"
    checkout.mkdir()
    if linked:
        (checkout / ".git").write_text("gitdir: /unused/worktree")
    else:
        (checkout / ".git").mkdir()
    nested = checkout / "nested"
    nested.mkdir()
    destination = nested / "audit.json"
    with pytest.raises(ValueError, match="unsafe_coverage_path"):
        api.draft_coverage(pdf, destination)
    assert not destination.exists()
    destination.write_bytes(audit.read_bytes())
    with pytest.raises(ValueError, match="unsafe_coverage_path"):
        api.inspect_coverage(destination)


def test_inventory_change_requires_new_draft_and_retains_prior_audit(tmp_path: Path) -> None:
    pdf = source(tmp_path)
    inventory = tmp_path / "inventory-v1.json"
    value = {"source_sha256": api.inspect_source(pdf, 1, 2)["sha256"], "records": []}
    inventory.write_text(json.dumps(value))
    audit = tmp_path / "audit.json"
    api.draft_coverage(pdf, audit, inventory=inventory)
    original = audit.read_bytes()
    inventory.write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="stale_coverage"):
        api.inspect_coverage(audit)
    assert audit.read_bytes() == original
    api.draft_coverage(pdf, tmp_path / "new.json", inventory=inventory)
    assert audit.read_bytes() == original


@pytest.mark.parametrize("collision", ["source", "catalog", "target"])
def test_inventory_cannot_alias_another_input_or_output(tmp_path: Path, collision: str) -> None:
    from test_exercises import fixture

    pdf, catalog, _ = fixture(tmp_path)
    audit = tmp_path / "audit.json"
    inventory = {"source": pdf, "catalog": catalog, "target": audit}[collision]
    if collision == "target":
        inventory.write_text("{}")
    before = inventory.read_bytes()
    with pytest.raises(ValueError, match="unsafe_coverage_path"):
        api.draft_coverage(pdf, audit, exercise_catalogs=[catalog], inventory=inventory)
    assert inventory.read_bytes() == before


@pytest.mark.parametrize("linked", [False, True])
def test_inventory_private_boundary_on_creation_and_after_relocation(
    tmp_path: Path, linked: bool
) -> None:
    pdf = source(tmp_path)
    folder = tmp_path / "inventory-folder"
    folder.mkdir()
    inventory = folder / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "source_sha256": api.inspect_source(pdf, 1, 2)["sha256"],
                "records": [],
            }
        )
    )
    audit = tmp_path / "audit.json"
    api.draft_coverage(pdf, audit, inventory=inventory)
    if linked:
        (folder / ".git").write_text("gitdir: /unused/worktree")
    else:
        (folder / ".git").mkdir()
    with pytest.raises(ValueError, match="unsafe_coverage_path"):
        api.draft_coverage(pdf, tmp_path / "new.json", inventory=inventory)
    with pytest.raises(ValueError, match="unsafe_coverage_path"):
        api.inspect_coverage(audit)


def test_inventory_source_must_match_before_draft_writes_output(tmp_path: Path) -> None:
    pdf = source(tmp_path)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"source_sha256": "another-source", "records": []}))
    audit = tmp_path / "audit.json"
    with pytest.raises(ValueError, match="invalid_coverage_manifest"):
        api.draft_coverage(pdf, audit, inventory=inventory)
    assert not audit.exists()


def test_unregistered_manual_record_cannot_be_appended(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    api.draft_coverage(source(tmp_path), audit)
    manifest = reviewed_pages(audit)
    manifest["records"].append(
        {
            "key": "unregistered",
            "kind": "principle",
            "pdf_pages": [2],
            "status": "unresolved",
            "reason": "Not registered",
            "dependencies": [],
        }
    )
    audit.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid_coverage_manifest"):
        api.inspect_coverage(audit)


def test_repeated_index_arguments_do_not_duplicate_inventory(tmp_path: Path) -> None:
    from test_exercises import accept, build, fixture

    pdf, catalog, _ = fixture(tmp_path)
    accept(catalog)
    index = build(pdf, catalog, tmp_path)
    audit = tmp_path / "audit.json"
    result = api.draft_coverage(
        pdf, audit, indexes=[index, index], exercise_catalogs=[catalog, catalog]
    )
    assert len(result["records"]) == 2
    assert len(json.loads(audit.read_text())["binding"]["artifacts"]["indexes"]) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", ""),
        ("source_sha256", "not-a-source"),
        ("dependencies", "not-a-list"),
        ("pdf_pages", [True]),
        ("pdf_pages", [2, 2]),
    ],
)
def test_missing_record_inventory_rejects_invalid_identity(
    tmp_path: Path, field: str, value: str | list[int] | list[bool]
) -> None:
    audit = tmp_path / "audit.json"
    pdf = source(tmp_path)
    record = {
        "key": "missing-1",
        "kind": "principle",
        "pdf_pages": [2],
        "status": "unresolved",
        "reason": "Missing body",
        "dependencies": [],
    }
    record[field] = value
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "source_sha256": api.inspect_source(pdf, 1, 2)["sha256"],
                "records": [record],
            }
        )
    )
    original_pdf = pdf.read_bytes()
    original_inventory = inventory.read_bytes()
    with pytest.raises(ValueError, match="invalid_coverage_manifest"):
        api.draft_coverage(pdf, audit, inventory=inventory)
    assert not audit.exists()
    assert pdf.read_bytes() == original_pdf
    assert inventory.read_bytes() == original_inventory
    record.update(kind="principle", pdf_pages=[2], dependencies=[])
    record.pop("source_sha256", None)
    value = json.loads(inventory.read_text())
    value["records"] = [record]
    inventory.write_text(json.dumps(value))
    result = api.draft_coverage(pdf, audit, inventory=inventory)
    assert result == api.inspect_coverage(audit)
    assert result["records"][0]["key"] == "missing-1"


@pytest.mark.parametrize("changed_input", ["source", "index", "inventory", "gate", "case", "audit"])
def test_input_change_during_evidence_read_cannot_return_delivery_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_input: str
) -> None:
    import hashlib

    from authored_embedder import AuthoredEmbedder

    pdf = source(tmp_path)
    index = tmp_path / "prose.sqlite"
    source_hash = api.inspect_source(pdf, 1, 2)["sha256"]
    api.ingest(
        pdf, index, 1, 2, AuthoredEmbedder(), approval={"sha256": source_hash, "pages": [1, 2]}
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"source_sha256": source_hash, "records": []}))
    audit = tmp_path / "audit.json"
    api.draft_coverage(pdf, audit, indexes=[index], inventory=inventory)
    manifest = reviewed_pages(audit)
    evidence = tmp_path / "gate.txt"
    evidence.write_text("Synthetic owner evidence, not live model acceptance")
    reference = {"path": str(evidence), "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}
    for gate in manifest["gates"].values():
        gate.update(status="passed", evidence=reference)
    for category in (
        "exact_name",
        "paraphrase",
        "variation",
        "session_filter",
        "unsupported",
        "conflict",
    ):
        locations = (
            []
            if category == "unsupported"
            else [{"pdf_page": 2, "printed_label": "ii", "quote": "A synthetic instruction"}]
        )
        if category == "conflict":
            locations.append({"pdf_page": 2, "printed_label": "ii", "quote": "enough words"})
        case = {
            "id": category,
            "category": category,
            "question": "Authored question?",
            "expected_behavior": "abstain"
            if category == "unsupported"
            else "disclose_conflict"
            if category == "conflict"
            else "answer",
            "locations": locations,
        }
        case_file = tmp_path / (category + ".json")
        case_file.write_text(
            json.dumps(
                {
                    "case": {**case, "source_sha256": manifest["binding"]["source_sha256"]},
                    "result": "Authored review only",
                }
            )
        )
        manifest["evaluations"].append(
            {
                **case,
                "status": "passed",
                "claim_support": "Authored source comparison",
                "evidence": {
                    "path": str(case_file),
                    "sha256": hashlib.sha256(case_file.read_bytes()).hexdigest(),
                },
            }
        )
    audit.write_text(json.dumps(manifest))
    assert api.inspect_coverage(audit)["delivery_complete"] is True
    original_audit = audit.read_bytes()
    changed_path = {
        "source": pdf,
        "index": index,
        "inventory": inventory,
        "gate": evidence,
        "case": case_file,
        "audit": audit,
    }[changed_input]
    original_input = changed_path.read_bytes()
    original_read = Path.read_bytes
    triggered = False

    def read_and_change_input(path: Path) -> bytes:
        nonlocal triggered
        data = original_read(path)
        if path == case_file and not triggered:
            triggered = True
            changed_path.write_bytes(original_input + b"\n")
        return data

    monkeypatch.setattr(Path, "read_bytes", read_and_change_input)
    with pytest.raises(ValueError, match="stale_coverage"):
        api.inspect_coverage(audit)
    assert triggered
    assert audit.read_bytes() == original_audit + (b"\n" if changed_input == "audit" else b"")


def test_transitive_catalog_change_during_evidence_read_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from test_exercises import accept, build, fixture

    pdf, catalog, _ = fixture(tmp_path)
    accept(catalog)
    index = build(pdf, catalog, tmp_path)
    audit = tmp_path / "audit.json"
    api.draft_coverage(pdf, audit, indexes=[index])
    manifest = json.loads(audit.read_text())
    evidence = tmp_path / "gate.txt"
    evidence.write_text("Authored gate evidence")
    manifest["gates"]["offline"] = {
        "status": "passed",
        "evidence": {
            "path": str(evidence),
            "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        },
    }
    audit.write_text(json.dumps(manifest))
    original_audit = audit.read_bytes()
    original_catalog = catalog.read_bytes()
    original_read = Path.read_bytes
    triggered = False

    def read_and_change_dependency(path: Path) -> bytes:
        nonlocal triggered
        data = original_read(path)
        if path == evidence and not triggered:
            triggered = True
            catalog.write_bytes(original_catalog + b"\n")
        return data

    monkeypatch.setattr(Path, "read_bytes", read_and_change_dependency)
    with pytest.raises(ValueError, match="stale_coverage"):
        api.inspect_coverage(audit)
    assert triggered
    assert audit.read_bytes() == original_audit


def test_atomic_draft_collision_preserves_concurrent_valid_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    pdf = source(tmp_path)
    previous = tmp_path / "previous.json"
    api.draft_coverage(pdf, previous)
    reviewed_pages(previous)
    original = previous.read_bytes()
    target = tmp_path / "audit.json"
    original_link = os.link
    triggered = False

    def link_after_concurrent_publication(src: Path, dst: Path) -> None:
        nonlocal triggered
        assert not dst.exists()
        assert json.loads(src.read_text())["schema"] == 1
        assert src.stat().st_mode & 0o777 == 0o600
        triggered = True
        dst.write_bytes(original)
        original_link(src, dst)

    monkeypatch.setattr(os, "link", link_after_concurrent_publication)
    with pytest.raises(ValueError, match="unsafe_coverage_path"):
        api.draft_coverage(pdf, target)
    assert triggered
    assert previous.read_bytes() == original
    assert target.read_bytes() == original
    assert api.inspect_coverage(target)["coverage_complete"] is True
    assert list(tmp_path.glob(".coverage-*")) == []
