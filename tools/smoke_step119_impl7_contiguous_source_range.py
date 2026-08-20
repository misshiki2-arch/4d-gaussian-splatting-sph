#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import tools.smoke_step119_impl2_population_aligned_reference as IMPL2  # noqa: E402


TOOL = IMPL2.TOOL
MANIFEST = IMPL2.MANIFEST


def population_cli(**overrides):
    values = {
        "population_prefix_count": None,
        "population_range_start": None,
        "population_range_count": None,
        "population_provenance": None,
        "population_spl4": None,
        "reuse_legacy_training_report": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def write_range_provenance(
    path: Path,
    checkpoint_path: Path,
    spl4_path: Path,
    *,
    source_count: int = 7,
    start: int = 2,
    count: int = 4,
    overrides=None,
):
    checkpoint_identity = MANIFEST.file_identity(checkpoint_path)
    spl4_identity = MANIFEST.file_identity(spl4_path)
    range_hash = "7b" * 32
    provenance = {
        "schemaVersion": TOOL.POPULATION_PROVENANCE_SCHEMA_VERSION,
        "verificationMode": TOOL.POPULATION_PROVENANCE_VERIFICATION_MODE,
        "decision": "ready",
        "blockedReasons": [],
        "exactPopulationMappingReady": True,
        "rangeHashesMatch": True,
        "recordCountsMatch": True,
        "headerCompatible": True,
        "rangeInBounds": True,
        "assetPayloadLengthValid": True,
        "checkpointSourceGaussianCount": source_count,
        "assetRecordCount": source_count,
        "selection": {
            "policy": "contiguous-source-index-range",
            "startInclusive": start,
            "endExclusive": start + count,
            "selectedRecordCount": count,
        },
        "checkpoint": checkpoint_identity,
        "spl4Asset": spl4_identity,
        "spl4Format": {
            "headerSize": 128,
            "recordCount": source_count,
            "recordStrideBytes": 260,
        },
        "expectedSerializedByteCount": count * 260,
        "actualAssetRangeByteCount": count * 260,
        "checkpointRangeSha256": range_hash,
        "assetRangeSha256": range_hash,
    }
    if overrides:
        provenance.update(overrides)
    path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return provenance


def make_gate_root(root: Path, *, start: int = 2, count: int = 4):
    root.mkdir(parents=True)
    checkpoint = root / "checkpoint.pth"
    spl4 = root / "scene.splat4d"
    provenance = root / "provenance.json"
    checkpoint.write_bytes(b"synthetic-checkpoint-identity")
    spl4.write_bytes(b"SPL4" + bytes(128 + 7 * 260 - 4))
    write_range_provenance(
        provenance,
        checkpoint,
        spl4,
        start=start,
        count=count,
    )
    gate = TOOL.validate_population_provenance(
        provenance_path=provenance,
        checkpoint_path=checkpoint,
        spl4_path=spl4,
        requested_start=start,
        requested_count=count,
    )
    return checkpoint, spl4, provenance, gate


def test_cli_contract(temp: Path):
    assert TOOL.validate_population_selection_cli(population_cli()) is None

    prefix = TOOL.validate_population_selection_cli(population_cli(
        population_prefix_count=4,
        population_provenance=str(temp / "provenance.json"),
        population_spl4=str(temp / "scene.splat4d"),
    ))
    assert prefix["mode"] == "prefix"
    assert prefix["startInclusive"] == 0
    assert prefix["selectedCount"] == 4

    selected_range = TOOL.validate_population_selection_cli(population_cli(
        population_range_start=2,
        population_range_count=4,
        population_provenance=str(temp / "provenance.json"),
        population_spl4=str(temp / "scene.splat4d"),
    ))
    assert selected_range["mode"] == "range"
    assert selected_range["startInclusive"] == 2
    assert selected_range["selectedCount"] == 4
    assert selected_range["endExclusive"] == 6

    invalid_cases = (
        (
            population_cli(
                population_prefix_count=4,
                population_range_start=0,
                population_range_count=4,
                population_provenance="provenance.json",
                population_spl4="scene.splat4d",
            ),
            "population-prefix-and-range-mutually-exclusive",
        ),
        (
            population_cli(
                population_range_start=-1,
                population_range_count=4,
                population_provenance="provenance.json",
                population_spl4="scene.splat4d",
            ),
            "population-range-start-must-be-non-negative",
        ),
        (
            population_cli(
                population_range_start=2,
                population_range_count=0,
                population_provenance="provenance.json",
                population_spl4="scene.splat4d",
            ),
            "population-range-count-must-be-positive",
        ),
        (
            population_cli(
                population_range_start=2,
                population_provenance="provenance.json",
                population_spl4="scene.splat4d",
            ),
            "population-range-count-required",
        ),
    )
    for cli, expected in invalid_cases:
        IMPL2.assert_population_error(
            lambda cli=cli: TOOL.validate_population_selection_cli(cli),
            expected,
        )


def test_range_selection():
    capture = IMPL2.make_capture(record_count=7)
    original_tensors = {
        capture_index: value.clone()
        for capture_index, _ in TOOL.CHECKPOINT_PER_GAUSSIAN_FIELD_INDICES
        for value in (capture[capture_index],)
    }
    selected, application = TOOL.select_checkpoint_source_range(
        capture,
        start_inclusive=2,
        selected_count=4,
        expected_source_count=7,
    )
    assert len(application["perGaussianFieldCounts"]) == 13
    assert application["selectedPerGaussianFieldCount"] == 13
    for capture_index, field_name in TOOL.CHECKPOINT_PER_GAUSSIAN_FIELD_INDICES:
        assert torch.equal(selected[capture_index], capture[capture_index][2:6])
        assert torch.equal(capture[capture_index], original_tensors[capture_index])
        assert application["perGaussianFieldCounts"][field_name] == {
            "sourceRecordCount": 7,
            "selectedRecordCount": 4,
        }
    assert application["startInclusive"] == 2
    assert application["endExclusive"] == 6
    assert application["rasterizerLocalIndexStart"] == 0
    assert application["rasterizerLocalIndexEndExclusive"] == 4
    assert application["originalSourceIndexStart"] == 2
    assert application["originalSourceIndexEndExclusive"] == 6
    assert application["rasterizerLocalIndexEqualsCheckpointSourceIndex"] is False

    prefix, _ = TOOL.select_checkpoint_source_prefix(
        capture,
        selected_count=4,
        expected_source_count=7,
    )
    range_from_zero, zero_application = TOOL.select_checkpoint_source_range(
        capture,
        start_inclusive=0,
        selected_count=4,
        expected_source_count=7,
    )
    for capture_index, _ in TOOL.CHECKPOINT_PER_GAUSSIAN_FIELD_INDICES:
        assert torch.equal(prefix[capture_index], range_from_zero[capture_index])
    assert zero_application["rasterizerLocalIndexEqualsCheckpointSourceIndex"] is True

    for start, count in ((-1, 4), (2, 0), (4, 4)):
        IMPL2.assert_population_error(
            lambda start=start, count=count: TOOL.select_checkpoint_source_range(
                capture,
                start_inclusive=start,
                selected_count=count,
                expected_source_count=7,
            ),
            "population-source-range-out-of-bounds",
        )

    mismatched = list(capture)
    mismatched[4] = mismatched[4][:-1]
    IMPL2.assert_population_error(
        lambda: TOOL.select_checkpoint_source_range(
            tuple(mismatched),
            start_inclusive=2,
            selected_count=4,
            expected_source_count=7,
        ),
        "population-checkpoint-field-count-mismatch",
    )


def test_provenance(temp: Path):
    checkpoint, spl4, provenance, gate = make_gate_root(temp / "valid")
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (checkpoint, spl4, provenance)
    }
    assert all(gate["validationPredicates"].values())
    assert gate["selection"]["startInclusive"] == 2
    assert gate["selection"]["endExclusive"] == 6
    assert gate["serializedByteRange"] == {
        "payloadByteOffset": 648,
        "serializedByteCount": 1040,
        "byteEndExclusive": 1688,
        "recordStrideBytes": 260,
        "headerSizeBytes": 128,
    }
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (checkpoint, spl4, provenance)
    }
    assert before == after

    cases = (
        ("start", {"selection": {
            "policy": "contiguous-source-index-range",
            "startInclusive": 1,
            "endExclusive": 5,
            "selectedRecordCount": 4,
        }}, "selectionMatchesRequestedRange", None),
        ("count", {"selection": {
            "policy": "contiguous-source-index-range",
            "startInclusive": 2,
            "endExclusive": 5,
            "selectedRecordCount": 3,
        }}, "selectionMatchesRequestedRange", None),
        ("hash", {"assetRangeSha256": "00" * 32}, "rangeHashIdentityValid", None),
        ("checkpoint", {}, "checkpointIdentityMatches", "checkpoint"),
        ("spl4", {}, "spl4IdentityMatches", "spl4"),
    )
    for name, overrides, expected, mutate in cases:
        case_root = temp / name
        case_root.mkdir()
        case_checkpoint = case_root / "checkpoint.pth"
        case_spl4 = case_root / "scene.splat4d"
        case_provenance = case_root / "provenance.json"
        case_checkpoint.write_bytes(b"synthetic-checkpoint-identity")
        case_spl4.write_bytes(b"SPL4" + bytes(128 + 7 * 260 - 4))
        write_range_provenance(
            case_provenance,
            case_checkpoint,
            case_spl4,
            overrides=overrides,
        )
        if mutate == "checkpoint":
            case_checkpoint.write_bytes(case_checkpoint.read_bytes() + b"-changed")
        if mutate == "spl4":
            case_spl4.write_bytes(case_spl4.read_bytes() + b"-changed")
        IMPL2.assert_population_error(
            lambda cp=case_checkpoint, asset=case_spl4, prov=case_provenance: (
                TOOL.validate_population_provenance(
                    provenance_path=prov,
                    checkpoint_path=cp,
                    spl4_path=asset,
                    requested_start=2,
                    requested_count=4,
                )
            ),
            expected,
        )
    return checkpoint, gate


def build_range_contract(gate):
    capture = IMPL2.make_capture(record_count=7)
    original_new_model = TOOL.new_gaussian_model
    original_torch_load = TOOL.torch.load
    try:
        selected_model = IMPL2.FakeGaussianModel()
        TOOL.new_gaussian_model = lambda args: selected_model
        TOOL.torch.load = lambda path, map_location: (capture, 12000)
        model, contract = TOOL.build_population_aligned_gaussian_model(
            SimpleNamespace(),
            Path("unused"),
            gate,
            selection_mode="range",
        )
    finally:
        TOOL.new_gaussian_model = original_new_model
        TOOL.torch.load = original_torch_load
    assert model.get_xyz.shape[0] == 4
    assert contract["schemaVersion"] == TOOL.POPULATION_RANGE_SELECTION_SCHEMA_VERSION
    assert contract["selectionMode"] == "range"
    assert contract["requestedSelection"]["startInclusive"] == 2
    assert contract["requestedSelection"]["endExclusive"] == 6
    assert contract["productionRasterizerRecordCount"] == 4
    assert contract["indexLineage"]["mappingPolicy"] == (
        TOOL.POPULATION_LOCAL_TO_ORIGINAL_INDEX_POLICY
    )
    assert all(contract["predicates"].values())
    return contract


def test_manifest(temp: Path, contract):
    original_module_identity = MANIFEST.module_identity
    try:
        MANIFEST.module_identity = lambda name: {
            "moduleName": name,
            "available": True,
            "fixture": True,
        }
        manifest = IMPL2.build_manifest_fixture(temp, contract, gaussian_count=4)
    finally:
        MANIFEST.module_identity = original_module_identity
    published = manifest["lineage"]["populationSelection"]
    assert published["schemaVersion"] == TOOL.POPULATION_RANGE_SELECTION_SCHEMA_VERSION
    assert published["requestedSelection"] == {
        "policy": "contiguous-source-index-range",
        "startInclusive": 2,
        "endExclusive": 6,
        "selectedCount": 4,
    }
    assert published["appliedSelection"]["selectedPerGaussianFieldCount"] == 13
    assert published["indexLineage"]["rangeStart"] == 2
    assert published["indexLineage"]["rangeCount"] == 4
    assert published["indexLineage"]["rangeEndExclusive"] == 6
    assert published["provenance"]["artifact"]["sha256"]
    assert published["populationAlignmentReady"] is True


class FakeEvidenceModel:
    get_xyz = torch.zeros(4, 3)
    get_scaling = torch.ones(4, 3)
    get_rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4)


def test_direct_evidence(temp: Path, contract):
    args = SimpleNamespace(debug_preprocess_target_index=-1)
    cli = SimpleNamespace(
        direct_rasterizer_indices="3,5",
        direct_rasterizer_candidate_count=32,
        direct_rasterizer_max_records=8,
    )

    def fake_render(viewpoint, gm, render_args, bg):
        row = torch.zeros(96, dtype=torch.float32)
        row[0] = 1.0
        row[1] = float(render_args.debug_preprocess_target_index)
        row[77] = 1.0
        row[78] = 0.0
        row[79] = 0.0
        row[80] = 1.0
        row[81] = 1.0
        return {"cuda_preprocess_debug": row}

    out_path = temp / "direct-evidence.json"
    evidence = TOOL.capture_direct_rasterizer_evidence(
        render_func=fake_render,
        viewpoint=object(),
        gm=FakeEvidenceModel(),
        args=args,
        bg=None,
        render_pkg={"visibility_filter": torch.tensor([True] * 4)},
        cli=cli,
        out_path=out_path,
        run_id="synthetic-range-run",
        population_selection=contract,
    )
    assert args.debug_preprocess_target_index == -1
    assert evidence["schemaVersion"] == (
        "phase3-step114-direct-cuda-rasterizer-evidence-v2"
    )
    assert evidence["selectionPolicy"]["selectedIndices"] == [3, 5]
    assert evidence["selectionPolicy"]["selectedOriginalSourceIndices"] == [3, 5]
    assert evidence["selectionPolicy"]["selectedRasterizerLocalIndices"] == [1, 3]
    assert evidence["validRecordCount"] == 2
    assert [record["rasterizerLocalIndex"] for record in evidence["records"]] == [1, 3]
    assert [record["originalSrcIndex"] for record in evidence["records"]] == [3, 5]
    assert [record["srcIndex"] for record in evidence["records"]] == [3, 5]
    assert evidence["indexLineage"]["rangeStart"] == 2
    assert evidence["indexLineage"]["rangeEndExclusive"] == 6
    persisted = json.loads(out_path.read_text(encoding="utf-8"))
    assert persisted["records"] == evidence["records"]


def main():
    with tempfile.TemporaryDirectory(prefix="step119-impl7-") as temp_dir:
        root = Path(temp_dir)
        test_cli_contract(root)
        test_range_selection()
        checkpoint, gate = test_provenance(root / "provenance-cases")
        assert checkpoint.is_file()
        contract = build_range_contract(gate)
        manifest_root = root / "manifest"
        manifest_root.mkdir()
        test_manifest(manifest_root, contract)
        test_direct_evidence(root, contract)
    print("step119 impl7 contiguous source-range smoke: ok")


if __name__ == "__main__":
    main()
