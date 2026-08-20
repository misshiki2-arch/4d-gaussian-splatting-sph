#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPOSITORY_ROOT / "tools" / "render_named_reference_from_ckpt.py"


def load_render_tool():
    spec = importlib.util.spec_from_file_location("step119_impl2_tool", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = load_render_tool()
import tools.reference_manifest_builder as MANIFEST  # noqa: E402


def make_capture(record_count: int = 6):
    def field(width: int, offset: float):
        values = torch.arange(record_count * width, dtype=torch.float32)
        return values.reshape(record_count, width) + offset

    return (
        2,
        field(3, 1.0),
        field(3, 2.0).reshape(record_count, 1, 3),
        field(12, 3.0).reshape(record_count, 4, 3),
        field(3, 4.0),
        field(4, 5.0),
        field(1, 6.0),
        field(1, 7.0).reshape(record_count),
        field(1, 8.0),
        field(1, 9.0),
        field(1, 10.0),
        {"optimizer": "not-restored-for-reference-render"},
        1.0,
        field(1, 11.0),
        field(1, 12.0),
        field(4, 13.0),
        True,
        torch.arange(9, dtype=torch.float32).reshape(3, 3),
        2,
    )


def write_ready_provenance(
    path: Path,
    checkpoint_path: Path,
    spl4_path: Path,
    *,
    source_count: int = 6,
    selected_count: int = 4,
    overrides=None,
):
    checkpoint_identity = MANIFEST.file_identity(checkpoint_path)
    spl4_identity = MANIFEST.file_identity(spl4_path)
    range_hash = "67" * 32
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
            "startInclusive": 0,
            "endExclusive": selected_count,
            "selectedRecordCount": selected_count,
        },
        "checkpoint": checkpoint_identity,
        "spl4Asset": spl4_identity,
        "spl4Format": {
            "recordCount": source_count,
            "recordStrideBytes": 260,
            "headerSize": 128,
        },
        "expectedSerializedByteCount": selected_count * 260,
        "actualAssetRangeByteCount": selected_count * 260,
        "checkpointRangeSha256": range_hash,
        "assetRangeSha256": range_hash,
    }
    if overrides:
        provenance.update(overrides)
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    return provenance


def assert_population_error(function, expected_fragment: str):
    try:
        function()
    except TOOL.PopulationSelectionError as exc:
        assert expected_fragment in str(exc), str(exc)
    else:
        raise AssertionError(f"expected PopulationSelectionError: {expected_fragment}")


class FakeGaussianModel:
    def __init__(self):
        self.restored = None

    def restore(self, model_args, training_args):
        assert training_args is None
        self.restored = model_args

    @property
    def get_xyz(self):
        return self.restored[1]


def test_cli_contract(temp: Path):
    default_cli = SimpleNamespace(
        population_prefix_count=None,
        population_provenance=None,
        population_spl4=None,
        reuse_legacy_training_report=False,
    )
    assert TOOL.validate_population_selection_cli(default_cli) is None

    for count in (0, -1):
        cli = SimpleNamespace(
            population_prefix_count=count,
            population_provenance=str(temp / "provenance.json"),
            population_spl4=str(temp / "scene.splat4d"),
            reuse_legacy_training_report=False,
        )
        assert_population_error(
            lambda cli=cli: TOOL.validate_population_selection_cli(cli),
            "population-prefix-count-must-be-positive",
        )

    legacy_cli = SimpleNamespace(
        population_prefix_count=4,
        population_provenance=str(temp / "provenance.json"),
        population_spl4=str(temp / "scene.splat4d"),
        reuse_legacy_training_report=True,
    )
    assert_population_error(
        lambda: TOOL.validate_population_selection_cli(legacy_cli),
        "population-selection-incompatible-with-legacy-reuse",
    )


def make_validated_gate(temp: Path):
    checkpoint = temp / "checkpoint.pth"
    spl4 = temp / "scene.splat4d"
    provenance = temp / "provenance.json"
    checkpoint.write_bytes(b"synthetic-checkpoint-identity")
    spl4.write_bytes(b"SPL4" + bytes(128 + 6 * 260 - 4))
    write_ready_provenance(provenance, checkpoint, spl4)
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (checkpoint, spl4, provenance)
    }
    gate = TOOL.validate_population_provenance(
        provenance_path=provenance,
        checkpoint_path=checkpoint,
        spl4_path=spl4,
        requested_prefix_count=4,
    )
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (checkpoint, spl4, provenance)
    }
    assert before == after
    assert all(gate["validationPredicates"].values())
    return checkpoint, spl4, provenance, gate


def test_provenance_fail_closed(temp: Path):
    checkpoint, spl4, provenance, _ = make_validated_gate(temp)

    blocked = temp / "blocked.json"
    write_ready_provenance(
        blocked,
        checkpoint,
        spl4,
        overrides={
            "decision": "blocked",
            "blockedReasons": ["synthetic-block"],
            "exactPopulationMappingReady": False,
        },
    )
    assert_population_error(
        lambda: TOOL.validate_population_provenance(
            provenance_path=blocked,
            checkpoint_path=checkpoint,
            spl4_path=spl4,
            requested_prefix_count=4,
        ),
        "decisionReady",
    )

    selection_mismatch = temp / "selection-mismatch.json"
    write_ready_provenance(selection_mismatch, checkpoint, spl4, selected_count=3)
    assert_population_error(
        lambda: TOOL.validate_population_provenance(
            provenance_path=selection_mismatch,
            checkpoint_path=checkpoint,
            spl4_path=spl4,
            requested_prefix_count=4,
        ),
        "selectionMatchesRequestedPrefix",
    )

    checkpoint_mismatch = temp / "checkpoint-mismatch.json"
    write_ready_provenance(checkpoint_mismatch, checkpoint, spl4)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"-changed")
    assert_population_error(
        lambda: TOOL.validate_population_provenance(
            provenance_path=checkpoint_mismatch,
            checkpoint_path=checkpoint,
            spl4_path=spl4,
            requested_prefix_count=4,
        ),
        "checkpointIdentityMatches",
    )

    checkpoint.write_bytes(b"synthetic-checkpoint-identity")
    spl4_mismatch = temp / "spl4-mismatch.json"
    write_ready_provenance(spl4_mismatch, checkpoint, spl4)
    spl4.write_bytes(spl4.read_bytes() + b"-changed")
    assert_population_error(
        lambda: TOOL.validate_population_provenance(
            provenance_path=spl4_mismatch,
            checkpoint_path=checkpoint,
            spl4_path=spl4,
            requested_prefix_count=4,
        ),
        "spl4IdentityMatches",
    )


def test_selection_and_default_model_path(gate):
    capture = make_capture()
    selected, application = TOOL.select_checkpoint_source_prefix(
        capture,
        selected_count=4,
        expected_source_count=6,
    )
    assert len(selected) == len(capture) == 19
    for capture_index, field_name in TOOL.CHECKPOINT_PREFIX_FIELD_INDICES:
        assert capture[capture_index].shape[0] == 6
        assert selected[capture_index].shape[0] == 4
        assert torch.equal(selected[capture_index], capture[capture_index][:4])
        assert application["perGaussianFieldCounts"][field_name] == {
            "sourceRecordCount": 6,
            "selectedRecordCount": 4,
        }
    assert application["sourceOrderPreserved"] is True
    assert application["selectionAppliedBeforeProductionRasterizer"] is True

    for invalid_count in (0, -1, 7):
        assert_population_error(
            lambda invalid_count=invalid_count: TOOL.select_checkpoint_source_prefix(
                capture,
                selected_count=invalid_count,
                expected_source_count=6,
            ),
            "population-prefix-range-out-of-bounds",
        )

    # Default build path restores the original full capture without selection.
    original_new_model = TOOL.new_gaussian_model
    original_torch_load = TOOL.torch.load
    try:
        full_model = FakeGaussianModel()
        TOOL.new_gaussian_model = lambda args: full_model
        TOOL.torch.load = lambda path, map_location: (capture, 12000)
        result = TOOL.build_gaussian_model(SimpleNamespace(), Path("unused"))
        assert result is full_model
        assert result.restored is capture
        assert result.get_xyz.shape[0] == 6

        selected_model = FakeGaussianModel()
        TOOL.new_gaussian_model = lambda args: selected_model
        result, contract = TOOL.build_population_aligned_gaussian_model(
            SimpleNamespace(), Path("unused"), gate
        )
        assert result.get_xyz.shape[0] == 4
        assert contract["originalSourceRecordCount"] == 6
        assert contract["productionRasterizerRecordCount"] == 4
        assert contract["populationAlignmentReady"] is True
        assert all(contract["predicates"].values())
    finally:
        TOOL.new_gaussian_model = original_new_model
        TOOL.torch.load = original_torch_load
    return contract


def build_manifest_fixture(temp: Path, contract, gaussian_count: int):
    for name in ("render.png", "gt.png", "alpha.png", "depth.png", "meta.json"):
        (temp / name).write_bytes(name.encode("utf-8"))
    args = SimpleNamespace(
        white_background=False,
        sh_degree=2,
        sh_degree_t=2,
        gaussian_dim=4,
        rot_4d=True,
        force_sh_3d=False,
        prefilter_var=-1.0,
    )
    cli = SimpleNamespace(
        which="training-report-train-sample",
        sample_indices="5,10,15,20,25",
        seed=6666,
        iteration=12000,
        limit=1,
    )
    meta = {
        "frame_number": 151,
        "view_id": 13,
        "image_name": "000151_v13",
        "timestamp": 23.2,
        "width": 1280,
        "height": 720,
        "FoVx": 1.0,
        "FoVy": 0.8,
        "fx": 1.0,
        "fy": 1.0,
        "cx": 640.0,
        "cy": 360.0,
        "world_view_transform": [],
        "full_proj_transform": [],
        "transform_matrix": [],
        "camera_center": [],
        "render_context": {"timeDuration": [0.0, 49.9]},
    }
    fake_model = SimpleNamespace(get_xyz=torch.zeros(gaussian_count, 3))
    return MANIFEST.build_cuda_reference_manifest(
        run_id="synthetic-run",
        repo_path=REPOSITORY_ROOT,
        script_path=TOOL_PATH,
        cfg_path=temp / "cfg_args",
        argv=["synthetic"],
        cwd=temp,
        args=args,
        cli=cli,
        ckpt_path=temp / "checkpoint.pth",
        source_path=temp,
        model_path=temp,
        camera=SimpleNamespace(uid=0),
        meta=meta,
        render_paths={
            "render": str(temp / "render.png"),
            "gt": str(temp / "gt.png"),
            "alpha": str(temp / "alpha.png"),
            "depth": str(temp / "depth.png"),
        },
        meta_path=temp / "meta.json",
        manifest_path=temp / "manifest.json",
        gaussian_model=fake_model,
        dataset_identity={"fixture": True},
        path_remaps=[],
        reuse_legacy_training_report=False,
        direct_rasterizer_evidence=None,
        population_selection=contract,
    )


def test_manifest_contract(temp: Path, contract):
    original_module_identity = MANIFEST.module_identity
    try:
        MANIFEST.module_identity = lambda name: {
            "moduleName": name,
            "available": True,
            "fixture": True,
        }
        aligned = build_manifest_fixture(temp, contract, gaussian_count=4)
        assert aligned["schemaVersion"] == MANIFEST.SCHEMA_VERSION
        assert aligned["manifestKind"] == "cuda-reference-render-state"
        assert aligned["renderState"]["gaussianCount"] == 4
        published = aligned["lineage"]["populationSelection"]
        assert published["originalSourceRecordCount"] == 6
        assert published["appliedSelection"]["startInclusive"] == 0
        assert published["appliedSelection"]["endExclusive"] == 4
        assert published["appliedSelection"]["selectedCount"] == 4
        assert published["productionRasterizerRecordCount"] == 4
        assert published["provenance"]["artifact"]["sha256"]
        assert published["populationAlignmentReady"] is True

        default = build_manifest_fixture(temp, None, gaussian_count=6)
        assert default["renderState"]["gaussianCount"] == 6
        assert "populationSelection" not in default["lineage"]
        for key in ("checkpoint", "datasetRoot", "datasetIdentity", "modelPath"):
            assert key in default["lineage"]
        for key in ("camera", "imageSpaceConvention", "artifacts", "renderContext"):
            assert key in default
    finally:
        MANIFEST.module_identity = original_module_identity


def main():
    with tempfile.TemporaryDirectory(prefix="step119-impl2-") as temp_dir:
        root = Path(temp_dir)
        test_cli_contract(root)

        valid_root = root / "valid"
        valid_root.mkdir()
        checkpoint, _, _, gate = make_validated_gate(valid_root)
        contract = test_selection_and_default_model_path(gate)
        # Manifest fixture uses the same verified checkpoint identity path.
        manifest_root = valid_root
        assert checkpoint == manifest_root / "checkpoint.pth"
        test_manifest_contract(manifest_root, contract)

        failure_root = root / "failures"
        failure_root.mkdir()
        test_provenance_fail_closed(failure_root)

    print("step119 impl2 population-aligned reference smoke: ok")


if __name__ == "__main__":
    main()
