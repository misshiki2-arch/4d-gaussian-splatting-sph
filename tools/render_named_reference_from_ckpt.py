#!/usr/bin/env python3
"""Render camera-name-preserving CUDA references from a 4DGS checkpoint.

This tool exists to bridge the gap between training_report()'s legacy
`000_render.png`-style dumps and downstream comparisons that need to know the
source image / frame / time / pose for each rendered image.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scene.dataset_readers import readCamerasFromTransforms
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos
from utils.data_utils import CameraDataset

from tools.reference_manifest_builder import (
    build_cuda_reference_manifest,
    build_run_id,
    file_identity,
    write_manifest,
)


POPULATION_PROVENANCE_SCHEMA_VERSION = (
    "phase3-checkpoint-spl4-population-provenance-v1"
)
POPULATION_SELECTION_SCHEMA_VERSION = (
    "phase3-cuda-reference-population-selection-v1"
)
POPULATION_RANGE_SELECTION_SCHEMA_VERSION = (
    "phase3-cuda-reference-population-selection-v2"
)
POPULATION_PROVENANCE_VERIFICATION_MODE = (
    "checkpoint-existing-spl4-v2-contiguous-source-index-range"
)
POPULATION_LOCAL_TO_ORIGINAL_INDEX_POLICY = (
    "original-source-index=range-start+rasterizer-local-index"
)
CHECKPOINT_PER_GAUSSIAN_FIELD_INDICES = (
    (1, "xyz"),
    (2, "featuresDc"),
    (3, "featuresRest"),
    (4, "scalingXyz"),
    (5, "rotation"),
    (6, "opacity"),
    (7, "maxRadii2d"),
    (8, "xyzGradientAccum"),
    (9, "timeGradientAccum"),
    (10, "denominator"),
    (13, "time"),
    (14, "scalingTime"),
    (15, "rotationRight"),
)
# Public compatibility name used by the existing Impl2 focused smoke.
CHECKPOINT_PREFIX_FIELD_INDICES = CHECKPOINT_PER_GAUSSIAN_FIELD_INDICES


class PopulationSelectionError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render named CUDA reference images with sidecar metadata.")
    ap.add_argument("--model-path", required=True, help="Path to the training output directory.")
    ap.add_argument("--source-path", default=None, help="Optional dataset root override.")
    ap.add_argument("--ckpt", default=None, help="Checkpoint path override. Defaults to best checkpoint under model-path.")
    ap.add_argument("--iteration", type=int, default=12000, help="Iteration label for output grouping.")
    ap.add_argument("--which", choices=["training-report-train-sample", "train", "test"], default="training-report-train-sample")
    ap.add_argument("--sample-indices", default="5,10,15,20,25", help="Indices used when reproducing training_report train samples.")
    ap.add_argument("--seed", type=int, default=6666)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="Optional limit after selection.")
    ap.add_argument(
        "--reuse-legacy-training-report",
        action="store_true",
        help="Reuse existing renders_png/iter_xxxxxx/train/{000..004}_*.png instead of re-rendering."
    )
    ap.add_argument(
        "--direct-rasterizer-evidence",
        action="store_true",
        help="Emit Step114 v2 direct CUDA rasterizer preprocess evidence for representative Gaussians."
    )
    ap.add_argument(
        "--direct-rasterizer-max-records",
        type=int,
        default=8,
        help="Maximum representative Gaussian direct rasterizer evidence records."
    )
    ap.add_argument(
        "--direct-rasterizer-candidate-count",
        type=int,
        default=32,
        help="Visible Gaussian candidate count sampled before representative selection."
    )
    ap.add_argument(
        "--direct-rasterizer-indices",
        default="",
        help="Optional comma-separated source indices for direct rasterizer evidence."
    )
    ap.add_argument(
        "--population-prefix-count",
        type=int,
        default=None,
        help="Opt in to checkpoint source-prefix population selection before CUDA rendering."
    )
    ap.add_argument(
        "--population-range-start",
        type=int,
        default=None,
        help="Opt in to a contiguous checkpoint source range at this inclusive original index."
    )
    ap.add_argument(
        "--population-range-count",
        type=int,
        default=None,
        help="Number of contiguous checkpoint source records selected by --population-range-start."
    )
    ap.add_argument(
        "--population-provenance",
        default=None,
        help="Accepted checkpoint-to-SPL4 population provenance JSON for prefix or range rendering."
    )
    ap.add_argument(
        "--population-spl4",
        default=None,
        help="Existing SPL4-v2 asset whose identity is accepted by the provenance artifact."
    )
    return ap.parse_args()


def parse_namespace_repr(text: str) -> Dict[str, object]:
    expr = ast.parse(text.strip(), mode="eval").body
    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name) or expr.func.id != "Namespace":
        raise ValueError("cfg_args is not a Namespace(...) repr")
    out: Dict[str, object] = {}
    for kw in expr.keywords:
        out[kw.arg] = ast.literal_eval(kw.value)
    return out


def remap_work_path(path_value: Optional[str]) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    if not isinstance(path_value, str) or not path_value:
        return path_value, None
    if os.path.exists(path_value):
        return path_value, None
    if path_value.startswith("/home/demo/"):
        remapped = path_value.replace("/home/demo/", "/home/demo/work/", 1)
        if os.path.exists(remapped):
            return remapped, {
                "originalPath": path_value,
                "remappedPath": remapped,
                "reason": "workspace-relocated-from-home-demo"
            }
    return path_value, None


def load_cfg_args(model_path: Path) -> Tuple[SimpleNamespace, List[Dict[str, str]]]:
    cfg_path = model_path / "cfg_args"
    cfg = parse_namespace_repr(cfg_path.read_text())
    path_remaps: List[Dict[str, str]] = []
    for key in ("source_path", "model_path", "loaded_pth"):
        remapped, info = remap_work_path(cfg.get(key))
        cfg[key] = remapped
        if info:
            info["key"] = key
            path_remaps.append(info)
    cfg.setdefault("images", "images")
    cfg.setdefault("eval", False)
    cfg.setdefault("extension", "")
    cfg.setdefault("frame_ratio", 1)
    cfg.setdefault("dataloader", True)
    cfg.setdefault("data_device", "cuda")
    cfg.setdefault("white_background", False)
    cfg.setdefault("debug", False)
    cfg.setdefault("compute_cov3D_python", False)
    cfg.setdefault("convert_SHs_python", False)
    cfg.setdefault("env_map_res", 0)
    cfg.setdefault("gaussian_dim", 4)
    cfg.setdefault("rot_4d", True)
    cfg.setdefault("force_sh_3d", False)
    cfg.setdefault("sh_degree_t", 0)
    cfg.setdefault("prefilter_var", -1.0)
    return SimpleNamespace(**cfg), path_remaps


def resolve_checkpoint(args: SimpleNamespace, cli_ckpt: Optional[str], model_path: Path, iteration: int) -> Path:
    candidates: List[Path] = []
    if cli_ckpt:
        candidates.append(Path(cli_ckpt))
    loaded_pth = getattr(args, "loaded_pth", None)
    if isinstance(loaded_pth, str) and loaded_pth:
        candidates.append(Path(loaded_pth))
    candidates.append(model_path / f"chkpnt_best_{iteration}.pth")
    candidates.append(model_path / f"chkpnt{iteration}.pth")
    candidates.append(model_path / "chkpnt_best.pth")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found for model_path={model_path}")


def read_frame_lists(source_path: Path) -> Tuple[List[dict], List[dict], Dict[str, dict]]:
    train_frames = json.loads((source_path / "transforms_train.json").read_text())["frames"]
    test_frames = json.loads((source_path / "transforms_test.json").read_text())["frames"]
    frame_lookup: Dict[str, dict] = {}
    for frame in train_frames + test_frames:
        image_name = Path(frame["file_path"]).stem
        frame_lookup[image_name] = frame
    return train_frames, test_frames, frame_lookup


def infer_time_duration(train_frames: Sequence[dict], test_frames: Sequence[dict]) -> Optional[List[float]]:
    times: List[float] = []
    for frame in list(train_frames) + list(test_frames):
        if "time" not in frame:
            continue
        try:
            times.append(float(frame["time"]))
        except (TypeError, ValueError):
            continue
    if not times:
        return None
    return [min(times), max(times)]


def build_camera_infos(args: SimpleNamespace, which: str) -> List:
    train_infos = readCamerasFromTransforms(
        args.source_path,
        "transforms_train.json",
        args.white_background,
        extension=args.extension,
        time_duration=getattr(args, "time_duration", None),
        frame_ratio=int(args.frame_ratio),
        dataloader=bool(args.dataloader)
    )
    test_infos = readCamerasFromTransforms(
        args.source_path,
        "transforms_test.json",
        args.white_background,
        extension=args.extension,
        time_duration=getattr(args, "time_duration", None),
        frame_ratio=int(args.frame_ratio),
        dataloader=bool(args.dataloader)
    )
    if which == "train":
        return train_infos
    if which == "test":
        return test_infos
    combined = list(train_infos)
    if not bool(args.eval):
        combined.extend(test_infos)
    return combined


def select_camera_infos(camera_infos: Sequence, which: str, sample_indices: Sequence[int], seed: int) -> List:
    selected = list(camera_infos)
    if which == "training-report-train-sample":
        random.seed(seed)
        random.shuffle(selected)
        selected = [selected[idx] for idx in sample_indices]
    return selected


def parse_sample_indices(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_optional_indices(value: str) -> List[int]:
    if not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def save_rgb_tensor_chw01(t: torch.Tensor, path: Path) -> None:
    arr = (t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_single_channel_tensor_chw01_as_rgb(t: torch.Tensor, path: Path) -> None:
    chw = t.detach().clamp(0, 1)
    if chw.shape[0] == 1:
        chw = chw.repeat(3, 1, 1)
    save_rgb_tensor_chw01(chw, path)


def infer_frame_and_view(image_name: str) -> Tuple[Optional[int], Optional[int]]:
    if "_v" not in image_name:
        return None, None
    frame_part, view_part = image_name.split("_v", 1)
    try:
        return int(frame_part), int(view_part)
    except ValueError:
        return None, None


def focal_to_fov(size: int, focal: float) -> Optional[float]:
    if size <= 0 or focal <= 0:
        return None
    return 2.0 * math.atan(size / (2.0 * focal))


def validate_population_selection_cli(cli: argparse.Namespace) -> Optional[Dict[str, object]]:
    prefix_count = getattr(cli, "population_prefix_count", None)
    range_start = getattr(cli, "population_range_start", None)
    range_count = getattr(cli, "population_range_count", None)
    provenance_path = getattr(cli, "population_provenance", None)
    spl4_path = getattr(cli, "population_spl4", None)
    configured = any(
        value is not None
        for value in (
            prefix_count,
            range_start,
            range_count,
            provenance_path,
            spl4_path,
        )
    )
    if not configured:
        return None

    reasons = []
    prefix_configured = prefix_count is not None
    range_configured = range_start is not None or range_count is not None
    if prefix_configured and range_configured:
        reasons.append("population-prefix-and-range-mutually-exclusive")
    if prefix_configured:
        if isinstance(prefix_count, bool) or prefix_count <= 0:
            reasons.append("population-prefix-count-must-be-positive")
        selection_mode = "prefix"
        start_inclusive = 0
        selected_count = prefix_count
    elif range_configured:
        if range_start is None:
            reasons.append("population-range-start-required")
        elif isinstance(range_start, bool) or range_start < 0:
            reasons.append("population-range-start-must-be-non-negative")
        if range_count is None:
            reasons.append("population-range-count-required")
        elif isinstance(range_count, bool) or range_count <= 0:
            reasons.append("population-range-count-must-be-positive")
        selection_mode = "range"
        start_inclusive = range_start
        selected_count = range_count
    else:
        reasons.append("population-selection-count-required")
        selection_mode = None
        start_inclusive = None
        selected_count = None
    if not provenance_path:
        reasons.append("population-provenance-required")
    if not spl4_path:
        reasons.append("population-spl4-required")
    if bool(getattr(cli, "reuse_legacy_training_report", False)):
        reasons.append("population-selection-incompatible-with-legacy-reuse")
    if reasons:
        raise PopulationSelectionError(
            "population-selection-cli-invalid:" + ",".join(reasons)
        )
    end_exclusive = int(start_inclusive) + int(selected_count)
    return {
        "mode": selection_mode,
        "startInclusive": int(start_inclusive),
        "selectedCount": int(selected_count),
        "endExclusive": end_exclusive,
        "prefixCount": int(prefix_count) if prefix_configured else None,
        "provenancePath": Path(provenance_path),
        "spl4Path": Path(spl4_path),
    }


def population_identity_matches(claimed: object, actual: Dict[str, object]) -> bool:
    if not isinstance(claimed, dict):
        return False
    claimed_path = claimed.get("absolutePath")
    return bool(
        isinstance(claimed_path, str)
        and os.path.abspath(claimed_path) == actual.get("absolutePath")
        and claimed.get("sizeBytes") == actual.get("sizeBytes")
        and isinstance(claimed.get("sha256"), str)
        and claimed.get("sha256") == actual.get("sha256")
    )


def validate_population_provenance(
    *,
    provenance_path: Path,
    checkpoint_path: Path,
    spl4_path: Path,
    requested_start: Optional[int] = None,
    requested_count: Optional[int] = None,
    requested_prefix_count: Optional[int] = None,
) -> Dict[str, object]:
    if requested_prefix_count is not None:
        if requested_start is not None or requested_count is not None:
            raise PopulationSelectionError(
                "population-provenance-request-ambiguous"
            )
        requested_start = 0
        requested_count = requested_prefix_count
    if (
        isinstance(requested_start, bool)
        or not isinstance(requested_start, int)
        or requested_start < 0
    ):
        raise PopulationSelectionError(
            "population-provenance-request-start-invalid"
        )
    if (
        isinstance(requested_count, bool)
        or not isinstance(requested_count, int)
        or requested_count <= 0
    ):
        raise PopulationSelectionError(
            "population-provenance-request-count-invalid"
        )
    requested_end = requested_start + requested_count
    if not provenance_path.is_file():
        raise PopulationSelectionError("population-provenance-artifact-missing")
    try:
        provenance_bytes = provenance_path.read_bytes()
        provenance = json.loads(provenance_bytes.decode("utf-8"))
    except Exception as exc:
        raise PopulationSelectionError(
            f"population-provenance-artifact-invalid:{type(exc).__name__}:{exc}"
        ) from exc
    if not isinstance(provenance, dict):
        raise PopulationSelectionError("population-provenance-root-must-be-object")

    provenance_artifact = {
        "absolutePath": str(provenance_path.resolve()),
        "exists": True,
        "sizeBytes": len(provenance_bytes),
        "sha256": hashlib.sha256(provenance_bytes).hexdigest(),
    }
    current_checkpoint_identity = file_identity(checkpoint_path)
    current_spl4_identity = file_identity(spl4_path)
    selection = provenance.get("selection")
    spl4_format = provenance.get("spl4Format")
    checkpoint_count = provenance.get("checkpointSourceGaussianCount")
    asset_count = provenance.get("assetRecordCount")
    range_hash = provenance.get("checkpointRangeSha256")
    asset_range_hash = provenance.get("assetRangeSha256")
    record_stride = (
        spl4_format.get("recordStrideBytes")
        if isinstance(spl4_format, dict)
        else None
    )
    header_size = (
        spl4_format.get("headerSize")
        if isinstance(spl4_format, dict)
        else None
    )
    serialized_byte_range = None
    if (
        isinstance(record_stride, int)
        and not isinstance(record_stride, bool)
        and record_stride > 0
        and isinstance(header_size, int)
        and not isinstance(header_size, bool)
        and header_size > 0
    ):
        byte_start = header_size + requested_start * record_stride
        byte_count = requested_count * record_stride
        serialized_byte_range = {
            "payloadByteOffset": byte_start,
            "serializedByteCount": byte_count,
            "byteEndExclusive": byte_start + byte_count,
            "recordStrideBytes": record_stride,
            "headerSizeBytes": header_size,
        }

    selection_matches_requested_range = bool(
        isinstance(selection, dict)
        and selection.get("policy") == "contiguous-source-index-range"
        and selection.get("startInclusive") == requested_start
        and selection.get("endExclusive") == requested_end
        and selection.get("selectedRecordCount") == requested_count
    )

    predicates = {
        "schemaAccepted": provenance.get("schemaVersion")
        == POPULATION_PROVENANCE_SCHEMA_VERSION,
        "verificationModeAccepted": provenance.get("verificationMode")
        == POPULATION_PROVENANCE_VERIFICATION_MODE,
        "decisionReady": provenance.get("decision") == "ready",
        "blockedReasonsEmpty": provenance.get("blockedReasons") == [],
        "exactPopulationMappingReady": provenance.get("exactPopulationMappingReady")
        is True,
        "rangeHashesMatch": provenance.get("rangeHashesMatch") is True,
        "recordCountsMatch": (
            isinstance(checkpoint_count, int)
            and not isinstance(checkpoint_count, bool)
            and checkpoint_count > 0
            and checkpoint_count == asset_count
            and isinstance(spl4_format, dict)
            and spl4_format.get("recordCount") == asset_count
            and provenance.get("recordCountsMatch") is True
        ),
        "selectionMatchesRequestedRange": selection_matches_requested_range,
        "checkpointIdentityMatches": population_identity_matches(
            provenance.get("checkpoint"), current_checkpoint_identity
        ),
        "spl4IdentityMatches": population_identity_matches(
            provenance.get("spl4Asset"), current_spl4_identity
        ),
        "rangeHashIdentityValid": (
            isinstance(range_hash, str)
            and len(range_hash) == 64
            and all(character in "0123456789abcdef" for character in range_hash)
            and range_hash == asset_range_hash
        ),
        "serializedRangeLengthValid": (
            isinstance(spl4_format, dict)
            and isinstance(record_stride, int)
            and not isinstance(record_stride, bool)
            and record_stride > 0
            and provenance.get("expectedSerializedByteCount")
            == requested_count * record_stride
            and provenance.get("actualAssetRangeByteCount")
            == provenance.get("expectedSerializedByteCount")
            and provenance.get("assetPayloadLengthValid") is True
            and provenance.get("headerCompatible") is True
            and provenance.get("rangeInBounds") is True
        ),
        "serializedByteRangeValid": (
            isinstance(serialized_byte_range, dict)
            and serialized_byte_range["serializedByteCount"]
            == provenance.get("expectedSerializedByteCount")
            and serialized_byte_range["byteEndExclusive"]
            <= current_spl4_identity.get("sizeBytes", -1)
        ),
    }
    if requested_prefix_count is not None:
        predicates["selectionMatchesRequestedPrefix"] = (
            selection_matches_requested_range
        )
    blocked_reasons = [
        name for name, ready in predicates.items() if ready is not True
    ]
    if blocked_reasons:
        raise PopulationSelectionError(
            "population-provenance-validation-failed:" + ",".join(blocked_reasons)
        )
    return {
        "artifact": provenance_artifact,
        "schemaVersion": provenance.get("schemaVersion"),
        "verificationMode": provenance.get("verificationMode"),
        "decision": provenance.get("decision"),
        "blockedReasons": list(provenance.get("blockedReasons", [])),
        "checkpointSourceRecordCount": checkpoint_count,
        "assetRecordCount": asset_count,
        "selection": dict(selection),
        "requestedSelection": {
            "policy": "contiguous-source-index-range",
            "startInclusive": requested_start,
            "selectedRecordCount": requested_count,
            "endExclusive": requested_end,
        },
        "serializedByteRange": serialized_byte_range,
        "checkpoint": dict(provenance["checkpoint"]),
        "spl4Asset": dict(provenance["spl4Asset"]),
        "checkpointRangeSha256": range_hash,
        "assetRangeSha256": asset_range_hash,
        "exactPopulationMappingReady": True,
        "rangeHashesMatch": True,
        "currentCheckpointIdentity": current_checkpoint_identity,
        "currentSpl4Identity": current_spl4_identity,
        "validationPredicates": predicates,
    }


def build_meta(camera, frame_record: Optional[dict], render_paths: Dict[str, str], context: Dict[str, object]) -> Dict[str, object]:
    frame_number, view_id = infer_frame_and_view(camera.image_name)
    fx = float(camera.fl_x)
    fy = float(camera.fl_y)
    width = int(camera.image_width)
    height = int(camera.image_height)
    fovx = float(camera.FoVx) if getattr(camera, "FoVx", None) is not None else None
    fovy = float(camera.FoVy) if getattr(camera, "FoVy", None) is not None else None
    if fovx is None or fovx <= 0:
        fovx = focal_to_fov(width, fx)
    if fovy is None or fovy <= 0:
        fovy = focal_to_fov(height, fy)
    return {
        "image_name": camera.image_name,
        "source_image_path": camera.image_path,
        "frame_number": frame_number,
        "view_id": view_id,
        "timestamp": float(camera.timestamp),
        "width": width,
        "height": height,
        "FoVx": fovx,
        "FoVy": fovy,
        "fx": fx,
        "fy": fy,
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        "world_view_transform": camera.world_view_transform.detach().cpu().tolist(),
        "full_proj_transform": camera.full_proj_transform.detach().cpu().tolist(),
        "camera_center": camera.camera_center.detach().cpu().tolist(),
        "transform_matrix": frame_record.get("transform_matrix") if frame_record else None,
        "frame_record": frame_record,
        "render_outputs": render_paths,
        "render_context": context
    }


def new_gaussian_model(args: SimpleNamespace) -> GaussianModel:
    return GaussianModel(
        sh_degree=int(args.sh_degree),
        gaussian_dim=int(args.gaussian_dim),
        time_duration=getattr(args, "time_duration", None),
        rot_4d=bool(args.rot_4d),
        force_sh_3d=bool(getattr(args, "force_sh_3d", False)),
        sh_degree_t=int(getattr(args, "sh_degree_t", 0)),
        prefilter_var=float(getattr(args, "prefilter_var", -1.0)),
    )


def select_checkpoint_source_range(
    model_args: Sequence[object],
    *,
    start_inclusive: int,
    selected_count: int,
    expected_source_count: int,
) -> Tuple[Tuple[object, ...], Dict[str, object]]:
    if not isinstance(model_args, (tuple, list)) or len(model_args) != 19:
        raise PopulationSelectionError(
            "population-checkpoint-capture-contract-invalid"
        )
    xyz = model_args[1]
    if not torch.is_tensor(xyz) or xyz.ndim < 1:
        raise PopulationSelectionError("population-checkpoint-xyz-invalid")
    original_count = int(xyz.shape[0])
    if original_count != expected_source_count:
        raise PopulationSelectionError(
            "population-checkpoint-source-count-mismatch:"
            f"{original_count}!={expected_source_count}"
        )
    if (
        isinstance(start_inclusive, bool)
        or not isinstance(start_inclusive, int)
        or start_inclusive < 0
        or isinstance(selected_count, bool)
        or not isinstance(selected_count, int)
        or selected_count <= 0
        or start_inclusive + selected_count > original_count
    ):
        raise PopulationSelectionError(
            "population-source-range-out-of-bounds:"
            f"start={start_inclusive},count={selected_count},"
            f"sourceCount={original_count}"
        )
    end_exclusive = start_inclusive + selected_count

    selected_args = list(model_args)
    field_counts = {}
    for capture_index, field_name in CHECKPOINT_PER_GAUSSIAN_FIELD_INDICES:
        value = model_args[capture_index]
        if not torch.is_tensor(value) or value.ndim < 1:
            raise PopulationSelectionError(
                f"population-checkpoint-field-invalid:{field_name}"
            )
        field_source_count = int(value.shape[0])
        if field_source_count != original_count:
            raise PopulationSelectionError(
                "population-checkpoint-field-count-mismatch:"
                f"{field_name}:{field_source_count}!={original_count}"
            )
        selected_value = value[start_inclusive:end_exclusive].contiguous()
        if int(selected_value.shape[0]) != selected_count:
            raise PopulationSelectionError(
                f"population-checkpoint-field-selection-failed:{field_name}"
            )
        selected_args[capture_index] = selected_value
        field_counts[field_name] = {
            "sourceRecordCount": field_source_count,
            "selectedRecordCount": int(selected_value.shape[0]),
        }

    return tuple(selected_args), {
        "originalSourceRecordCount": original_count,
        "selectedRecordCount": selected_count,
        "startInclusive": start_inclusive,
        "endExclusive": end_exclusive,
        "sourceOrderPreserved": True,
        "selectionAppliedBeforeGaussianModelRestore": True,
        "selectionAppliedBeforeProductionRasterizer": True,
        "rasterizerLocalIndexEqualsCheckpointSourceIndex": start_inclusive == 0,
        "rasterizerLocalIndexStart": 0,
        "rasterizerLocalIndexEndExclusive": selected_count,
        "originalSourceIndexStart": start_inclusive,
        "originalSourceIndexEndExclusive": end_exclusive,
        "localToOriginalIndexMappingPolicy": (
            POPULATION_LOCAL_TO_ORIGINAL_INDEX_POLICY
        ),
        "perGaussianFieldCounts": field_counts,
        "selectedPerGaussianFieldCount": len(field_counts),
        "allPerGaussianFieldsSelected": all(
            counts["selectedRecordCount"] == selected_count
            for counts in field_counts.values()
        ) and len(field_counts) == len(CHECKPOINT_PER_GAUSSIAN_FIELD_INDICES),
    }


def select_checkpoint_source_prefix(
    model_args: Sequence[object],
    *,
    selected_count: int,
    expected_source_count: int,
) -> Tuple[Tuple[object, ...], Dict[str, object]]:
    if (
        isinstance(selected_count, bool)
        or not isinstance(selected_count, int)
        or selected_count <= 0
        or selected_count > expected_source_count
    ):
        raise PopulationSelectionError(
            "population-prefix-range-out-of-bounds:"
            f"count={selected_count},sourceCount={expected_source_count}"
        )
    return select_checkpoint_source_range(
        model_args,
        start_inclusive=0,
        selected_count=selected_count,
        expected_source_count=expected_source_count,
    )


def build_population_selection_contract(
    provenance_gate: Dict[str, object],
    application: Dict[str, object],
    production_rasterizer_record_count: int,
    *,
    selection_mode: str = "prefix",
) -> Dict[str, object]:
    if selection_mode not in ("prefix", "range"):
        raise PopulationSelectionError(
            f"population-selection-mode-invalid:{selection_mode}"
        )
    requested_start = int(provenance_gate["selection"]["startInclusive"])
    requested_count = int(provenance_gate["selection"]["selectedRecordCount"])
    requested_end = int(provenance_gate["selection"]["endExclusive"])
    original_count = int(provenance_gate["checkpointSourceRecordCount"])
    predicates = {
        "provenanceAccepted": all(
            provenance_gate["validationPredicates"].values()
        ),
        "requestedSelectionMatchesProvenance": (
            application.get("startInclusive") == requested_start
            and application.get("endExclusive") == requested_end
            and application.get("selectedRecordCount") == requested_count
        ),
        "checkpointSourceCountMatchesProvenance": application.get(
            "originalSourceRecordCount"
        ) == original_count,
        "allPerGaussianFieldsSelected": application.get(
            "allPerGaussianFieldsSelected"
        ) is True,
        "sourceOrderPreserved": application.get("sourceOrderPreserved") is True,
        "selectionAppliedBeforeProductionRasterizer": application.get(
            "selectionAppliedBeforeProductionRasterizer"
        ) is True,
        "productionRasterizerCountMatchesSelection": (
            production_rasterizer_record_count == requested_count
        ),
        "rasterizerLocalToOriginalIndexMappingValid": (
            application.get("rasterizerLocalIndexStart") == 0
            and application.get("rasterizerLocalIndexEndExclusive")
            == requested_count
            and application.get("originalSourceIndexStart") == requested_start
            and application.get("originalSourceIndexEndExclusive")
            == requested_end
            and application.get("localToOriginalIndexMappingPolicy")
            == POPULATION_LOCAL_TO_ORIGINAL_INDEX_POLICY
        ),
    }
    if selection_mode == "prefix":
        predicates["rasterizerLocalIndexMatchesCheckpointSourceIndex"] = (
            application.get("rasterizerLocalIndexEqualsCheckpointSourceIndex")
            is True
        )
    blocked_reasons = [
        name for name, ready in predicates.items() if ready is not True
    ]
    if blocked_reasons:
        raise PopulationSelectionError(
            "population-selection-application-failed:" + ",".join(blocked_reasons)
        )
    selection_policy = (
        "contiguous-source-index-prefix"
        if selection_mode == "prefix"
        else "contiguous-source-index-range"
    )
    return {
        "schemaVersion": (
            POPULATION_SELECTION_SCHEMA_VERSION
            if selection_mode == "prefix"
            else POPULATION_RANGE_SELECTION_SCHEMA_VERSION
        ),
        "policy": (
            "checkpoint-source-prefix-before-production-rasterizer"
            if selection_mode == "prefix"
            else "checkpoint-source-contiguous-range-before-production-rasterizer"
        ),
        "selectionMode": selection_mode,
        "sourceIndexSpace": "checkpoint-capture-and-spl4-source-record-index",
        "rasterizerIndexSpace": "selected-population-local-index",
        "originalSourceRecordCount": original_count,
        "requestedSelection": {
            "policy": selection_policy,
            "startInclusive": requested_start,
            "endExclusive": requested_end,
            "selectedCount": requested_count,
        },
        "appliedSelection": {
            "policy": selection_policy,
            "startInclusive": application["startInclusive"],
            "endExclusive": application["endExclusive"],
            "selectedCount": application["selectedRecordCount"],
            "sourceOrderPreserved": application["sourceOrderPreserved"],
            "appliedBeforeGaussianModelRestore": application[
                "selectionAppliedBeforeGaussianModelRestore"
            ],
            "appliedBeforeProductionRasterizer": application[
                "selectionAppliedBeforeProductionRasterizer"
            ],
            "rasterizerLocalIndexEqualsCheckpointSourceIndex": application[
                "rasterizerLocalIndexEqualsCheckpointSourceIndex"
            ],
            "localToOriginalIndexMappingPolicy": application[
                "localToOriginalIndexMappingPolicy"
            ],
            "rasterizerLocalIndexStart": application[
                "rasterizerLocalIndexStart"
            ],
            "rasterizerLocalIndexEndExclusive": application[
                "rasterizerLocalIndexEndExclusive"
            ],
            "originalSourceIndexStart": application["originalSourceIndexStart"],
            "originalSourceIndexEndExclusive": application[
                "originalSourceIndexEndExclusive"
            ],
            "selectedPerGaussianFieldCount": application[
                "selectedPerGaussianFieldCount"
            ],
            "perGaussianFieldCounts": application["perGaussianFieldCounts"],
        },
        "indexLineage": {
            "rasterizerIndexSpace": "selected-population-local-index",
            "originalIndexSpace": (
                "checkpoint-capture-and-spl4-source-record-index"
            ),
            "mappingPolicy": POPULATION_LOCAL_TO_ORIGINAL_INDEX_POLICY,
            "rangeStart": requested_start,
            "rangeCount": requested_count,
            "rangeEndExclusive": requested_end,
            "sourceOrderPreserved": application["sourceOrderPreserved"],
        },
        "productionRasterizerRecordCount": production_rasterizer_record_count,
        "provenance": {
            "artifact": provenance_gate["artifact"],
            "schemaVersion": provenance_gate["schemaVersion"],
            "verificationMode": provenance_gate["verificationMode"],
            "decision": provenance_gate["decision"],
            "blockedReasons": provenance_gate["blockedReasons"],
            "selection": provenance_gate["selection"],
            "requestedSelection": provenance_gate["requestedSelection"],
            "serializedByteRange": provenance_gate["serializedByteRange"],
            "checkpoint": provenance_gate["checkpoint"],
            "spl4Asset": provenance_gate["spl4Asset"],
            "checkpointRangeSha256": provenance_gate["checkpointRangeSha256"],
            "assetRangeSha256": provenance_gate["assetRangeSha256"],
            "exactPopulationMappingReady": provenance_gate[
                "exactPopulationMappingReady"
            ],
            "rangeHashesMatch": provenance_gate["rangeHashesMatch"],
        },
        "predicates": predicates,
        "blockedReasons": [],
        "populationAlignmentReady": True,
        "currentCheckpointIdentity": provenance_gate["currentCheckpointIdentity"],
        "currentSpl4Identity": provenance_gate["currentSpl4Identity"],
    }


def build_gaussian_model(args: SimpleNamespace, ckpt_path: Path) -> GaussianModel:
    gm = new_gaussian_model(args)
    model_args, _ = torch.load(ckpt_path, map_location="cuda")
    gm.restore(model_args, training_args=None)
    return gm


def build_population_aligned_gaussian_model(
    args: SimpleNamespace,
    ckpt_path: Path,
    provenance_gate: Dict[str, object],
    *,
    selection_mode: str = "prefix",
) -> Tuple[GaussianModel, Dict[str, object]]:
    gm = new_gaussian_model(args)
    model_args, _ = torch.load(ckpt_path, map_location="cuda")
    selected_args, application = select_checkpoint_source_range(
        model_args,
        start_inclusive=int(provenance_gate["selection"]["startInclusive"]),
        selected_count=int(provenance_gate["selection"]["selectedRecordCount"]),
        expected_source_count=int(provenance_gate["checkpointSourceRecordCount"]),
    )
    gm.restore(selected_args, training_args=None)
    production_count = int(gm.get_xyz.shape[0])
    contract = build_population_selection_contract(
        provenance_gate,
        application,
        production_rasterizer_record_count=production_count,
        selection_mode=selection_mode,
    )
    return gm, contract


def finite_float(value) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def finite_list(values, count: int) -> Optional[List[float]]:
    if values is None:
        return None
    out: List[float] = []
    for item in list(values)[:count]:
        v = finite_float(item)
        if v is None:
            return None
        out.append(v)
    return out if len(out) == count else None


def decode_preprocess_debug_row(row: List[float]) -> Dict[str, object]:
    def vec(start: int, count: int) -> Optional[List[float]]:
        return finite_list(row[start:start + count], count)

    active = len(row) >= 96 and finite_float(row[0]) == 1.0
    radius = finite_float(row[77]) if len(row) > 77 else None
    rect_min = vec(78, 2)
    rect_max = vec(80, 2)
    tiles_touched = None
    if rect_min and rect_max:
        tiles_touched = max(0.0, rect_max[0] - rect_min[0]) * max(0.0, rect_max[1] - rect_min[1])
    valid = bool(active and radius is not None and radius > 0 and tiles_touched is not None and tiles_touched > 0)
    return {
        "active": active,
        "srcIndex": int(row[1]) if active and finite_float(row[1]) is not None else None,
        "worldPositionInput": vec(2, 3),
        "worldPositionAfterTemporal": vec(5, 3),
        "opacityBeforeTemporal": finite_float(row[8]) if len(row) > 8 else None,
        "opacityAfterTemporal": finite_float(row[9]) if len(row) > 9 else None,
        "scale": vec(10, 3),
        "scaleT": finite_float(row[13]) if len(row) > 13 else None,
        "rotation": vec(14, 4),
        "rotationR": vec(18, 4),
        "gaussianTime": finite_float(row[22]) if len(row) > 22 else None,
        "timestamp": finite_float(row[23]) if len(row) > 23 else None,
        "timeDelta": finite_float(row[24]) if len(row) > 24 else None,
        "timeMask": finite_float(row[25]) if len(row) > 25 else None,
        "covariance3D": vec(26, 6),
        "cameraSpacePositionUnclamped": vec(32, 3),
        "cameraSpacePositionClamped": vec(35, 3),
        "projectionJacobianRaw": vec(38, 9),
        "viewLinearMatrixRaw": vec(47, 9),
        "ndc": vec(56, 3),
        "covarianceDeterminant": finite_float(row[73]) if len(row) > 73 else None,
        "conic": vec(74, 3),
        "radius": radius,
        "rectMin": rect_min,
        "rectMax": rect_max,
        "screenCenter": vec(82, 2),
        "depth": finite_float(row[84]) if len(row) > 84 else None,
        "clip": vec(85, 4),
        "debugPixel": vec(89, 2),
        "samplePower": finite_float(row[93]) if len(row) > 93 else None,
        "sampleRawAlpha": finite_float(row[94]) if len(row) > 94 else None,
        "sampleAlpha": finite_float(row[95]) if len(row) > 95 else None,
        "valid": valid,
        "culled": not valid,
        "tilesTouched": tiles_touched,
    }


def tensor_to_cpu_list(tensor) -> List[float]:
    if tensor is None or not hasattr(tensor, "detach"):
        return []
    return [float(x) for x in tensor.detach().cpu().flatten().tolist()]


def build_direct_evidence_index_lineage(
    gm: GaussianModel,
    population_selection: Optional[Dict[str, object]],
) -> Dict[str, object]:
    selected_count = int(gm.get_xyz.shape[0])
    if population_selection is None:
        return {
            "selectionMode": "full-population",
            "rangeStart": 0,
            "rangeCount": selected_count,
            "rangeEndExclusive": selected_count,
            "productionRasterizerRecordCount": selected_count,
            "sourceOrderPreserved": True,
            "mappingPolicy": POPULATION_LOCAL_TO_ORIGINAL_INDEX_POLICY,
            "rasterizerIndexSpace": "full-population-local-index",
            "originalIndexSpace": (
                "checkpoint-capture-and-spl4-source-record-index"
            ),
            "populationAlignmentReady": True,
        }
    if population_selection.get("populationAlignmentReady") is not True:
        raise PopulationSelectionError(
            "direct-evidence-population-selection-not-ready"
        )
    contract_lineage = population_selection.get("indexLineage")
    if not isinstance(contract_lineage, dict):
        raise PopulationSelectionError(
            "direct-evidence-index-lineage-missing"
        )
    range_start = contract_lineage.get("rangeStart")
    range_count = contract_lineage.get("rangeCount")
    range_end = contract_lineage.get("rangeEndExclusive")
    if (
        not isinstance(range_start, int)
        or isinstance(range_start, bool)
        or range_start < 0
        or range_count != selected_count
        or range_end != range_start + selected_count
        or population_selection.get("productionRasterizerRecordCount")
        != selected_count
        or contract_lineage.get("mappingPolicy")
        != POPULATION_LOCAL_TO_ORIGINAL_INDEX_POLICY
        or contract_lineage.get("sourceOrderPreserved") is not True
    ):
        raise PopulationSelectionError(
            "direct-evidence-index-lineage-invalid"
        )
    return {
        "selectionMode": population_selection.get("selectionMode"),
        "rangeStart": range_start,
        "rangeCount": selected_count,
        "rangeEndExclusive": range_end,
        "productionRasterizerRecordCount": selected_count,
        "sourceOrderPreserved": True,
        "mappingPolicy": POPULATION_LOCAL_TO_ORIGINAL_INDEX_POLICY,
        "rasterizerIndexSpace": "selected-population-local-index",
        "originalIndexSpace": (
            "checkpoint-capture-and-spl4-source-record-index"
        ),
        "populationAlignmentReady": True,
    }


def select_direct_rasterizer_candidates(
    gm: GaussianModel,
    render_pkg: Dict[str, object],
    cli: argparse.Namespace,
    population_selection: Optional[Dict[str, object]] = None,
) -> List[int]:
    lineage = build_direct_evidence_index_lineage(gm, population_selection)
    explicit = parse_optional_indices(cli.direct_rasterizer_indices)
    if explicit:
        range_start = int(lineage["rangeStart"])
        range_end = int(lineage["rangeEndExclusive"])
        local_indices = []
        for original_index in explicit[: max(1, cli.direct_rasterizer_max_records)]:
            if original_index < range_start or original_index >= range_end:
                raise PopulationSelectionError(
                    "direct-rasterizer-original-index-out-of-range:"
                    f"index={original_index},range=[{range_start},{range_end})"
                )
            local_indices.append(original_index - range_start)
        return local_indices
    visibility = render_pkg.get("visibility_filter")
    if visibility is None or not hasattr(visibility, "nonzero"):
        return []
    visible = visibility.detach().nonzero(as_tuple=False).flatten()
    if visible.numel() == 0:
        return []
    max_candidates = max(cli.direct_rasterizer_max_records, cli.direct_rasterizer_candidate_count)
    step = max(1, int(visible.numel()) // max_candidates)
    sampled = visible[::step][: max_candidates].detach().cpu().tolist()
    scales = gm.get_scaling.detach().cpu()
    rotations = gm.get_rotation.detach().cpu()
    scored: List[Tuple[float, int]] = []
    for index in sampled:
        if index < 0 or index >= scales.shape[0]:
            continue
        scale = scales[index]
        rotation = rotations[index] if index < rotations.shape[0] else None
        anisotropy = float((scale.max() - scale.min()).abs().item()) if scale.numel() >= 3 else 0.0
        rotation_vec_norm = float(torch.linalg.vector_norm(rotation[1:]).item()) if rotation is not None and rotation.numel() >= 4 else 0.0
        scored.append((anisotropy + rotation_vec_norm, int(index)))
    scored.sort(reverse=True)
    return [index for _, index in scored[: max(1, cli.direct_rasterizer_max_records)]]


def capture_direct_rasterizer_evidence(
    *,
    render_func,
    viewpoint,
    gm: GaussianModel,
    args: SimpleNamespace,
    bg,
    render_pkg: Dict[str, object],
    cli: argparse.Namespace,
    out_path: Path,
    run_id: str,
    population_selection: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    index_lineage = build_direct_evidence_index_lineage(
        gm,
        population_selection,
    )
    selected_local = select_direct_rasterizer_candidates(
        gm,
        render_pkg,
        cli,
        population_selection,
    )
    range_start = int(index_lineage["rangeStart"])
    selected_original = [range_start + index for index in selected_local]
    original_target = getattr(args, "debug_preprocess_target_index", -1)
    records = []
    try:
        for local_index, original_index in zip(
            selected_local,
            selected_original,
        ):
            args.debug_preprocess_target_index = int(local_index)
            debug_pkg = render_func(viewpoint, gm, args, bg)
            row = tensor_to_cpu_list(debug_pkg.get("cuda_preprocess_debug"))
            if row:
                decoded = decode_preprocess_debug_row(row)
                cuda_reported_local_index = decoded.get("srcIndex")
                if (
                    decoded.get("active") is True
                    and cuda_reported_local_index != local_index
                ):
                    raise PopulationSelectionError(
                        "direct-rasterizer-local-index-mismatch:"
                        f"requested={local_index},reported={cuda_reported_local_index}"
                    )
            else:
                decoded = {
                    "active": False,
                    "srcIndex": int(original_index),
                    "valid": False,
                    "culled": True,
                    "missingReason": "empty-cuda-preprocess-debug-row",
                }
                cuda_reported_local_index = None
            decoded["cudaReportedRasterizerLocalIndex"] = (
                int(cuda_reported_local_index)
                if cuda_reported_local_index is not None
                else None
            )
            decoded["rasterizerLocalIndex"] = int(local_index)
            decoded["requestedRasterizerLocalIndex"] = int(local_index)
            decoded["originalSrcIndex"] = int(original_index)
            decoded["srcIndex"] = int(original_index)
            decoded["requestedSrcIndex"] = int(original_index)
            decoded["actualEvidenceSource"] = "cuda-production-rasterizer-preprocess-kernel-debug-row"
            decoded["sameProductionRasterizerInvocation"] = True
            records.append(decoded)
    finally:
        args.debug_preprocess_target_index = original_target

    valid_records = [record for record in records if record.get("valid") is True]
    evidence = {
        "schemaVersion": (
            "phase3-step114-direct-cuda-rasterizer-evidence-v2"
            if population_selection is not None
            and population_selection.get("selectionMode") == "range"
            else "phase3-step114-direct-cuda-rasterizer-evidence-v1"
        ),
        "runId": run_id,
        "available": len(records) > 0,
        "actualEvidenceSource": "cuda-production-rasterizer-preprocess-kernel-debug-row",
        "selectionPolicy": {
            "explicitIndices": parse_optional_indices(cli.direct_rasterizer_indices),
            "candidateCount": cli.direct_rasterizer_candidate_count,
            "maxRecords": cli.direct_rasterizer_max_records,
            "selectedIndices": selected_original,
            "selectedOriginalSourceIndices": selected_original,
            "selectedRasterizerLocalIndices": selected_local,
        },
        "indexLineage": index_lineage,
        "populationSelection": (
            None
            if population_selection is None
            else {
                "schemaVersion": population_selection.get("schemaVersion"),
                "selectionMode": population_selection.get("selectionMode"),
                "requestedSelection": population_selection.get(
                    "requestedSelection"
                ),
                "productionRasterizerRecordCount": population_selection.get(
                    "productionRasterizerRecordCount"
                ),
                "populationAlignmentReady": population_selection.get(
                    "populationAlignmentReady"
                ),
            }
        ),
        "screenSpaceUnits": {
            "centerUnits": "pixel",
            "pixelOrigin": "top-left-rasterizer-image-space",
            "xDirection": "increasing-column-index-right",
            "yDirection": "increasing-row-index-down",
            "ndcToPixel": "((v + 1) * S - 1) * 0.5",
            "halfPixelConvention": "ndc-minus-one-maps-to-minus-0.5; ndc-plus-one-maps-to-S-minus-0.5",
        },
        "recordCount": len(records),
        "validRecordCount": len(valid_records),
        "records": records,
        "artifact": {
            "absolutePath": str(out_path),
            "exists": out_path.exists(),
            "runId": run_id,
        },
    }
    out_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    import hashlib
    evidence["artifact"] = {
        "absolutePath": str(out_path),
        "exists": out_path.exists(),
        "sizeBytes": out_path.stat().st_size if out_path.exists() else None,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest() if out_path.exists() else None,
        "runId": run_id,
    }
    out_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return evidence


@torch.no_grad()
def main() -> None:
    cli = parse_args()
    model_path = Path(cli.model_path)
    args, path_remaps = load_cfg_args(model_path)
    if cli.source_path:
        args.source_path = cli.source_path
    args.source_path = str(Path(args.source_path))
    args.model_path = str(model_path)

    source_path = Path(args.source_path)
    ckpt_path = resolve_checkpoint(args, cli.ckpt, model_path, cli.iteration)
    population_request = validate_population_selection_cli(cli)
    population_provenance_gate = None
    if population_request is not None:
        population_provenance_gate = validate_population_provenance(
            provenance_path=population_request["provenancePath"],
            checkpoint_path=ckpt_path,
            spl4_path=population_request["spl4Path"],
            requested_start=population_request["startInclusive"],
            requested_count=population_request["selectedCount"],
        )
    train_frames, test_frames, frame_lookup = read_frame_lists(source_path)
    inferred_time_duration = infer_time_duration(train_frames, test_frames)
    if getattr(args, "time_duration", None) is None and inferred_time_duration is not None:
        args.time_duration = inferred_time_duration
    camera_infos = build_camera_infos(args, cli.which)
    sample_indices = parse_sample_indices(cli.sample_indices)
    selected_infos = select_camera_infos(camera_infos, cli.which, sample_indices, cli.seed)
    if cli.limit > 0:
        selected_infos = selected_infos[:cli.limit]

    out_dir = Path(cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir = model_path / "renders_png" / f"iter_{cli.iteration:06d}" / "train"
    cameras = cameraList_from_camInfos(selected_infos, resolution_scale=1.0, args=args)
    camera_dataset = CameraDataset(cameras, args.white_background)
    gm = None
    bg = None
    population_selection = None
    if not cli.reuse_legacy_training_report:
        from gaussian_renderer import render
        if population_provenance_gate is None:
            gm = build_gaussian_model(args, ckpt_path)
        else:
            gm, population_selection = build_population_aligned_gaussian_model(
                args,
                ckpt_path,
                population_provenance_gate,
                selection_mode=population_request["mode"],
            )
        bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda") if bool(args.white_background) else torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    index_records = []
    dataset_identity = {
        "transforms_train_frame_count": len(train_frames),
        "transforms_test_frame_count": len(test_frames),
        "image_file_count": len(list((source_path / "images").glob("*.png"))),
        "mask_file_count": len(list((source_path / "masks").glob("*.png")))
    }

    for local_idx, (camera_info, camera) in enumerate(zip(selected_infos, cameras)):
        gt_image, viewpoint = camera_dataset[local_idx]
        image_name = viewpoint.image_name

        render_path = out_dir / f"{image_name}_render.png"
        gt_path = out_dir / f"{image_name}_gt.png"
        alpha_path = out_dir / f"{image_name}_alpha.png"
        depth_path = out_dir / f"{image_name}_depth.png"
        meta_path = out_dir / f"{image_name}_meta.json"
        manifest_path = out_dir / f"{image_name}_render_state_manifest.json"
        direct_rasterizer_evidence_path = out_dir / f"{image_name}_direct_rasterizer_evidence.json"

        legacy_render_path = legacy_dir / f"{local_idx:03d}_render.png"
        legacy_gt_path = legacy_dir / f"{local_idx:03d}_gt.png"
        legacy_alpha_path = legacy_dir / f"{local_idx:03d}_alpha.png"
        legacy_depth_path = legacy_dir / f"{local_idx:03d}_depth.png"

        render_pkg = None
        if cli.reuse_legacy_training_report:
            if not legacy_render_path.exists():
                raise FileNotFoundError(f"Missing legacy training-report render: {legacy_render_path}")
            shutil.copy2(legacy_render_path, render_path)
            if legacy_gt_path.exists():
                shutil.copy2(legacy_gt_path, gt_path)
            else:
                save_rgb_tensor_chw01(gt_image, gt_path)
            if legacy_alpha_path.exists():
                shutil.copy2(legacy_alpha_path, alpha_path)
            if legacy_depth_path.exists():
                shutil.copy2(legacy_depth_path, depth_path)
        else:
            viewpoint = viewpoint.cuda()
            render_pkg = render(viewpoint, gm, args, bg)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            alpha = torch.clamp(render_pkg["alpha"], 0.0, 1.0)
            depth = render_pkg["depth"]

            save_rgb_tensor_chw01(image, render_path)
            save_rgb_tensor_chw01(gt_image, gt_path)
            save_single_channel_tensor_chw01_as_rgb(alpha, alpha_path)

            depth_vis = depth[0]
            depth_min = float(depth_vis.min().item()) if depth_vis.numel() else 0.0
            depth_max = float(depth_vis.max().item()) if depth_vis.numel() else 0.0
            denom = max(depth_max - depth_min, 1e-8)
            depth_vis = ((depth_vis - depth_min) / denom).clamp(0.0, 1.0).unsqueeze(0)
            save_single_channel_tensor_chw01_as_rgb(depth_vis, depth_path)

        frame_record = frame_lookup.get(image_name)
        render_context = {
            "selectionMode": cli.which,
            "selectionSeed": cli.seed,
            "trainingReportSampleIndices": sample_indices if cli.which == "training-report-train-sample" else [],
            "localSelectionIndex": local_idx,
            "reuseLegacyTrainingReport": bool(cli.reuse_legacy_training_report),
            "checkpointPath": str(ckpt_path),
            "iteration": cli.iteration,
            "modelPath": str(model_path),
            "sourcePath": str(source_path),
            "pathRemaps": path_remaps,
            "datasetIdentity": dataset_identity,
            "timeDuration": list(getattr(args, "time_duration", [])) if getattr(args, "time_duration", None) is not None else None,
            "legacyTrainingReportSlot": local_idx if cli.which == "training-report-train-sample" else None
        }
        render_paths = {
            "render": str(render_path),
            "gt": str(gt_path),
            "alpha": str(alpha_path),
            "depth": str(depth_path)
        }
        meta = build_meta(viewpoint, frame_record, render_paths, render_context)

        if legacy_render_path.exists():
            meta["legacyTrainingReportRenderPath"] = str(legacy_render_path)
        if legacy_gt_path.exists():
            meta["legacyTrainingReportGtPath"] = str(legacy_gt_path)
        if legacy_alpha_path.exists():
            meta["legacyTrainingReportAlphaPath"] = str(legacy_alpha_path)
        if legacy_depth_path.exists():
            meta["legacyTrainingReportDepthPath"] = str(legacy_depth_path)

        meta_path.write_text(json.dumps(meta, indent=2))
        run_id = build_run_id(image_name)
        direct_rasterizer_evidence = None
        if (
            cli.direct_rasterizer_evidence
            and render_pkg is not None
            and gm is not None
            and not cli.reuse_legacy_training_report
        ):
            direct_rasterizer_evidence = capture_direct_rasterizer_evidence(
                render_func=render,
                viewpoint=viewpoint,
                gm=gm,
                args=args,
                bg=bg,
                render_pkg=render_pkg,
                cli=cli,
                out_path=direct_rasterizer_evidence_path,
                run_id=run_id,
                population_selection=population_selection,
            )
        manifest = build_cuda_reference_manifest(
            run_id=run_id,
            repo_path=Path(__file__).resolve().parents[1],
            script_path=Path(__file__).resolve(),
            cfg_path=model_path / "cfg_args",
            argv=sys.argv,
            cwd=Path.cwd(),
            args=args,
            cli=cli,
            ckpt_path=ckpt_path,
            source_path=source_path,
            model_path=model_path,
            camera=viewpoint,
            meta=meta,
            render_paths=render_paths,
            meta_path=meta_path,
            manifest_path=manifest_path,
            gaussian_model=gm,
            dataset_identity=dataset_identity,
            path_remaps=path_remaps,
            reuse_legacy_training_report=bool(cli.reuse_legacy_training_report),
            direct_rasterizer_evidence=direct_rasterizer_evidence,
            population_selection=population_selection,
        )
        manifest = write_manifest(manifest_path, manifest)
        index_records.append({
            "image_name": image_name,
            "frame_number": meta["frame_number"],
            "view_id": meta["view_id"],
            "timestamp": meta["timestamp"],
            "meta_path": str(meta_path),
            "render_path": str(render_path),
            "render_state_manifest_path": str(manifest_path),
            "render_state_manifest_run_id": manifest.get("runId"),
            "legacy_slot": local_idx if cli.which == "training-report-train-sample" else None
        })

    index = {
        "selectionMode": cli.which,
        "iteration": cli.iteration,
        "checkpointPath": str(ckpt_path),
        "modelPath": str(model_path),
        "sourcePath": str(source_path),
        "pathRemaps": path_remaps,
        "datasetIdentity": dataset_identity,
        "records": index_records
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"[DONE] wrote {len(index_records)} named reference renders to {out_dir}")


if __name__ == "__main__":
    main()
