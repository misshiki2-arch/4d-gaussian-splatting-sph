#!/usr/bin/env python3
"""Shared CUDA reference render-state manifest helpers.

The manifest intentionally records unknown evidence explicitly.  Step114 uses
that to block visual parity comparisons until CUDA and WebGPU render states can
be compared from canonical sources instead of inferred filenames.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


SCHEMA_VERSION = "phase3-render-state-manifest-v2"


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_identity(path: Path) -> Dict[str, Any]:
    return {
        "absolutePath": str(path.resolve()) if path.exists() else str(path),
        "exists": path.exists(),
        "sizeBytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256_file(path),
    }


def git_info(repo_path: Path) -> Dict[str, Any]:
    def run_git(args: Iterable[str]) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo_path), *args],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return None

    status = run_git(["status", "--short"])
    return {
        "repoPath": str(repo_path.resolve()) if repo_path.exists() else str(repo_path),
        "revision": run_git(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "statusShort": status,
    }


def status_value(value: Any, *, source: str, status: str = "available") -> Dict[str, Any]:
    return {"status": status, "source": source, "value": value}


def unknown(reason: str, *, source: str = "cuda-reference-render-tool") -> Dict[str, Any]:
    return {"status": "unknown", "source": source, "reason": reason, "value": None}


def module_identity(module_name: str) -> Dict[str, Any]:
    try:
        module = __import__(module_name)
    except Exception as exc:  # noqa: BLE001
        return {"moduleName": module_name, "available": False, "error": str(exc)}
    module_file = getattr(module, "__file__", None)
    path = Path(module_file) if module_file else None
    return {
        "moduleName": module_name,
        "available": True,
        "file": str(path) if path else None,
        "fileIdentity": file_identity(path) if path else None,
    }


def build_run_id(image_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"cuda-reference-{image_name}-{stamp}-{uuid.uuid4().hex[:8]}"


def build_cuda_reference_manifest(
    *,
    run_id: str,
    repo_path: Path,
    script_path: Path,
    cfg_path: Path,
    argv: list[str],
    cwd: Path,
    args: Any,
    cli: Any,
    ckpt_path: Path,
    source_path: Path,
    model_path: Path,
    camera: Any,
    meta: Dict[str, Any],
    render_paths: Dict[str, str],
    meta_path: Path,
    manifest_path: Path,
    gaussian_model: Any,
    dataset_identity: Dict[str, Any],
    path_remaps: list[Dict[str, str]],
    reuse_legacy_training_report: bool,
    direct_rasterizer_evidence: Optional[Dict[str, Any]] = None,
    population_selection: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    frame_number, view_id = meta.get("frame_number"), meta.get("view_id")
    gaussian_count = None
    if gaussian_model is not None:
        try:
            gaussian_count = int(gaussian_model.get_xyz.shape[0])
        except Exception:
            gaussian_count = None

    render_artifacts: Dict[str, Any] = {
        key: file_identity(Path(value)) for key, value in render_paths.items()
    }
    render_artifacts["meta"] = file_identity(meta_path)
    render_artifacts["manifest"] = {
        "absolutePath": str(manifest_path),
        "exists": manifest_path.exists(),
        "runId": run_id,
    }

    world_view = meta.get("world_view_transform")
    full_proj = meta.get("full_proj_transform")
    transform_matrix = meta.get("transform_matrix")
    direct_evidence = (
        direct_rasterizer_evidence
        if isinstance(direct_rasterizer_evidence, dict)
        else {
            "available": False,
            "reason": "screen-space centers/Jacobian were not emitted by CUDA rasterizer",
        }
    )
    direct_evidence_available = direct_evidence.get("available") is True
    direct_artifact = direct_evidence.get("artifact") if isinstance(direct_evidence.get("artifact"), dict) else None
    direct_units = direct_evidence.get("screenSpaceUnits", {}) if isinstance(direct_evidence.get("screenSpaceUnits"), dict) else {}
    if direct_artifact is not None:
        render_artifacts["directRasterizerEvidence"] = direct_artifact

    population_contract = None
    checkpoint_identity = None
    if population_selection is not None:
        if not isinstance(population_selection, dict):
            raise TypeError("population_selection must be a mapping")
        if population_selection.get("populationAlignmentReady") is not True:
            raise ValueError("population_selection is not ready")
        population_contract = population_selection
        checkpoint_identity = population_contract.get("currentCheckpointIdentity")
        if not isinstance(checkpoint_identity, dict):
            raise ValueError("population_selection checkpoint identity is missing")
    if checkpoint_identity is None:
        checkpoint_identity = file_identity(ckpt_path)

    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "manifestKind": "cuda-reference-render-state",
        "runId": run_id,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "execution": {
            "argv": argv,
            "cwd": str(cwd),
            "repo": git_info(repo_path),
            "python": {
                "executable": sys.executable,
                "version": sys.version,
                "platform": platform.platform(),
            },
            "environment": {
                "condaDefaultEnv": os.environ.get("CONDA_DEFAULT_ENV"),
                "cudaVisibleDevices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "torch": {
                "version": getattr(__import__("torch"), "__version__", None),
                "cudaVersion": getattr(__import__("torch").version, "cuda", None),
            },
            "rasterizer": module_identity("diff_gaussian_rasterization"),
            "scriptIdentity": file_identity(script_path),
            "configIdentity": file_identity(cfg_path),
        },
        "lineage": {
            "checkpoint": checkpoint_identity,
            "checkpointPath": str(ckpt_path),
            "datasetRoot": str(source_path),
            "datasetIdentity": dataset_identity,
            "modelPath": str(model_path),
            "pathRemaps": path_remaps,
            "reuseLegacyTrainingReport": bool(reuse_legacy_training_report),
        },
        "renderState": {
            "cameraMetadataName": meta.get("image_name"),
            "cameraLabel": meta.get("image_name"),
            "cameraIndex": getattr(camera, "uid", None),
            "frameNumber": frame_number,
            "viewId": view_id,
            "timestamp": meta.get("timestamp"),
            "timeDuration": meta.get("render_context", {}).get("timeDuration"),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "backgroundPolicy": {
                "whiteBackground": bool(getattr(args, "white_background", False)),
                "rgb": [1, 1, 1] if bool(getattr(args, "white_background", False)) else [0, 0, 0],
            },
            "gaussianCount": gaussian_count,
            "shDegree": getattr(args, "sh_degree", None),
            "shDegreeTemporal": getattr(args, "sh_degree_t", None),
            "gaussianDim": getattr(args, "gaussian_dim", None),
            "rot4d": getattr(args, "rot_4d", None),
            "forceSh3d": getattr(args, "force_sh_3d", None),
            "prefilterVar": getattr(args, "prefilter_var", None),
            "opacityPolicy": unknown("opacity evaluation is internal to CUDA renderer; no per-render aggregate emitted"),
            "temporalWeightingPolicy": status_value(
                {
                    "timestamp": meta.get("timestamp"),
                    "timeDuration": meta.get("render_context", {}).get("timeDuration"),
                    "gaussianDim": getattr(args, "gaussian_dim", None),
                    "rot4d": getattr(args, "rot_4d", None),
                },
                source="cuda-reference-render-tool",
            ),
        },
        "camera": {
            "source": "CUDA Camera object from cameraList_from_camInfos/CameraDataset",
            "intrinsics": {
                "fx": meta.get("fx"),
                "fy": meta.get("fy"),
                "cx": meta.get("cx"),
                "cy": meta.get("cy"),
                "FoVx": meta.get("FoVx"),
                "FoVy": meta.get("FoVy"),
                "tanFovx": None if meta.get("FoVx") is None else __import__("math").tan(float(meta["FoVx"]) * 0.5),
                "tanFovy": None if meta.get("FoVy") is None else __import__("math").tan(float(meta["FoVy"]) * 0.5),
            },
            "cameraCenter": meta.get("camera_center"),
            "worldViewTransform": world_view,
            "fullProjTransform": full_proj,
            "transformMatrix": transform_matrix,
            "matrixConvention": {
                "layout": "torch-tensor-list-as-emitted-by-Camera-object",
                "multiplicationOrder": "renderer-consumes-world_view_transform/full_proj_transform-directly",
                "status": "emitted-not-independently-proven",
            },
        },
        "imageSpaceConvention": {
            "pixelOrigin": status_value(
                direct_units.get("pixelOrigin"),
                source="cuda-rasterizer-preprocess-debug-row",
            ) if direct_evidence_available and direct_units.get("pixelOrigin") else unknown("CUDA rasterizer screen-coordinate origin is not emitted directly"),
            "xDirection": status_value("increasing-column-index-right", source="PNG writer"),
            "yDirection": status_value(
                direct_units.get("yDirection"),
                source="cuda-rasterizer-preprocess-debug-row",
            ) if direct_evidence_available and direct_units.get("yDirection") else unknown("CUDA rasterizer y convention is not emitted directly"),
            "ndcToPixel": status_value(
                direct_units.get("ndcToPixel"),
                source="diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h::ndc2Pix",
            ) if direct_evidence_available and direct_units.get("ndcToPixel") else unknown("CUDA rasterizer NDC-to-pixel formula is not emitted directly"),
            "halfPixelConvention": status_value(
                direct_units.get("halfPixelConvention"),
                source="diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h::ndc2Pix",
            ) if direct_evidence_available and direct_units.get("halfPixelConvention") else unknown("CUDA rasterizer half-pixel convention is not emitted directly"),
            "pngRowOrder": status_value(
                "top-to-bottom-row-major-after-tensor-CHW-to-HWC",
                source="save_rgb_tensor_chw01",
            ),
            "transposeOrFlipApplied": status_value(False, source="save_rgb_tensor_chw01"),
            "directRasterizerScreenCoordinateEvidence": direct_evidence,
        },
        "artifacts": render_artifacts,
        "renderContext": meta.get("render_context", {}),
        "cli": {
            "selectionMode": getattr(cli, "which", None),
            "sampleIndices": getattr(cli, "sample_indices", None),
            "seed": getattr(cli, "seed", None),
            "iteration": getattr(cli, "iteration", None),
            "limit": getattr(cli, "limit", None),
        },
    }
    if population_contract is not None:
        manifest["lineage"]["populationSelection"] = population_contract
    return manifest


def write_manifest(path: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest = dict(manifest)
    artifacts = dict(manifest.get("artifacts", {}))
    artifacts["manifest"] = file_identity(path)
    artifacts["manifest"]["runId"] = manifest.get("runId")
    manifest["artifacts"] = artifacts
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
