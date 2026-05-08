#!/usr/bin/env python3
"""Capture CUDA rasterizer contributor debug for one Step90 pixel."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from tools.render_named_reference_from_ckpt import (
    build_camera_infos,
    build_gaussian_model,
    infer_time_duration,
    load_cfg_args,
    read_frame_lists,
    resolve_checkpoint,
)
from utils.camera_utils import cameraList_from_camInfos


DEBUG_COLUMNS = [
    "localOrder",
    "globalRangeOffset",
    "gaussianIndex",
    "depth",
    "centerX",
    "centerY",
    "conicX",
    "conicY",
    "conicZ",
    "opacity",
    "colorR",
    "colorG",
    "colorB",
    "dx",
    "dy",
    "power",
    "rawAlpha",
    "alpha",
    "skipCode",
    "TBefore",
    "test_T",
    "TAfter",
    "contributionR",
    "contributionG",
    "contributionB",
    "accumR",
    "accumG",
    "accumB",
    "contributed",
    "lastContributor",
    "accumDepth",
    "reserved",
]

SKIP_REASONS = {
    0: "none",
    1: "power-positive",
    2: "alpha-below-1-over-255",
    3: "early-out-testT-below-0.0001",
}

PREPROCESS_DEBUG_COLUMNS = [
    "valid", "gaussianIndex",
    "inputMeanX", "inputMeanY", "inputMeanZ",
    "conditionedMeanX", "conditionedMeanY", "conditionedMeanZ",
    "opacityBeforeTime", "opacityAfterTime",
    "scaleX", "scaleY", "scaleZ", "scaleT",
    "rotationW", "rotationX", "rotationY", "rotationZ",
    "rotationRW", "rotationRX", "rotationRY", "rotationRZ",
    "tCenter", "timestamp", "dt", "timeMask",
    "cov3D00", "cov3D01", "cov3D02", "cov3D11", "cov3D12", "cov3D22",
    "viewSpaceX", "viewSpaceY", "viewSpaceZ",
    "viewSpaceClampedX", "viewSpaceClampedY", "viewSpaceClampedZ",
    "J00", "J01", "J02", "J10", "J11", "J12", "J20", "J21", "J22",
    "W00", "W01", "W02", "W10", "W11", "W12", "W20", "W21", "W22",
    "T00", "T01", "T02", "T10", "T11", "T12", "T20", "T21", "T22",
    "cov2DBefore00", "cov2DBefore01", "cov2DBefore10", "cov2DBefore11",
    "cov2DAfter00", "cov2DAfter01", "cov2DAfter10", "cov2DAfter11",
    "determinant", "conicA", "conicB", "conicC",
    "radius", "tileRectMinX", "tileRectMinY", "tileRectMaxX", "tileRectMaxY",
    "centerX", "centerY", "depth", "focalX", "focalY", "tanFovX", "tanFovY",
    "debugPixelX", "debugPixelY", "dx", "dy", "power", "rawAlpha", "alpha",
]


def parse_pixel(value: str) -> Tuple[int, int]:
    x_text, y_text = value.split(",", 1)
    return int(x_text.strip()), int(y_text.strip())


def load_viewer_pixel(path: Path, pixel: Tuple[int, int]) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    for item in data.get("pixels", []):
        if item.get("pixel") == [pixel[0], pixel[1]]:
            return item
    return None


def read_png_rgb01(path: Path, pixel: Tuple[int, int]) -> Optional[List[float]]:
    if not path.exists():
        return None
    image = Image.open(path).convert("RGBA")
    rgba = image.getpixel(pixel)
    return [rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0]


def render_tensor_to_rgb_image(tensor: torch.Tensor) -> Image.Image:
    """Convert renderer output C,H,W RGB float tensor in [0,1] to an 8-bit RGB PIL image."""
    rgb = tensor.detach().clamp(0, 1).cpu()
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"Expected render tensor shape [3,H,W], got {tuple(rgb.shape)}")
    hwc = rgb.permute(1, 2, 0).mul(255.0).round().to(torch.uint8).numpy()
    return Image.fromarray(hwc, mode="RGB")


def ellipse_summary_from_cov2d(cov: Sequence[Sequence[float]]) -> Dict[str, Any]:
    a = float(cov[0][0])
    b = float(cov[0][1])
    c = float(cov[1][1])
    trace = a + c
    disc = math.sqrt(max(0.0, (a - c) * (a - c) + 4.0 * b * b))
    lambda_min = 0.5 * (trace - disc)
    lambda_max = 0.5 * (trace + disc)
    axis_min = math.sqrt(max(0.0, lambda_min))
    axis_max = math.sqrt(max(0.0, lambda_max))
    angle = 0.5 * math.degrees(math.atan2(2.0 * b, a - c))
    return {
        "eigenvalues": [lambda_min, lambda_max],
        "sqrtEigenAxesPx": [axis_min, axis_max],
        "axisRatio": axis_max / axis_min if axis_min > 0 else None,
        "majorAxisOrientationDeg": angle,
    }


def cov2d_from_conic(conic: Sequence[float]) -> Optional[List[List[float]]]:
    if not conic or len(conic) < 3:
        return None
    a, b, c = [float(v) for v in conic[:3]]
    det = a * c - b * b
    if det == 0.0 or not math.isfinite(det):
        return None
    inv = 1.0 / det
    return [[c * inv, -b * inv], [-b * inv, a * inv]]


def decode_preprocess_debug_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
    if tensor.numel() == 0:
        return {"valid": False, "reason": "debug-preprocess-empty"}
    row = tensor.detach().cpu().tolist()
    values = {name: row[i] for i, name in enumerate(PREPROCESS_DEBUG_COLUMNS) if i < len(row)}
    valid = bool(round(values.get("valid", 0.0)))
    result: Dict[str, Any] = {
        "valid": valid,
        "schemaVersion": "step90-cuda-preprocess-trace-v1",
        "rawRow": row,
        "rawColumns": PREPROCESS_DEBUG_COLUMNS,
    }
    if not valid:
        result["reason"] = "target-index-not-recorded"
        return result
    cov3d = [
        [values["cov3D00"], values["cov3D01"], values["cov3D02"]],
        [values["cov3D01"], values["cov3D11"], values["cov3D12"]],
        [values["cov3D02"], values["cov3D12"], values["cov3D22"]],
    ]
    cov_before = [
        [values["cov2DBefore00"], values["cov2DBefore01"]],
        [values["cov2DBefore10"], values["cov2DBefore11"]],
    ]
    cov_after = [
        [values["cov2DAfter00"], values["cov2DAfter01"]],
        [values["cov2DAfter10"], values["cov2DAfter11"]],
    ]
    result.update({
        "index": int(round(values["gaussianIndex"])),
        "inputMean": [values["inputMeanX"], values["inputMeanY"], values["inputMeanZ"]],
        "conditionedMean": [values["conditionedMeanX"], values["conditionedMeanY"], values["conditionedMeanZ"]],
        "opacityBeforeTime": values["opacityBeforeTime"],
        "opacityAfterTime": values["opacityAfterTime"],
        "scale": [values["scaleX"], values["scaleY"], values["scaleZ"]],
        "scaleT": values["scaleT"],
        "rotation": [values["rotationW"], values["rotationX"], values["rotationY"], values["rotationZ"]],
        "rotationR": [values["rotationRW"], values["rotationRX"], values["rotationRY"], values["rotationRZ"]],
        "tCenter": values["tCenter"],
        "timestamp": values["timestamp"],
        "dt": values["dt"],
        "timeMask": bool(round(values["timeMask"])),
        "cov3D": cov3d,
        "viewSpace": [values["viewSpaceX"], values["viewSpaceY"], values["viewSpaceZ"]],
        "viewSpaceClamped": [values["viewSpaceClampedX"], values["viewSpaceClampedY"], values["viewSpaceClampedZ"]],
        "jacobian": [
            [values["J00"], values["J01"], values["J02"]],
            [values["J10"], values["J11"], values["J12"]],
            [values["J20"], values["J21"], values["J22"]],
        ],
        "viewRotationW": [
            [values["W00"], values["W01"], values["W02"]],
            [values["W10"], values["W11"], values["W12"]],
            [values["W20"], values["W21"], values["W22"]],
        ],
        "T_W_mul_J": [
            [values["T00"], values["T01"], values["T02"]],
            [values["T10"], values["T11"], values["T12"]],
            [values["T20"], values["T21"], values["T22"]],
        ],
        "cov2DBeforeOffset": cov_before,
        "cov2DAfterOffset": cov_after,
        "lowPassOffset": [0.3, 0.3],
        "determinant": values["determinant"],
        "conic": [values["conicA"], values["conicB"], values["conicC"]],
        "radius": values["radius"],
        "tileRect": [values["tileRectMinX"], values["tileRectMinY"], values["tileRectMaxX"], values["tileRectMaxY"]],
        "centerPx": [values["centerX"], values["centerY"]],
        "depth": values["depth"],
        "focal": [values["focalX"], values["focalY"]],
        "tanFov": [values["tanFovX"], values["tanFovY"]],
        "pixel": [int(round(values["debugPixelX"])), int(round(values["debugPixelY"]))],
        "dx": values["dx"],
        "dy": values["dy"],
        "power": values["power"],
        "rawAlpha": values["rawAlpha"],
        "alpha": values["alpha"],
        "ellipse": ellipse_summary_from_cov2d(cov_after),
    })
    return result


def compare_rgb_images(reference_path: Path, debug_image: Image.Image, pixel: Tuple[int, int], absdiff_path: Optional[Path]) -> Dict[str, Any]:
    reference = Image.open(reference_path).convert("RGB")
    debug = debug_image.convert("RGB")
    if reference.size != debug.size:
        raise ValueError(f"Image size mismatch: reference={reference.size}, debug={debug.size}")
    ref_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(reference.tobytes())).view(reference.height, reference.width, 3).float() / 255.0
    dbg_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(debug.tobytes())).view(debug.height, debug.width, 3).float() / 255.0
    diff = (ref_tensor - dbg_tensor).abs()
    if absdiff_path is not None:
        absdiff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_image = Image.fromarray(diff.mul(255.0).round().clamp(0, 255).to(torch.uint8).numpy(), mode="RGB")
        diff_image.save(absdiff_path)
    ref_nonblack = ref_tensor.amax(dim=2) > 0
    dbg_nonblack = dbg_tensor.amax(dim=2) > 0
    mask_diff = ref_nonblack != dbg_nonblack
    x, y = pixel
    return {
        "referencePath": str(reference_path),
        "debugImageSize": list(debug.size),
        "referenceImageSize": list(reference.size),
        "debugImageMode": debug.mode,
        "referenceImageMode": reference.mode,
        "pixel": [x, y],
        "referencePixelRgb": ref_tensor[y, x].tolist(),
        "debugPixelRgb": dbg_tensor[y, x].tolist(),
        "pixelAbsDiff": diff[y, x].tolist(),
        "meanAbsError": float(diff.mean().item()),
        "maxAbsError": float(diff.max().item()),
        "rmse": float(torch.sqrt(((ref_tensor - dbg_tensor) ** 2).mean()).item()),
        "nonBlackMask": {
            "threshold": "rgb max > 0",
            "referenceCount": int(ref_nonblack.sum().item()),
            "debugCount": int(dbg_nonblack.sum().item()),
            "differentCount": int(mask_diff.sum().item()),
            "differentFraction": float(mask_diff.float().mean().item()),
        },
        "absdiffPath": str(absdiff_path) if absdiff_path is not None else None,
    }


def first_divergence(cuda_entries: Sequence[Dict[str, Any]], viewer_entries: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    n = min(len(cuda_entries), len(viewer_entries))
    for i in range(n):
        c = cuda_entries[i]
        v = viewer_entries[i]
        if int(c["gaussianIndex"]) != int(v.get("originalSplatIndex", -1)):
            return {
                "kind": "gaussian-index",
                "order": i,
                "cudaGaussianIndex": int(c["gaussianIndex"]),
                "viewerOriginalSplatIndex": v.get("originalSplatIndex"),
            }
        if c["skipReason"] != v.get("skipReason"):
            return {
                "kind": "skip-reason",
                "order": i,
                "gaussianIndex": int(c["gaussianIndex"]),
                "cudaSkipReason": c["skipReason"],
                "viewerSkipReason": v.get("skipReason"),
            }
    if len(cuda_entries) != len(viewer_entries):
        return {
            "kind": "length",
            "cudaLength": len(cuda_entries),
            "viewerLength": len(viewer_entries),
        }
    return None


def decode_debug_tensor(tensor: torch.Tensor, pixel: Tuple[int, int]) -> Dict[str, Any]:
    if tensor.numel() == 0:
        return {
            "valid": False,
            "reason": "debug-tensor-empty",
            "targetPixel": list(pixel),
        }
    rows = tensor.detach().cpu().tolist()
    header = rows[0]
    record_count = max(0, int(round(header[13])))
    entries: List[Dict[str, Any]] = []
    for row in rows[1:1 + record_count]:
        entry = {name: row[i] for i, name in enumerate(DEBUG_COLUMNS)}
        skip_code = int(round(entry["skipCode"]))
        entry["skipReason"] = SKIP_REASONS.get(skip_code, f"unknown-{skip_code}")
        entry["gaussianIndex"] = int(round(entry["gaussianIndex"]))
        entry["localOrder"] = int(round(entry["localOrder"]))
        entry["globalRangeOffset"] = int(round(entry["globalRangeOffset"]))
        entry["contributed"] = bool(round(entry["contributed"]))
        entry["centerPx"] = [entry.pop("centerX"), entry.pop("centerY")]
        entry["conic"] = [entry.pop("conicX"), entry.pop("conicY"), entry.pop("conicZ")]
        entry["color"] = [entry.pop("colorR"), entry.pop("colorG"), entry.pop("colorB")]
        entry["contributionRgb"] = [entry.pop("contributionR"), entry.pop("contributionG"), entry.pop("contributionB")]
        entry["accumColorAfter"] = [entry.pop("accumR"), entry.pop("accumG"), entry.pop("accumB")]
        entries.append(entry)
    return {
        "valid": bool(round(header[23])),
        "schemaVersion": "step90-cuda-pixel-contributor-debug-v1",
        "targetPixel": [int(round(header[3])), int(round(header[4]))],
        "actualPixel": [int(round(header[5])), int(round(header[6]))],
        "imageSize": [int(round(header[1])), int(round(header[2]))],
        "tile": [int(round(header[7])), int(round(header[8]))],
        "tileId": int(round(header[9])),
        "range": [int(round(header[10])), int(round(header[11]))],
        "tilePayloadCount": int(round(header[12])),
        "recordedCount": record_count,
        "contributorCounter": int(round(header[14])),
        "lastContributor": int(round(header[15])),
        "finalT": header[16],
        "accumColor": [header[17], header[18], header[19]],
        "accumDepth": header[20],
        "earlyOutTriggered": bool(round(header[21])),
        "earlyOutAtLocalOrder": int(round(header[22])),
        "done": bool(round(header[24])),
        "contributionCount": int(round(header[25])),
        "powerSkipCount": int(round(header[26])),
        "alphaSkipCount": int(round(header[27])),
        "finalRgb": [header[28], header[29], header[30]],
        "entries": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/home/demo/work/outputs/sph_scene_4dgs")
    parser.add_argument("--source-path", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--iteration", type=int, default=12000)
    parser.add_argument("--image-name", default="000195_v26")
    parser.add_argument("--pixel", default="655,363")
    parser.add_argument("--max-entries", type=int, default=2048)
    parser.add_argument("--viewer-debug-json", default="/home/demo/work/json/step90_tile_accumulation_debug.json")
    parser.add_argument("--viewer-association-json", default="/home/demo/work/json/step90_viewer_payload_index_association_debug_live_same_state_fix1.json")
    parser.add_argument("--cuda-reference-png", default="/home/demo/work/outputs/sph_scene_4dgs/cuda_reference_named/iter_012000/000195_v26_render.png")
    parser.add_argument("--cuda-output-json", default="/home/demo/work/json/step90_cuda_pixel_contributor_debug_655_363.json")
    parser.add_argument("--compare-output-json", default="/home/demo/work/json/step90_cuda_vs_viewer_pixel_655_363_compare.json")
    parser.add_argument("--preprocess-debug-index", type=int, default=-1)
    parser.add_argument("--preprocess-trace-output-json", default="/home/demo/work/json/step90_cuda_preprocess_trace_2718004.json")
    parser.add_argument("--cuda-debug-render-png", default=None)
    parser.add_argument("--cuda-debug-vs-reference-absdiff-png", default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    cli = parse_args()
    pixel = parse_pixel(cli.pixel)
    model_path = Path(cli.model_path)
    args, path_remaps = load_cfg_args(model_path)
    if cli.source_path:
        args.source_path = cli.source_path
    args.source_path = str(Path(args.source_path))
    args.model_path = str(model_path)
    source_path = Path(args.source_path)
    ckpt_path = resolve_checkpoint(args, cli.ckpt, model_path, cli.iteration)
    train_frames, test_frames, _ = read_frame_lists(source_path)
    inferred_time_duration = infer_time_duration(train_frames, test_frames)
    if getattr(args, "time_duration", None) is None and inferred_time_duration is not None:
        args.time_duration = inferred_time_duration
    args.debug_pixel_x = pixel[0]
    args.debug_pixel_y = pixel[1]
    args.debug_pixel_max_entries = int(cli.max_entries)
    args.debug_preprocess_target_index = int(cli.preprocess_debug_index)

    camera_infos = build_camera_infos(args, "training-report-train-sample")
    camera_info = next((item for item in camera_infos if item.image_name == cli.image_name), None)
    if camera_info is None:
        raise RuntimeError(f"Could not find camera image_name={cli.image_name}")
    viewpoint = cameraList_from_camInfos([camera_info], resolution_scale=1.0, args=args)[0].cuda()
    gm = build_gaussian_model(args, ckpt_path)
    bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda") if bool(args.white_background) else torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    render_pkg = render(viewpoint, gm, args, bg)
    render_tensor = render_pkg["render"]
    debug_render_image = render_tensor_to_rgb_image(render_tensor)
    debug_render_png_path = Path(cli.cuda_debug_render_png) if cli.cuda_debug_render_png else None
    if debug_render_png_path is not None:
        debug_render_png_path.parent.mkdir(parents=True, exist_ok=True)
        debug_render_image.save(debug_render_png_path)
    absdiff_path = Path(cli.cuda_debug_vs_reference_absdiff_png) if cli.cuda_debug_vs_reference_absdiff_png else None
    debug_render_compare = compare_rgb_images(Path(cli.cuda_reference_png), debug_render_image, pixel, absdiff_path)
    cuda_debug = decode_debug_tensor(render_pkg["cuda_pixel_debug"], pixel)
    cuda_preprocess_debug = decode_preprocess_debug_tensor(render_pkg.get("cuda_preprocess_debug", torch.empty(0)))
    cuda_debug.update({
        "imageName": cli.image_name,
        "checkpointPath": str(ckpt_path),
        "modelPath": str(model_path),
        "sourcePath": str(source_path),
        "pathRemaps": path_remaps,
        "renderTensorSummary": {
            "shape": list(render_tensor.shape),
            "channelOrder": "RGB",
            "tensorLayout": "C,H,W",
            "valueRangeBeforeSave": "float, expected 0..1",
            "saveClamp": "clamp to [0,1]",
            "saveQuantization": "round(value * 255) to uint8",
            "gammaCorrection": "none",
            "yFlip": "none",
            "debugRenderPng": str(debug_render_png_path) if debug_render_png_path is not None else None,
            "referenceCompare": debug_render_compare,
        },
        "preprocessTrace": cuda_preprocess_debug,
    })
    cuda_output = Path(cli.cuda_output_json)
    cuda_output.parent.mkdir(parents=True, exist_ok=True)
    cuda_output.write_text(json.dumps(cuda_debug, indent=2))

    viewer_pixel = load_viewer_pixel(Path(cli.viewer_debug_json), pixel)
    viewer_acc = viewer_pixel.get("accumulation", {}) if viewer_pixel else {}
    viewer_entries = viewer_acc.get("entries", [])
    cuda_ref_rgb = read_png_rgb01(Path(cli.cuda_reference_png), pixel)
    cuda_render_rgb = render_tensor[:, pixel[1], pixel[0]].detach().clamp(0, 1).cpu().tolist()
    viewer_framebuffer_rgb = viewer_pixel.get("framebuffer", {}).get("rgb") if viewer_pixel else None
    compare = {
        "schemaVersion": "step90-cuda-vs-viewer-pixel-compare-v1",
        "pixel": list(pixel),
        "imageName": cli.image_name,
        "cudaReferencePngRgb": cuda_ref_rgb,
        "cudaRenderTensorRgb": cuda_render_rgb,
        "cudaDebugRenderPng": str(debug_render_png_path) if debug_render_png_path is not None else None,
        "cudaDebugVsReference": debug_render_compare,
        "viewerFramebufferReadbackRgb": viewer_framebuffer_rgb,
        "rgbDeltaCudaReferenceMinusViewer": [
            cuda_ref_rgb[i] - viewer_framebuffer_rgb[i] for i in range(3)
        ] if cuda_ref_rgb and viewer_framebuffer_rgb else None,
        "cudaSummary": {k: cuda_debug.get(k) for k in [
            "valid", "tileId", "tile", "range", "tilePayloadCount", "recordedCount",
            "contributorCounter", "lastContributor", "contributionCount",
            "powerSkipCount", "alphaSkipCount", "earlyOutTriggered",
            "earlyOutAtLocalOrder", "finalT", "finalRgb"
        ]},
        "viewerSummary": {
            "tileId": viewer_pixel.get("tileId") if viewer_pixel else None,
            "tile": viewer_pixel.get("tile") if viewer_pixel else None,
            "tilePayloadCount": viewer_pixel.get("tilePayloadCount") if viewer_pixel else None,
            "contributorCounter": viewer_acc.get("contributorCounter"),
            "lastContributingLocalOrder": viewer_acc.get("lastContributingLocalOrder"),
            "contributionCount": viewer_acc.get("contributionCount"),
            "powerSkipCount": viewer_acc.get("powerSkipCount"),
            "alphaSkipCount": viewer_acc.get("alphaSkipCount"),
            "earlyOutTriggered": viewer_acc.get("earlyOutTriggered"),
            "earlyOutAtLocalOrder": viewer_acc.get("earlyOutAtLocalOrder"),
            "finalT": viewer_acc.get("finalT"),
            "finalRgb": viewer_acc.get("finalRgb"),
        },
        "firstDivergenceAllEvaluated": first_divergence(cuda_debug.get("entries", []), viewer_entries),
        "firstDivergenceContributorsOrEarlyOut": first_divergence(
            [e for e in cuda_debug.get("entries", []) if e.get("contributed") or e.get("skipReason") == "early-out-testT-below-0.0001"],
            [e for e in viewer_entries if e.get("contributes") or e.get("skipReason") == "early-out-testT-below-0.0001"],
        ),
    }
    compare_output = Path(cli.compare_output_json)
    compare_output.parent.mkdir(parents=True, exist_ok=True)
    compare_output.write_text(json.dumps(compare, indent=2))

    preprocess_trace_output = Path(cli.preprocess_trace_output_json)
    if cli.preprocess_debug_index >= 0:
        target_index = int(cli.preprocess_debug_index)
        cuda_pixel_entry = next((e for e in cuda_debug.get("entries", []) if int(e.get("gaussianIndex", -1)) == target_index), None)
        viewer_tile_entry = next((e for e in viewer_entries if int(e.get("originalSplatIndex", -1)) == target_index), None)
        viewer_association_entry = None
        association_path = Path(cli.viewer_association_json)
        if association_path.exists():
            association_data = json.loads(association_path.read_text())
            viewer_association_entry = next(
                (e for e in association_data.get("entries", []) if int(e.get("originalSplatIndex", -1)) == target_index),
                None,
            )
        viewer_cov_after = cov2d_from_conic(viewer_tile_entry.get("conic", [])) if viewer_tile_entry else None
        cuda_cov_after = cuda_preprocess_debug.get("cov2DAfterOffset") if cuda_preprocess_debug.get("valid") else None
        difference_summary: Dict[str, Any] = {}
        if cuda_preprocess_debug.get("valid") and viewer_tile_entry:
            difference_summary = {
                "viewerMinusCuda": {
                    "centerPx": [
                        viewer_tile_entry["centerPx"][0] - cuda_preprocess_debug["centerPx"][0],
                        viewer_tile_entry["centerPx"][1] - cuda_preprocess_debug["centerPx"][1],
                    ],
                    "depth": viewer_tile_entry.get("depth") - cuda_preprocess_debug.get("depth"),
                    "opacity": viewer_tile_entry.get("opacity") - cuda_preprocess_debug.get("opacityAfterTime"),
                    "conic": [
                        viewer_tile_entry["conic"][i] - cuda_preprocess_debug["conic"][i]
                        for i in range(3)
                    ],
                    "cov2DAfterOffset": [
                        [
                            viewer_cov_after[r][c] - cuda_cov_after[r][c]
                            for c in range(2)
                        ]
                        for r in range(2)
                    ] if viewer_cov_after and cuda_cov_after else None,
                    "power": viewer_tile_entry.get("power") - cuda_preprocess_debug.get("power"),
                    "alpha": viewer_tile_entry.get("computedAlpha") - cuda_preprocess_debug.get("alpha"),
                }
            }
        preprocess_trace = {
            "schemaVersion": "step90-cuda-preprocess-vs-viewer-trace-v1",
            "pixel": list(pixel),
            "imageName": cli.image_name,
            "targetIndex": target_index,
            "inputs": {
                "cudaReferencePng": cli.cuda_reference_png,
                "viewerDebugJson": cli.viewer_debug_json,
                "viewerAssociationJson": cli.viewer_association_json,
                "cudaOutputJson": str(cuda_output),
                "compareOutputJson": str(compare_output),
            },
            "cudaPreprocessTrace": cuda_preprocess_debug,
            "cudaPixelDebugEntry": cuda_pixel_entry,
            "viewerLiveTileEntry": viewer_tile_entry,
            "viewerAssociationEntry": viewer_association_entry,
            "viewerCov2DAfterOffsetInferredFromConic": viewer_cov_after,
            "viewerEllipse": ellipse_summary_from_cov2d(viewer_cov_after) if viewer_cov_after else None,
            "differenceSummary": difference_summary,
        }
        preprocess_trace_output.parent.mkdir(parents=True, exist_ok=True)
        preprocess_trace_output.write_text(json.dumps(preprocess_trace, indent=2))
    print(json.dumps({
        "cudaOutputJson": str(cuda_output),
        "compareOutputJson": str(compare_output),
        "preprocessTraceOutputJson": str(preprocess_trace_output) if cli.preprocess_debug_index >= 0 else None,
        "cudaSummary": compare["cudaSummary"],
        "viewerSummary": compare["viewerSummary"],
        "rgbDeltaCudaReferenceMinusViewer": compare["rgbDeltaCudaReferenceMinusViewer"],
        "firstDivergenceContributorsOrEarlyOut": compare["firstDivergenceContributorsOrEarlyOut"],
        "cudaDebugVsReference": debug_render_compare,
    }, indent=2))


if __name__ == "__main__":
    main()
