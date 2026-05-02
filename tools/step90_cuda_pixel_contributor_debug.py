#!/usr/bin/env python3
"""Capture CUDA rasterizer contributor debug for one Step90 pixel."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--cuda-reference-png", default="/home/demo/work/outputs/sph_scene_4dgs/cuda_reference_named/iter_012000/000195_v26_render.png")
    parser.add_argument("--cuda-output-json", default="/home/demo/work/json/step90_cuda_pixel_contributor_debug_655_363.json")
    parser.add_argument("--compare-output-json", default="/home/demo/work/json/step90_cuda_vs_viewer_pixel_655_363_compare.json")
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

    camera_infos = build_camera_infos(args, "training-report-train-sample")
    camera_info = next((item for item in camera_infos if item.image_name == cli.image_name), None)
    if camera_info is None:
        raise RuntimeError(f"Could not find camera image_name={cli.image_name}")
    viewpoint = cameraList_from_camInfos([camera_info], resolution_scale=1.0, args=args)[0].cuda()
    gm = build_gaussian_model(args, ckpt_path)
    bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda") if bool(args.white_background) else torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    render_pkg = render(viewpoint, gm, args, bg)
    cuda_debug = decode_debug_tensor(render_pkg["cuda_pixel_debug"], pixel)
    cuda_debug.update({
        "imageName": cli.image_name,
        "checkpointPath": str(ckpt_path),
        "modelPath": str(model_path),
        "sourcePath": str(source_path),
        "pathRemaps": path_remaps,
    })
    cuda_output = Path(cli.cuda_output_json)
    cuda_output.parent.mkdir(parents=True, exist_ok=True)
    cuda_output.write_text(json.dumps(cuda_debug, indent=2))

    viewer_pixel = load_viewer_pixel(Path(cli.viewer_debug_json), pixel)
    viewer_acc = viewer_pixel.get("accumulation", {}) if viewer_pixel else {}
    viewer_entries = viewer_acc.get("entries", [])
    cuda_ref_rgb = read_png_rgb01(Path(cli.cuda_reference_png), pixel)
    cuda_render_rgb = render_pkg["render"][:, pixel[1], pixel[0]].detach().clamp(0, 1).cpu().tolist()
    viewer_framebuffer_rgb = viewer_pixel.get("framebuffer", {}).get("rgb") if viewer_pixel else None
    compare = {
        "schemaVersion": "step90-cuda-vs-viewer-pixel-compare-v1",
        "pixel": list(pixel),
        "imageName": cli.image_name,
        "cudaReferencePngRgb": cuda_ref_rgb,
        "cudaRenderTensorRgb": cuda_render_rgb,
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
    print(json.dumps({
        "cudaOutputJson": str(cuda_output),
        "compareOutputJson": str(compare_output),
        "cudaSummary": compare["cudaSummary"],
        "viewerSummary": compare["viewerSummary"],
        "rgbDeltaCudaReferenceMinusViewer": compare["rgbDeltaCudaReferenceMinusViewer"],
        "firstDivergenceContributorsOrEarlyOut": compare["firstDivergenceContributorsOrEarlyOut"],
    }, indent=2))


if __name__ == "__main__":
    main()
