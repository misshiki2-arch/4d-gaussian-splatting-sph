#!/usr/bin/env python3
"""Render camera-name-preserving CUDA references from a 4DGS checkpoint.

This tool exists to bridge the gap between training_report()'s legacy
`000_render.png`-style dumps and downstream comparisons that need to know the
source image / frame / time / pose for each rendered image.
"""

from __future__ import annotations

import argparse
import ast
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


def build_gaussian_model(args: SimpleNamespace, ckpt_path: Path) -> GaussianModel:
    gm = GaussianModel(
        sh_degree=int(args.sh_degree),
        gaussian_dim=int(args.gaussian_dim),
        time_duration=getattr(args, "time_duration", None),
        rot_4d=bool(args.rot_4d),
        force_sh_3d=bool(getattr(args, "force_sh_3d", False)),
        sh_degree_t=int(getattr(args, "sh_degree_t", 0)),
        prefilter_var=float(getattr(args, "prefilter_var", -1.0)),
    )
    model_args, _ = torch.load(ckpt_path, map_location="cuda")
    gm.restore(model_args, training_args=None)
    return gm


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
    if not cli.reuse_legacy_training_report:
        from gaussian_renderer import render
        gm = build_gaussian_model(args, ckpt_path)
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

        legacy_render_path = legacy_dir / f"{local_idx:03d}_render.png"
        legacy_gt_path = legacy_dir / f"{local_idx:03d}_gt.png"
        legacy_alpha_path = legacy_dir / f"{local_idx:03d}_alpha.png"
        legacy_depth_path = legacy_dir / f"{local_idx:03d}_depth.png"

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
        index_records.append({
            "image_name": image_name,
            "frame_number": meta["frame_number"],
            "view_id": meta["view_id"],
            "timestamp": meta["timestamp"],
            "meta_path": str(meta_path),
            "render_path": str(render_path),
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
