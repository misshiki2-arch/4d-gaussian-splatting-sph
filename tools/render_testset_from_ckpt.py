import os, sys, argparse
import yaml
import torch
import numpy as np
from PIL import Image
from types import SimpleNamespace

# repo root を import パスに入れる
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import cameraList_from_camInfos
from gaussian_renderer import render

def flatten_cfg(cfg: dict) -> dict:
    out = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            out.update(v)
        else:
            out[k] = v
    return out

def make_args(cfg_flat: dict) -> SimpleNamespace:
    d = dict(cfg_flat)

    # ---- 必須 or 安全デフォルト ----
    d.setdefault("images", "images")
    d.setdefault("eval", True)
    d.setdefault("extension", ".png")
    d.setdefault("frame_ratio", 1)
    d.setdefault("dataloader", True)      # 画像全読み込みを避ける
    d.setdefault("data_device", "cuda")   # camera_utils が参照
    d.setdefault("white_background", False)

    # 4D関連（yamlに無いときの保険）
    d.setdefault("gaussian_dim", 4)
    d.setdefault("rot_4d", True)
    d.setdefault("force_sh_3d", False)
    d.setdefault("sh_degree", 3)
    d.setdefault("sh_degree_t", 0)
    d.setdefault("prefilter_var", -1.0)

    # Pipelineが参照しうるもの（render()内で pipe.debug 等を使う）
    d.setdefault("debug", False)
    d.setdefault("compute_cov3D_python", False)
    d.setdefault("convert_SHs_python", False)
    d.setdefault("env_map_res", 0)

    return SimpleNamespace(**d)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--which", choices=["test","train"], default="test")
    args_cli = ap.parse_args()

    cfg = yaml.safe_load(open(args_cli.config, "r"))
    a = make_args(flatten_cfg(cfg))

    # ---- GaussianModel を作って restore（これが正規のchkpnt形式）----
    gm = GaussianModel(
        sh_degree=int(a.sh_degree),
        gaussian_dim=int(a.gaussian_dim),
        time_duration=getattr(a, "time_duration", None),
        rot_4d=bool(a.rot_4d),
        force_sh_3d=bool(getattr(a, "force_sh_3d", False)),
        sh_degree_t=int(getattr(a, "sh_degree_t", 0)),
        prefilter_var=float(getattr(a, "prefilter_var", -1.0)),
    )

    model_args, it = torch.load(args_cli.ckpt, map_location="cuda")
    gm.restore(model_args, training_args=None)  # capture形式を正しく復元 :contentReference[oaicite:1]{index=1}

    # ---- transformsから CameraInfo を読み、Camera を生成（Sceneは作らない）----
    # readCamerasFromTransforms は timestamp を CameraInfo.timestamp に入れる :contentReference[oaicite:2]{index=2}
    if args_cli.which == "test":
        cam_infos = readCamerasFromTransforms(a.source_path, "transforms_test.json",
                                              a.white_background, extension=a.extension,
                                              time_duration=getattr(a, "time_duration", None),
                                              frame_ratio=int(a.frame_ratio),
                                              dataloader=bool(a.dataloader))
    else:
        cam_infos = readCamerasFromTransforms(a.source_path, "transforms_train.json",
                                              a.white_background, extension=a.extension,
                                              time_duration=getattr(a, "time_duration", None),
                                              frame_ratio=int(a.frame_ratio),
                                              dataloader=bool(a.dataloader))

    cameras = cameraList_from_camInfos(cam_infos, resolution_scale=1.0, args=a)

    bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda") if bool(a.white_background) \
         else torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    os.makedirs(args_cli.out_dir, exist_ok=True)

    for i, cam in enumerate(cameras):
        out = render(cam, gm, a, bg)
        img = out["render"].detach().clamp(0,1).permute(1,2,0).cpu().numpy()
        Image.fromarray((img*255).astype(np.uint8)).save(os.path.join(args_cli.out_dir, f"{args_cli.which}_{i:05d}.png"))

    print("[DONE] wrote", len(cameras), "images to", args_cli.out_dir)

if __name__ == "__main__":
    main()
