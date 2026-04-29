# tools/export_3dply_sequence_from_4dckpt.py
# 目的:
#   4DGSチェックポイント（chkpnt*.pth）から、時刻t0ごとの「3D断面PLY」を生成する。
#   SuperSplat は time を評価しないため、1時刻=1PLY でアニメとして扱う想定。
#
# 依存:
#   pip install plyfile numpy

import argparse
import os
import numpy as np
import torch
from plyfile import PlyData, PlyElement

def _to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def write_ply(path, xyz, f_dc, f_rest_flat, opacity, scaling, rotation):
    N = xyz.shape[0]
    fields = []
    fields += [("x","f4", xyz[:,0]), ("y","f4", xyz[:,1]), ("z","f4", xyz[:,2])]
    fields += [("f_dc_0","f4", f_dc[:,0]), ("f_dc_1","f4", f_dc[:,1]), ("f_dc_2","f4", f_dc[:,2])]
    for i in range(f_rest_flat.shape[1]):
        fields.append((f"f_rest_{i}","f4", f_rest_flat[:,i]))
    fields.append(("opacity","f4", opacity.astype(np.float32)))
    fields += [("scale_0","f4", scaling[:,0]), ("scale_1","f4", scaling[:,1]), ("scale_2","f4", scaling[:,2])]
    fields += [("rot_0","f4", rotation[:,0]), ("rot_1","f4", rotation[:,1]), ("rot_2","f4", rotation[:,2]), ("rot_3","f4", rotation[:,3])]

    dtype = [(name, dt) for (name, dt, _) in fields]
    arr = np.empty(N, dtype=dtype)
    for (name, _, col) in fields:
        arr[name] = col

    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="chkpnt_best.pth etc")
    ap.add_argument("--out_dir", required=True, help="output directory for sequence ply")
    ap.add_argument("--start", type=int, default=35)
    ap.add_argument("--end", type=int, default=200)     # 例：35..200
    ap.add_argument("--fps", type=float, default=5.0)   # time=(frame-start)/fps

    # ---- 時間スライスの強さ（靄対策の肝）----
    # gate = exp(-0.5*((t - t0)/sigma)^2)
    # sigma は「時間方向の広がり」→ ここを小さくすると靄は減るが穴が空きやすい
    ap.add_argument("--sigma_sec", type=float, default=0.25, help="time gaussian sigma [sec]")

    # 近傍以外を完全に捨てる（速度・靄対策）
    ap.add_argument("--hard_window_sec", type=float, default=1.0, help="keep only |t-t0|<=window [sec]")

    # さらに靄を減らすためのフィルタ
    ap.add_argument("--min_opacity_after_gate", type=float, default=0.01, help="drop points with gated opacity < this")
    ap.add_argument("--max_scale", type=float, default=1e9, help="optional clamp (very large scales cause fog)")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model_args, it = torch.load(args.ckpt, map_location="cpu")
    if len(model_args) != 19:
        raise RuntimeError(f"Expected 4D checkpoint (len=19), got len={len(model_args)}")

    (active_sh_degree,
     xyz, f_dc, f_rest,
     scaling, rotation, opacity,
     max_radii2D, xyz_grad_accum, t_grad_accum, denom,
     opt_state, spatial_lr_scale,
     t, scaling_t, rotation_r, rot_4d, env_map, active_sh_degree_t) = model_args

    xyz = _to_np(xyz).astype(np.float32)               # [N,3]
    scaling = _to_np(scaling).astype(np.float32)       # [N,3]
    rotation = _to_np(rotation).astype(np.float32)     # [N,4]
    opacity = _to_np(opacity).astype(np.float32).reshape(-1)  # [N] (logitの可能性あり)

    # f_dc: [N,1,3] -> [N,3]
    f_dc = _to_np(f_dc).astype(np.float32)
    f_dc = f_dc[:, 0, :] if f_dc.ndim == 3 else f_dc

    # f_rest: [N,K,3] -> [N,K*3]
    f_rest = _to_np(f_rest).astype(np.float32)
    f_rest_flat = f_rest.reshape((f_rest.shape[0], -1)) if f_rest.ndim == 3 else (f_rest if f_rest.ndim == 2 else np.zeros((xyz.shape[0],0),np.float32))

    t = _to_np(t).astype(np.float32).reshape(-1)  # [N]
    # scaling_t は学習で使うが、SuperSplatでは無視されるので、ここでは gate だけに使う
    # （スクリプトでは固定sigma_secを使用）

    # 追加の靄対策：異常に大きいスケールを落とす
    if np.isfinite(args.max_scale) and args.max_scale < 1e9:
        smax = np.max(scaling, axis=1)
        keep_scale = (smax <= args.max_scale)
    else:
        keep_scale = np.ones_like(t, dtype=bool)

    for fr in range(args.start, args.end + 1):
        t0 = (fr - args.start) / args.fps

        dt = (t - t0)
        # hard window
        keep = keep_scale & (np.abs(dt) <= args.hard_window_sec)

        if not np.any(keep):
            print(f"[WARN] frame {fr:06d}: kept=0")
            continue

        dt_k = dt[keep]
        gate = np.exp(-0.5 * (dt_k / args.sigma_sec) ** 2).astype(np.float32)

        # opacityはlogitの可能性があるが、相対的な靄抑制には gate を掛けるだけでも効く
        op_k = opacity[keep] * gate

        keep2 = (op_k >= args.min_opacity_after_gate)
        if not np.any(keep2):
            print(f"[WARN] frame {fr:06d}: kept after opacity=0")
            continue

        idx = np.where(keep)[0][keep2]

        out_path = os.path.join(args.out_dir, f"point_cloud_{fr:06d}.ply")
        write_ply(
            out_path,
            xyz[idx],
            f_dc[idx],
            f_rest_flat[idx],
            op_k[keep2],
            scaling[idx],
            rotation[idx]
        )
        print(f"[OK] {fr:06d}: t0={t0:.3f} kept={len(idx)} -> {out_path}")

if __name__ == "__main__":
    main()
