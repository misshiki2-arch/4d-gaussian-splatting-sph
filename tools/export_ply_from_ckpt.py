import argparse
import numpy as np
import torch
from plyfile import PlyData, PlyElement

def _to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def export_ply(ckpt_path: str, out_ply: str):
    data = torch.load(ckpt_path, map_location="cpu")
    if isinstance(data, (list, tuple)) and len(data) == 2:
        model_args, iteration = data
    else:
        raise RuntimeError(f"Unexpected checkpoint format: {type(data)}")

    L = len(model_args)
    if L == 12:
        gaussian_dim = 3
        (active_sh_degree,
         xyz, f_dc, f_rest,
         scaling, rotation, opacity,
         max_radii2D, xyz_grad_accum, denom,
         opt_state, spatial_lr_scale) = model_args
        t = None; scaling_t = None; rotation_r = None; rot_4d = False
    elif L == 19:
        gaussian_dim = 4
        (active_sh_degree,
         xyz, f_dc, f_rest,
         scaling, rotation, opacity,
         max_radii2D, xyz_grad_accum, t_grad_accum, denom,
         opt_state, spatial_lr_scale,
         t, scaling_t, rotation_r, rot_4d, env_map, active_sh_degree_t) = model_args
    else:
        raise RuntimeError(f"Unknown model_args length={L} (expected 12 or 19)")

    xyz = _to_np(xyz).astype(np.float32)           # [N,3]
    scaling = _to_np(scaling).astype(np.float32)   # [N,3]
    rotation = _to_np(rotation).astype(np.float32) # [N,4]
    opacity = _to_np(opacity).astype(np.float32).reshape(-1)  # [N]

    f_dc = _to_np(f_dc).astype(np.float32)
    if f_dc.ndim == 3:
        f_dc = f_dc[:, 0, :]  # [N,1,3] -> [N,3]
    elif f_dc.ndim != 2 or f_dc.shape[1] != 3:
        raise RuntimeError(f"Unexpected f_dc shape: {f_dc.shape}")

    f_rest = _to_np(f_rest).astype(np.float32)
    if f_rest.size == 0:
        f_rest_flat = np.zeros((xyz.shape[0], 0), dtype=np.float32)
    else:
        if f_rest.ndim == 3:
            f_rest_flat = f_rest.reshape((f_rest.shape[0], -1))
        elif f_rest.ndim == 2:
            f_rest_flat = f_rest
        else:
            raise RuntimeError(f"Unexpected f_rest shape: {f_rest.shape}")

    N = xyz.shape[0]
    fields = []

    fields += [("x","f4", xyz[:,0]), ("y","f4", xyz[:,1]), ("z","f4", xyz[:,2])]
    fields += [("f_dc_0","f4", f_dc[:,0]), ("f_dc_1","f4", f_dc[:,1]), ("f_dc_2","f4", f_dc[:,2])]
    for i in range(f_rest_flat.shape[1]):
        fields.append((f"f_rest_{i}","f4", f_rest_flat[:,i]))
    fields.append(("opacity","f4", opacity))
    fields += [("scale_0","f4", scaling[:,0]), ("scale_1","f4", scaling[:,1]), ("scale_2","f4", scaling[:,2])]
    fields += [("rot_0","f4", rotation[:,0]), ("rot_1","f4", rotation[:,1]), ("rot_2","f4", rotation[:,2]), ("rot_3","f4", rotation[:,3])]

    if gaussian_dim == 4:
        t = _to_np(t).astype(np.float32).reshape(-1)
        st = _to_np(scaling_t).astype(np.float32).reshape(-1)
        fields.append(("t","f4", t))
        fields.append(("scale_t","f4", st))
        if rot_4d:
            rr = _to_np(rotation_r).astype(np.float32)
            fields += [("rot_r0","f4", rr[:,0]), ("rot_r1","f4", rr[:,1]), ("rot_r2","f4", rr[:,2]), ("rot_r3","f4", rr[:,3])]

    dtype = [(name, dt) for (name, dt, _) in fields]
    arr = np.empty(N, dtype=dtype)
    for (name, _, col) in fields:
        arr[name] = col

    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(out_ply)
    print(f"[DONE] wrote: {out_ply}  (N={N}, gaussian_dim={gaussian_dim})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    export_ply(args.ckpt, args.out)

if __name__ == "__main__":
    main()
