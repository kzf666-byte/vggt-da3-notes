"""VGGT 本地推理小脚本 —— 记录显存/耗时/输出。
用法: python run_case.py <图片目录> <输出目录> [--n 帧数] [--res 518] [--no-glb]
"""
import os, sys, time, json, argparse, glob
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

p = argparse.ArgumentParser()
p.add_argument("img_dir")
p.add_argument("out_dir")
p.add_argument("--n", type=int, default=0, help="取前 N 张, 0=全部")
p.add_argument("--ckpt", default=r"D:\VGGT\model.pt")
p.add_argument("--no-glb", action="store_true")
a = p.parse_args()

os.makedirs(a.out_dir, exist_ok=True)

paths = sorted(glob.glob(os.path.join(a.img_dir, "*.png")) +
               glob.glob(os.path.join(a.img_dir, "*.jpg")))
if a.n:
    paths = paths[:a.n]
assert paths, f"没找到图片: {a.img_dir}"
print(f"[in] {len(paths)} 张图, 第一张原始尺寸 {Image.open(paths[0]).size}")

device = "cuda"
dtype = torch.bfloat16
print("[1/4] load weights ...")
t0 = time.time()
model = VGGT()
model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
model.eval().to(device)
n_param = sum(x.numel() for x in model.parameters()) / 1e6
print(f"      参数量 {n_param:.1f}M, 权重加载 {time.time()-t0:.1f}s")

images = load_and_preprocess_images(paths).to(device)
print(f"[2/4] preprocessed tensor {tuple(images.shape)}")   # (S,3,H,W)

w_gb = torch.cuda.memory_allocated() / 1024**3   # 光权重就占这么多(默认 fp32)
print(f"       [仅权重] {w_gb:.2f} GB")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
    imgs = images[None]                      # (1,S,3,H,W)
    agg, ps_idx = model.aggregator(imgs)
    pose_enc = model.camera_head(agg)[-1]
    ext, intr = pose_encoding_to_extri_intri(pose_enc, imgs.shape[-2:])
    depth, depth_conf = model.depth_head(agg, imgs, ps_idx)
    pts, pts_conf = model.point_head(agg, imgs, ps_idx)
torch.cuda.synchronize()
dt = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"[3/4] forward {dt:.2f}s, 峰值显存 {peak:.2f} GB")

ext = ext.squeeze(0).cpu().numpy()        # (S,3,4) w2c
intr = intr.squeeze(0).cpu().numpy()      # (S,3,3)
depth = depth.squeeze(0).float().cpu().numpy()          # (S,H,W)
depth_conf = depth_conf.squeeze(0).float().cpu().numpy()
print(f"       extrinsics {ext.shape}  intrinsics {intr.shape}  depth {depth.shape}")

# 从深度+相机反投影的 3D 点 (通常比 point head 准)
wpts = unproject_depth_map_to_point_map(depth, ext, intr)
np.savez_compressed(os.path.join(a.out_dir, "pred.npz"),
                    depth=depth, depth_conf=depth_conf,
                    extrinsic=ext, intrinsic=intr,
                    world_points=wpts, images=images.cpu().numpy())

summary = dict(images=len(paths), resolution=[int(images.shape[2]), int(images.shape[3])],
               params_M=round(n_param, 1), forward_s=round(dt, 2),
               weights_GB=round(w_gb, 2),
               activate_GB=round(peak - w_gb, 2),
               peak_vram_GB=round(peak, 2),
               depth_shape=list(depth.shape),
               depth_min=float(depth.min()), depth_max=float(depth.max()),
               depth_mean=float(depth.mean()),
               conf_mean=float(depth_conf.mean()))

# 相机中心 (c2w 的平移)
c2w = np.concatenate([np.linalg.inv(np.concatenate([ext[i], [[0, 0, 0, 1]]], 0))[None]
                      for i in range(ext.shape[0])], 0)
cams = c2w[:, :3, 3]
summary["cam_centers"] = cams.round(3).tolist()
summary["cam_center_span"] = (cams.max(0) - cams.min(0)).round(3).tolist()
summary["intrinsic_fx"] = [round(float(intr[i][0, 0]), 1) for i in range(len(intr))]

# 深度可视化
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = depth.shape[0]
K = min(S, 4)
fig, axes = plt.subplots(2, K, figsize=(3 * K, 6), squeeze=False)
for i in range(K):
    axes[0, i].imshow(images[i].permute(1, 2, 0).cpu().numpy())
    axes[0, i].set_title(f"in {i}", fontsize=8)
    axes[1, i].imshow(np.squeeze(depth[i]), cmap="turbo")
    axes[1, i].set_title(f"depth {np.squeeze(depth[i]).min():.1f}~{np.squeeze(depth[i]).max():.1f}",
                         fontsize=8)
    for r in (0, 1):
        axes[r, i].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(a.out_dir, "depth_grid.png"), dpi=90)
plt.close()

if not a.no_glb:
    print("[4/4] export glb ...")
    t0 = time.time()
    import trimesh
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from visual_util import predictions_to_glb
    dc = depth_conf[..., 0] if (depth_conf.ndim == wpts.ndim and depth_conf.shape[-1] == 1) else depth_conf
    preds = {"world_points": wpts, "world_points_conf": dc,
             "images": images.permute(0, 2, 3, 1).cpu().numpy(), "extrinsic": ext}
    scene = predictions_to_glb(preds, conf_thres=50.0, show_cam=True,
                               target_dir=a.out_dir, prediction_mode="Predicted Pointmap")
    glb = os.path.join(a.out_dir, "scene.glb")
    scene.export(glb)
    summary["glb_MB"] = round(os.path.getsize(glb) / 1024**2, 1)
    summary["glb_s"] = round(time.time() - t0, 1)

json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
print("=== SUMMARY ===")
print(json.dumps({k: v for k, v in summary.items()
                  if k not in ("cam_centers", "intrinsic_fx")}, indent=1, ensure_ascii=False))
