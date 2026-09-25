# -*- coding: utf-8 -*-
"""run_vggt.py 的推理 + 点云 GLB 导出（不动 run_vggt.py 本身）。

用法:
    python run_vggt_glb.py                  # 两路都导：point head 与 depth 反投影
    python run_vggt_glb.py --mode point
    python run_vggt_glb.py --mode depth --conf 30
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# 与 run_vggt.py 完全一致的输入与权重
# 这里必须和 run_vggt.py 的 image_names 保持一致（当前是单图）。
IMAGE_NAMES = [
    r"D:\VGGT\a7edfbe0bacdad97277cc330f3eeeea9.jpg",
]
CKPT = r"D:\VGGT\model.pt"

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["both", "point", "depth"], default="both",
                help="point=点图头, depth=深度反投影, both=两路都导")
ap.add_argument("--conf", type=float, default=50.0, help="滤掉置信度最低的百分之几")
ap.add_argument("--out", default=r"D:\VGGT", help="GLB 输出目录")
ap.add_argument("--no-cam", action="store_true", help="不画相机锥")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# ---------------------------------------------------------------- 1) 推理
images = load_and_preprocess_images(IMAGE_NAMES).to(device)
model = VGGT()
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval().to(device)

t0 = time.time()
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)          # (S,3,H,W) -> 内部补 batch 维
torch.cuda.synchronize()
print("[1/2] forward %.2fs, 峰值显存 %.2f GB"
      % (time.time() - t0, torch.cuda.max_memory_allocated() / 1024**3))

S, _, H, W = images.shape
ext, intr = pose_encoding_to_extri_intri(predictions["pose_enc"], (H, W))

img_np = images.permute(0, 2, 3, 1).float().cpu().numpy()   # (S,H,W,3) 0~1
ext_np = ext.squeeze(0).float().cpu().numpy()               # (S,3,4) world->cam
intr_np = intr.squeeze(0).float().cpu().numpy()             # (S,3,3)

# ---------------------------------------------------------------- 2) 两路点云
branches = {}
if a.mode in ("both", "point"):
    branches["point"] = (
        predictions["world_points"].squeeze(0).float().cpu().numpy(),
        predictions["world_points_conf"].squeeze(0).float().cpu().numpy(),
    )
if a.mode in ("both", "depth"):
    depth = predictions["depth"].squeeze(0).float().cpu().numpy()       # (S,H,W,1)
    dconf = predictions["depth_conf"].squeeze(0).float().cpu().numpy()  # (S,H,W)
    wpts = unproject_depth_map_to_point_map(depth, ext_np, intr_np)     # (S,H,W,3)
    branches["depth"] = (wpts, dconf)

np.savez_compressed(os.path.join(a.out, "pred_run_vggt.npz"),
                    images=img_np, extrinsic=ext_np, intrinsic=intr_np,
                    **{k: v[0] for k, v in branches.items()},
                    **{k + "_conf": v[1] for k, v in branches.items()})

# ---------------------------------------------------------------- 3) 导出 GLB
from visual_util import predictions_to_glb

for name, (pts, conf) in branches.items():
    t0 = time.time()
    preds = {
        "world_points": pts,
        "world_points_conf": conf,
        "images": img_np,
        "extrinsic": ext_np,
    }
    scene = predictions_to_glb(preds, conf_thres=a.conf, show_cam=not a.no_cam,
                               target_dir=a.out, prediction_mode="Predicted Pointmap")
    glb = os.path.join(a.out, "pointcloud_%s.glb" % name)
    scene.export(glb)
    print("[2/2] %-5s %d 点 -> %s  (%.1f MB, %.1fs)"
          % (name, pts.reshape(-1, 3).shape[0], glb,
             os.path.getsize(glb) / 1024**2, time.time() - t0))
