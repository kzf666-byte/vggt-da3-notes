"""DA3 本地推理小脚本 —— 记录显存/耗时/输出, 与 VGGT 对齐口径。
用法: python run_case.py <图片目录> <输出目录> <模型名> [--n 帧数]
  --model 例如 DA3-SMALL 或 DA3NESTED-GIANT-LARGE-1.1
"""
import os, sys, glob, time, json, argparse, contextlib, io
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

sys.path.insert(0, r"D:\DA3\src")
from depth_anything_3.api import DepthAnything3

p = argparse.ArgumentParser()
p.add_argument("img_dir")
p.add_argument("out_dir")
p.add_argument("model", help="models 目录下的模型名")
p.add_argument("--n", type=int, default=0)
p.add_argument("--res", type=int, default=504)
a = p.parse_args()

os.makedirs(a.out_dir, exist_ok=True)
paths = sorted(glob.glob(os.path.join(a.img_dir, "*.png")) +
               glob.glob(os.path.join(a.img_dir, "*.jpg")))
if a.n:
    paths = paths[:a.n]
assert paths, f"没找到图片: {a.img_dir}"
print(f"[in] {len(paths)} 张图")

dev = torch.device("cuda")
mpath = os.path.join(r"D:\DA3\models", a.model)
t0 = time.time()
model = DepthAnything3.from_pretrained(mpath).to(dev).eval()
n_param = sum(x.numel() for x in model.parameters()) / 1e6
print(f"[1/3] {a.model} 参数量 {n_param:.0f}M, 加载 {time.time()-t0:.1f}s")

w_gb = torch.cuda.memory_allocated() / 1024**3
print(f"       [仅权重] {w_gb:.2f} GB")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.no_grad():
    pred = model.inference(
        paths,
        export_dir=a.out_dir,
        export_format="depth_vis-npz-glb",
        process_res=a.res,
    )
torch.cuda.synchronize()
dt = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"[2/3] 前向+导出 {dt:.2f}s, 峰值显存 {peak:.2f} GB")

depth = np.asarray(pred.depth)
conf = np.asarray(pred.conf)
ext = np.asarray(pred.extrinsics)
intr = np.asarray(pred.intrinsics)
print(f"       depth {depth.shape}  conf {conf.shape}  ext {ext.shape}  intr {intr.shape}")

np.savez_compressed(os.path.join(a.out_dir, "pred.npz"),
                    depth=depth, conf=conf, extrinsic=ext, intrinsic=intr,
                    images=np.asarray(pred.processed_images))

summary = dict(model=a.model, params_M=round(n_param), images=len(paths),
               process_res=a.res, forward_s=round(dt, 2),
               weights_GB=round(w_gb, 2), activate_GB=round(peak - w_gb, 2),
               peak_vram_GB=round(peak, 2),
               is_metric=bool(pred.is_metric),
               depth_shape=list(depth.shape),
               depth_min=float(depth.min()), depth_max=float(depth.max()),
               depth_mean=float(depth.mean()), conf_mean=float(conf.mean()),
               intrinsic_fx=[round(float(intr[i][0, 0]), 1) for i in range(len(intr))])
json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
print("[3/3] === SUMMARY ===")
print(json.dumps(summary, indent=1, ensure_ascii=False))
