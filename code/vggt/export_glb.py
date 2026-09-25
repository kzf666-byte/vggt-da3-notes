"""从 run_case.py 存下的 pred.npz 导出彩色点云 glb(带相机线框)。
用法: python export_glb.py <pred.npz> <输出目录>
"""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visual_util import predictions_to_glb

npz_path, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
d = np.load(npz_path)

wpts = d["world_points"]
conf = d["depth_conf"]
if conf.ndim == wpts.ndim:            # (S,H,W,1) -> (S,H,W)
    conf = conf[..., 0]
imgs = d["images"]                     # (S,3,H,W)
if imgs.shape[1] == 3:
    imgs = np.transpose(imgs, (0, 2, 3, 1))
ext = d["extrinsic"]

preds = {"world_points": wpts.astype(np.float32),
         "world_points_conf": conf.astype(np.float32),
         "images": imgs,
         "extrinsic": ext.astype(np.float32)}

t0 = time.time()
scene = predictions_to_glb(preds, conf_thres=50.0, show_cam=True,
                           target_dir=None, prediction_mode="Predicted Pointmap")
glb = os.path.join(out_dir, "scene.glb")
scene.export(glb)
print(f"导出 {glb}  {os.path.getsize(glb)/1024**2:.1f} MB, {time.time()-t0:.1f}s")
