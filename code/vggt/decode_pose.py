# 把 pose_enc 那 9 个数拆开看。不改 run_vggt.py，单独一个文件。
import os
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

device = "cuda"
dtype = torch.bfloat16

names = [
    r"D:\VGGT\EFCF6E12BCF25F3EC135312357B20A0B.jpg",
    r"D:\VGGT\a7edfbe0bacdad97277cc330f3eeeea9.jpg",
    r"D:\VGGT\802b7ad145a24e134ade9b0dcd85fb59.jpg",
]
names = [n for n in names if os.path.exists(n)]  # 不存在的自动跳过
print("用到的图:", [os.path.basename(n) for n in names])

images = load_and_preprocess_images(names).to(device)

model = VGGT()
model.load_state_dict(torch.load(r"D:\VGGT\model.pt", map_location="cpu"))
model.eval().to(device)

with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=dtype):
        pred = model(images)

pe = pred["pose_enc"][0].float().cpu()  # (S, 9)
ext, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])

R2D = 180.0 / 3.141592653589793

print()
print("=== pose_enc 原始 9 个数（每张图一行）===")
for i in range(len(names)):
    print(f"  图{i}: " + "  ".join(f"{v:+8.4f}" for v in pe[i]))

print()
print("=== 拆开读 ===")
for i in range(len(names)):
    T = pe[i, :3]
    q = pe[i, 3:7]
    fh, fw = float(pe[i, 7]), float(pe[i, 8])
    print(f"  --- 图{i} ---")
    print(f"    [0:3] 相机位置 xyz      = {T[0]:+.3f}  {T[1]:+.3f}  {T[2]:+.3f}")
    print(f"    [3:7] 朝向 四元数        = {q[0]:+.3f} {q[1]:+.3f} {q[2]:+.3f} {q[3]:+.3f}")
    print(f"    [7]   上下视场角 FoV_h   = {fh * R2D:6.1f} 度")
    print(f"    [8]   左右视场角 FoV_w   = {fw * R2D:6.1f} 度")
    print(f"    → 换算成焦距 fx/fy      = {intr[0, i, 0, 0]:7.1f} / {intr[0, i, 1, 1]:7.1f} 像素")

print()
print("=== 解码后的外参 extrinsic (3x4) ===")
for i in range(len(names)):
    m = ext[0, i].float().cpu().numpy()
    for r in m:
        print("    " + "  ".join(f"{v:+8.4f}" for v in r))
    print()
