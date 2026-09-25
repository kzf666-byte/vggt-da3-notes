"""把 VGGT 和 DA3 在同一批图上的深度并排画出来。"""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

V = np.load(r"D:\VGGT\runs\room8\pred.npz")
D = np.load(r"D:\DA3\runs\room8_giant\pred.npz")
img_dir = r"C:\Users\zifeng\vggt\examples\room\images"
paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])

vdep = np.squeeze(V["depth"])          # (8,392,518)
ddep = np.squeeze(D["depth"])          # (8,378,504)

rows, cols = 3, 4
fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 3.1 * rows), squeeze=False)
for j in range(cols):
    im = np.asarray(Image.open(paths[j]))
    ax[0, j].imshow(im); ax[0, j].set_title(f"input #{j}  {im.shape[1]}x{im.shape[0]}", fontsize=9)
    ax[1, j].imshow(vdep[j], cmap="turbo"); ax[1, j].set_title(
        f"VGGT depth\n{vdep[j].min():.2f} ~ {vdep[j].max():.2f}", fontsize=9)
    ax[2, j].imshow(ddep[j], cmap="turbo"); ax[2, j].set_title(
        f"DA3 depth\n{ddep[j].min():.2f} ~ {ddep[j].max():.2f}", fontsize=9)
    for r in range(rows):
        ax[r, j].axis("off")
plt.tight_layout()
out = r"D:\VGGT\runs\compare_depth.png"
plt.savefig(out, dpi=95)
print("saved", out)

# 数值对比
print("\n=== 每张图深度范围对比 (单位: 模型自己的尺度) ===")
print(f"{'帧':>3} {'VGGT min~max':>20} {'DA3 min~max':>20}")
for j in range(len(vdep)):
    print(f"{j:>3} {vdep[j].min():>8.3f}~{vdep[j].max():<10.3f} {ddep[j].min():>8.3f}~{ddep[j].max():<10.3f}")
