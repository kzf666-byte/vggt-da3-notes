"""点云对比渲染(自己写投影, 按深度排序 = 画家算法)。
比 matplotlib 的 3D scatter 好看, 也快。
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP = 3


def unproject(depth, ext, intr):
    S, h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    pix = np.stack([xs, ys, np.ones_like(xs)], -1).astype(np.float64)
    out = np.zeros((S, h, w, 3), np.float64)
    for i in range(S):
        K_inv = np.linalg.inv(intr[i])
        c2w = np.linalg.inv(np.vstack([ext[i], [0, 0, 0, 1]]))
        cam = (K_inv @ pix.reshape(-1, 3).T).T * depth[i].reshape(-1, 1)
        camh = np.concatenate([cam, np.ones((cam.shape[0], 1))], 1)
        out[i] = (c2w @ camh.T).T[:, :3].reshape(h, w, 3)
    return out


def load(npy, is_vggt):
    d = np.load(npy)
    dep = np.squeeze(d["depth"])
    pts = d["world_points"] if is_vggt else unproject(dep, d["extrinsic"], d["intrinsic"])
    im = d["images"]
    if im.shape[1] == 3:
        im = np.transpose(im, (0, 2, 3, 1))
    if im.dtype != np.uint8:
        im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    return pts, im


def pack(pts, im, step=STEP):
    P = pts[:, ::step, ::step].reshape(-1, 3)
    C = im[:, ::step, ::step].reshape(-1, 3).astype(np.float64) / 255.0
    ok = np.isfinite(P).all(1)
    med = np.median(P[ok], 0)
    dist = np.linalg.norm(P - med, axis=1)
    ok &= dist < np.percentile(dist[ok], 99.0)
    P, C = P[ok], C[ok]
    # 简单光照: 按法线方向没有, 就用深度梯度近似 -> 算了, 直接用颜色, 稍微提亮
    C = np.clip(C * 1.15, 0, 1)
    return P, C


def render(ax, P, C, elev_deg, azim_deg, title, s=0.35):
    """自己投影: 视线绕 y 轴 azim, 抬 elev。"""
    c = P.mean(0)
    Q = P - c
    a, e = np.deg2rad(azim_deg), np.deg2rad(elev_deg)
    # 相机基
    fwd = np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
    right = np.cross(np.array([0, 1, 0]), fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    R = np.stack([right, up, fwd])            # 相机坐标 = R @ 点
    Qc = Q @ R.T
    z = Qc[:, 2]
    scale = np.percentile(np.abs(Qc[:, :2]), 92) * 1.6
    px = Qc[:, 0] / (z + scale) * 0.9
    py = Qc[:, 1] / (z + scale) * 0.9
    order = np.argsort(z)                      # 远的先画
    ax.scatter(px[order], py[order], c=C[order], s=s, marker=".", linewidths=0,
               rasterized=True)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("black")
    ax.set_title(title, fontsize=13, color="black")


VP, VC = pack(*load(r"D:\VGGT\runs\room8\pred.npz", True))
DP, DC = pack(*load(r"D:\DA3\runs\room8_giant\pred.npz", False))
# DA3 尺度不同, 归一化到各自单位球再比形状
DP = (DP - DP.mean(0)) / np.percentile(np.linalg.norm(DP - DP.mean(0), axis=1), 99)
VP = (VP - VP.mean(0)) / np.percentile(np.linalg.norm(VP - VP.mean(0), axis=1), 99)
print("VGGT 点数", len(VP), "  DA3 点数", len(DP))

views = [(15, 0), (15, 90), (10, 200), (80, 45)]
fig, axes = plt.subplots(len(views), 2, figsize=(11, 4.6 * len(views)))
fig.patch.set_facecolor("black")
for r, vi in enumerate(views):
    render(axes[r, 0], VP, VC, *vi, "VGGT-1B (8 imgs)" if r == 0 else "")
    render(axes[r, 1], DP, DC, *vi, "DA3-GIANT (same 8 imgs)" if r == 0 else "")
    axes[r, 0].set_ylabel("")
plt.tight_layout(pad=0.4)
out = r"D:\VGGT\runs\compare_pc.png"
plt.savefig(out, dpi=105, facecolor="black")
print("saved", out)
