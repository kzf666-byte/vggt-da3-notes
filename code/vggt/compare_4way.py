# -*- coding: utf-8 -*-
"""四路点云对比: VGGT / DA3  x  单图 / 六图。

统一从导出的 GLB 里抠点 —— 也就是你在 Blender 里实际看到的那份东西，
而不是各个模型内部的中间张量，这样四路的口径才一致。

四路各自归一化到单位球再比形状（两个模型的绝对尺度不是一回事，
尺度差异单独用数字报出来，不混进图里）。

用法: python compare_4way.py
"""
import os

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = [
    ("VGGT", "6 view", r"D:\VGGT\pointcloud_point_6view.glb"),
    ("VGGT", "1 view", r"D:\VGGT\pointcloud_point.glb"),
    ("DA3", "6 view", r"D:\DA3\runs\cli_6view\scene.glb"),
    ("DA3", "1 view", r"D:\DA3\runs\cli_wrapper_test\scene.glb"),
]
OUT = r"D:\VGGT\runs\compare_4way.png"
MAX_PTS = 80000


def load_points(path):
    """把 GLB 里的点云抠出来。相机锥是 Trimesh, 不要。返回全长点云, 不抽稀。"""
    scene = trimesh.load(path)
    P, C = [], []
    for g in scene.geometry.values():
        v = getattr(g, "vertices", None)
        c = getattr(g, "colors", None)
        if v is None or c is None or not len(v):
            continue
        if not isinstance(g, trimesh.points.PointCloud):   # 跳过相机线框
            continue
        P.append(np.asarray(v, np.float64))
        C.append(np.asarray(c, np.float64)[:, :3] / 255.0)
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(P), np.concatenate(C)


def brighten(C, gain=1.9, gamma=0.65):
    """原场景是深色背景, 不提亮画出来基本全黑, 什么也看不出来。"""
    return np.clip(np.power(np.clip(C, 0, 1), gamma) * gain, 0, 1)



def normalize(P):
    """平移到中位数、按 95 分位半径缩放到单位球。用 95 分位是为了不被离群点带偏。"""
    P = P[np.isfinite(P).all(1)]
    Q = P - np.median(P, 0)
    r = np.percentile(np.linalg.norm(Q, axis=1), 95)
    return Q / max(r, 1e-9)


def render(ax, P, C, elev_deg, azim_deg, title="", s=0.35):
    """自己投影: 视线绕 y 轴 azim, 抬 elev。远的先画(画家算法)。"""
    Q = P - P.mean(0)
    a, e = np.deg2rad(azim_deg), np.deg2rad(elev_deg)
    fwd = np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
    right = np.cross(np.array([0, 1, 0]), fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    Qc = Q @ np.stack([right, up, fwd]).T
    z = Qc[:, 2]
    scale = np.percentile(np.abs(Qc[:, :2]), 92) * 1.6
    px = Qc[:, 0] / (z + scale) * 0.9
    py = Qc[:, 1] / (z + scale) * 0.9
    order = np.argsort(z)
    ax.scatter(px[order], py[order], c=C[order], s=s, marker=".",
               linewidths=0, rasterized=True)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("black")
    if title:
        ax.set_title(title, fontsize=13, color="black")


# ---------------------------------------------------------------- 载入 + 统计
# 统计一律用全长点云, 抽稀只用于画图, 否则点数/包围盒都不可比。
loaded = []
print("%-5s %-7s %8s  %-22s %8s %8s" % ("模型", "视角", "点数", "包围盒(x,y,z)", "对角线", "glb"))
for model, nview, path in SRC:
    if not os.path.exists(path):
        raise SystemExit("缺文件: %s" % path)
    P, C = load_points(path)
    fin = P[np.isfinite(P).all(1)]
    ext = (fin.max(0) - fin.min(0)) if len(fin) else np.zeros(3)
    print("%-5s %-7s %8d  %.3f x %.3f x %.3f %8.3f %7.1fMB"
          % (model, nview, len(P), ext[0], ext[1], ext[2],
             float(np.linalg.norm(ext)), os.path.getsize(path) / 1024**2))

    rng = np.random.default_rng(0)
    if len(P) > MAX_PTS:                               # 抽稀只为画图
        idx = rng.choice(len(P), MAX_PTS, replace=False)
        P, C = P[idx], C[idx]
    loaded.append((normalize(P), brighten(C)))

# ---------------------------------------------------------------- 出图
views = [(12, 0), (12, 90), (8, 200), (75, 45)]
fig, axes = plt.subplots(len(views), len(SRC),
                         figsize=(3.6 * len(SRC), 3.9 * len(views)))
fig.patch.set_facecolor("black")
for r, vi in enumerate(views):
    for c, (model, nview, _) in enumerate(SRC):
        P, C = loaded[c]
        render(axes[r, c], P, C, *vi,
               title="%s  %s" % (model, nview) if r == 0 else "")
        if c == 0:
            axes[r, c].set_ylabel("", fontsize=1)
plt.tight_layout(pad=0.3)
plt.savefig(OUT, dpi=105, facecolor="black")
print("\n出图:", OUT)
