import glob, os, torch
from depth_anything_3.api import DepthAnything3

MODEL = r"D:\DA3\models\DA3NESTED-GIANT-LARGE-1.1"
IMAGE_DIR = r"D:\DA3\picture"
OUT = r"D:\DA3\output"

device = torch.device("cuda")
print("loading model ...", flush=True)
model = DepthAnything3.from_pretrained(MODEL).to(device)
print("model loaded", flush=True)

images = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.jpeg"))
)
print(f"images: {images}", flush=True)

prediction = model.inference(
    images,
    export_dir=OUT,
    export_format="depth_vis-npz-glb",
)

print("processed_images", prediction.processed_images.shape, flush=True)
print("depth", prediction.depth.shape, flush=True)
print("conf", prediction.conf.shape, flush=True)
print("extrinsics", prediction.extrinsics.shape, flush=True)
print("intrinsics", prediction.intrinsics.shape, flush=True)
print("DONE", flush=True)
