import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Load and preprocess example images (replace with your own image paths)
# 每个路径前面必须加 r！否则 \a \t \n \b 这些会被当成转义符吃掉，
# 路径悄悄变样还不报错（\V 这种会警告，\a 这种连警告都没有）。
image_names = [
    r"D:\VGGT\a7edfbe0bacdad97277cc330f3eeeea9.jpg",
]
images = load_and_preprocess_images(image_names).to(device)

model = VGGT()
model.load_state_dict(torch.load(r"D:\VGGT\model.pt", map_location="cpu"))
model.eval().to(device)
  
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)

# 把结果打出来看看。不打印的话程序会安安静静跑完，你会以为它没跑。
# 注意：不是每个键都是张量，pose_enc_list 是个 list，没有 .shape，所以要分开处理。
print("模型吐出来的东西：")
for k, v in predictions.items():
    if hasattr(v, "shape"):
        print(f"  {k:20s} {tuple(v.shape)}")
    else:
        print(f"  {k:20s} {type(v).__name__}, 长度 {len(v)}")