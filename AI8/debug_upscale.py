"""
Debug upsampling artifacts: run model on first dataset image and save low-res and upsampled heatmaps using different methods.
"""
import torch
import numpy as np
import cv2
from pathlib import Path
from train_hrnet_facial import FacialKeypointsDataset, HRNetConfig
from model_utils import load_model_safely
import torch.nn.functional as F

out = Path('outputs/debug_upscale')
out.mkdir(parents=True, exist_ok=True)

config = HRNetConfig()
dataset = FacialKeypointsDataset('coco_annotations.json', 'dataset', config, is_train=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model_safely('checkpoints/simple/simple_epoch_9.pth', target_keypoints=config.num_keypoints, device=device)
model.eval()

item = dataset[0]
img_id = item['image_id']
bbox = None
try:
    ann = dataset.annotations[img_id][0]
    bbox = ann.get('bbox')
except Exception:
    bbox = None

# Recreate resized input
image_file = dataset.images[0]['file_name']
image_path = Path('dataset') / image_file
if not image_path.exists():
    image_path = None
    image_name = Path(image_file).name
    for split in ['train','valid','test']:
        alt = Path('dataset') / split / 'images' / image_name
        if alt.exists():
            image_path = alt
            break

if image_path is None:
    print('image not found')
    raise SystemExit

img = cv2.imread(str(image_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if bbox is not None:
    x,y,w,h = bbox
    pad = 0.3
    x = max(0, x - w*pad/2)
    y = max(0, y - h*pad/2)
    w = w*(1+pad)
    h = h*(1+pad)
    x = max(0, min(x, img.shape[1]-1))
    y = max(0, min(y, img.shape[0]-1))
    w = min(w, img.shape[1]-x)
    h = min(h, img.shape[0]-y)
    cropped = img[int(y):int(y+h), int(x):int(x+w)]
    if cropped.size == 0:
        cropped = img.copy()
    resized = cv2.resize(cropped, tuple(config.input_size))
else:
    resized = cv2.resize(img, tuple(config.input_size))

# Prepare input
inp = torch.FloatTensor(resized.transpose(2,0,1) / 255.0).unsqueeze(0)
inp = (inp - torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)) / torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
inp = inp.to(device)

with torch.no_grad():
    pred = model(inp)

if not (isinstance(pred, torch.Tensor) and pred.dim() == 4):
    print('unexpected model output')
    raise SystemExit

pred_np = pred.squeeze(0).cpu().numpy()  # shape (C, h, w)
print('pred shape:', pred_np.shape)

# Save low-res heatmaps as tiled image
C, h, w = pred_np.shape
# Normalize each channel for visualization
lr_vis = np.zeros((h, w*C), dtype=np.float32)
for i in range(C):
    ch = pred_np[i]
    chn = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
    lr_vis[:, i*w:(i+1)*w] = chn
# scale to 0-255
cv2.imwrite(str(out/'lowres_tiled.png'), (lr_vis*255).astype('uint8'))

# Upsample methods
# 1) torch.interpolate bilinear
pred_t = torch.from_numpy(pred_np).unsqueeze(0)
up_bilinear = F.interpolate(pred_t, size=(config.output_size[1], config.output_size[0]), mode='bilinear', align_corners=False).squeeze(0).numpy()
# 2) cv2.resize for each channel (bilinear)
up_cv = np.zeros((C, config.output_size[1], config.output_size[0]), dtype=np.float32)
for k in range(C):
    up_cv[k] = cv2.resize(pred_np[k], (config.output_size[0], config.output_size[1]), interpolation=cv2.INTER_LINEAR)
# 3) nearest neighbor
up_nearest = np.zeros_like(up_cv)
for k in range(C):
    up_nearest[k] = cv2.resize(pred_np[k], (config.output_size[0], config.output_size[1]), interpolation=cv2.INTER_NEAREST)

# Save example channels (1,5,10) as images
cho = [0,4,9,13] if C>13 else list(range(min(C,4)))
for idx in cho:
    def save_channel(arr, name):
        ch = arr[idx]
        chn = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        cv2.imwrite(str(out/f'{name}_ch{idx}.png'), (chn*255).astype('uint8'))
    save_channel(pred_np, 'raw')
    save_channel(up_bilinear, 'torch_bilinear')
    save_channel(up_cv, 'cv_bilinear')
    save_channel(up_nearest, 'nearest')

print('Saved debug images to', out)
