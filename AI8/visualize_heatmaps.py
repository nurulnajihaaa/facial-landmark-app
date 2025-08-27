"""
Visualize predicted heatmaps over cropped inputs.
- Loads model via load_model_safely
- For first N images: crops by GT bbox (training preprocessing), runs model
- Saves: summed heatmap overlay on crop, per-joint heatmap grid

Usage:
    python visualize_heatmaps.py --coco_file coco_annotations.json --image_root dataset --model checkpoints/checkpoint_epoch_99.pth --max_images 20

"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from model_utils import load_model_safely
from train_hrnet_facial import HRNetConfig


def crop_image(img, bbox, padding=0.3):
    x, y, w, h = bbox
    x = max(0, x - w * padding / 2)
    y = max(0, y - h * padding / 2)
    w = w * (1 + padding)
    h = h * (1 + padding)
    x = max(0, min(x, img.shape[1] - 1))
    y = max(0, min(y, img.shape[0] - 1))
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    crop = img[int(y):int(y+h), int(x):int(x+w)].copy()
    return crop, (x,y,w,h)


def preprocess_for_model(crop, input_size):
    resized = cv2.resize(crop, tuple(input_size))
    inp = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std
    inp = inp.transpose(2,0,1)
    inp_tensor = torch.from_numpy(inp).unsqueeze(0).float()
    return inp_tensor, resized


def heatmaps_to_numpy(heatmaps):
    # heatmaps: torch tensor 1 x J x H x W
    hm = heatmaps.cpu().numpy()[0]
    return hm  # shape JxHxW


def save_summed_overlay(out_dir, image_name, crop_rgb, hm_j, config):
    # Sum heatmaps
    summed = np.sum(hm_j, axis=0)
    # normalize
    s = summed - summed.min()
    if s.max() > 0:
        s = (s / s.max() * 255).astype(np.uint8)
    else:
        s = (s*0).astype(np.uint8)
    s_resized = cv2.resize(s, (crop_rgb.shape[1], crop_rgb.shape[0]))
    heatmap_color = cv2.applyColorMap(s_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)
    outp = out_dir / f"{image_name}_summed.png"
    cv2.imwrite(str(outp), overlay)


def save_per_joint_grid(out_dir, image_name, crop_rgb, hm_j, config):
    J, H, W = hm_j.shape
    # create grid: 4 cols
    cols = 4
    rows = (J + cols - 1) // cols
    thumb_h = 128
    thumb_w = 128
    grid = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    for j in range(J):
        r = j // cols
        c = j % cols
        hmap = hm_j[j]
        s = hmap - hmap.min()
        if s.max() > 0:
            s = (s / s.max() * 255).astype(np.uint8)
        else:
            s = (s*0).astype(np.uint8)
        s_resized = cv2.resize(s, (thumb_w, thumb_h))
        heat_color = cv2.applyColorMap(s_resized, cv2.COLORMAP_JET)
        # blend with resized crop thumbnail
        crop_thumb = cv2.resize(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR), (thumb_w, thumb_h))
        blended = cv2.addWeighted(crop_thumb, 0.5, heat_color, 0.5, 0)
        # annotate
        cv2.putText(blended, str(j+1), (6,14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        grid[r*thumb_h:(r+1)*thumb_h, c*thumb_w:(c+1)*thumb_w] = blended
    outp = out_dir / f"{image_name}_per_joint_grid.png"
    cv2.imwrite(str(outp), grid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_file', type=str, required=True)
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--max_images', type=int, default=20)
    parser.add_argument('--out_dir', type=str, default='outputs/diagnostics/heatmaps')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.coco_file, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    images = {img['id']: img for img in coco.get('images', [])}
    anns = coco.get('annotations', [])
    img_to_ann = {}
    for ann in anns:
        img_to_ann.setdefault(ann['image_id'], []).append(ann)

    device = 'cpu'
    model = load_model_safely(args.model, target_keypoints=14, device=device)
    model.eval()
    config = HRNetConfig()

    saved = 0
    for img_id, img_info in images.items():
        if img_id not in img_to_ann:
            continue
        ann = img_to_ann[img_id][0]
        file_name = img_info['file_name']
        # resolve path
        candidates = [Path(args.image_root) / file_name, Path(args.image_root) / Path(file_name).name]
        for split in ['train','valid','test']:
            candidates.append(Path(args.image_root)/split/'images'/Path(file_name).name)
        img_path = None
        for c in candidates:
            if c.exists():
                img_path = c
                break
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bbox = ann.get('bbox')
        crop, crop_box = crop_image(img, bbox)
        inp_tensor, resized = preprocess_for_model(crop, config.input_size)

        with torch.no_grad():
            out = model(inp_tensor)
            if isinstance(out, (tuple, list)):
                heatmaps = out[0]
            else:
                heatmaps = out
        hm_j = heatmaps_to_numpy(heatmaps)

        image_name = Path(file_name).stem
        save_summed_overlay(out_dir, image_name, resized, hm_j, config)
        save_per_joint_grid(out_dir, image_name, resized, hm_j, config)

        saved += 1
        if saved >= args.max_images:
            break

    print(f"Saved heatmap visualizations for {saved} images to {out_dir}")

if __name__ == '__main__':
    main()
