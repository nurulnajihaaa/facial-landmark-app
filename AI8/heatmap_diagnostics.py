"""
Heatmap diagnostics: analyze concentration and compare raw vs spatial-softmax predictions.

Usage:
  python heatmap_diagnostics.py --coco_file coco_annotations.json --image_root dataset --model checkpoints/finetune/checkpoint_epoch_2.pth --max_images 50
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from model_utils import load_model_safely
from train_hrnet_facial import HRNetConfig


def crop_and_preprocess(img, bbox, input_size, padding=0.3):
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
    resized = cv2.resize(crop, tuple(input_size))
    inp = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std
    inp = inp.transpose(2,0,1)
    inp_tensor = torch.from_numpy(inp).unsqueeze(0).float()
    meta = {'crop_xywh': (x,y,w,h), 'crop_shape': crop.shape, 'resized_shape': resized.shape}
    return inp_tensor, meta


def heatmaps_to_preds_raw(hm, config, meta):
    # hm: torch tensor 1xJxHxW
    hm_np = hm.cpu().numpy()[0]
    J, H, W = hm_np.shape
    preds = np.zeros((J,2), dtype=np.float32)
    for j in range(J):
        ind = np.unravel_index(np.argmax(hm_np[j]), (H,W))
        py, px = ind
        # scale to resized -> crop -> orig
        scale_x = config.input_size[0] / W
        scale_y = config.input_size[1] / H
        x_in = px * scale_x
        y_in = py * scale_y
        x_crop, y_crop, w_crop, h_crop = meta['crop_xywh']
        sx = w_crop / config.input_size[0]
        sy = h_crop / config.input_size[1]
        x_orig = x_crop + x_in * sx
        y_orig = y_crop + y_in * sy
        preds[j,0] = x_orig
        preds[j,1] = y_orig
    return preds


def heatmaps_to_preds_softmax(hm, config, meta):
    # apply spatial softmax per joint
    b, J, H, W = hm.shape
    hm_flat = hm.view(b, J, -1)
    soft = F.softmax(hm_flat, dim=2).cpu().numpy()[0]
    preds = np.zeros((J,2), dtype=np.float32)
    for j in range(J):
        idx = np.argmax(soft[j])
        py = idx // W
        px = idx % W
        scale_x = config.input_size[0] / W
        scale_y = config.input_size[1] / H
        x_in = px * scale_x
        y_in = py * scale_y
        x_crop, y_crop, w_crop, h_crop = meta['crop_xywh']
        sx = w_crop / config.input_size[0]
        sy = h_crop / config.input_size[1]
        x_orig = x_crop + x_in * sx
        y_orig = y_crop + y_in * sy
        preds[j,0] = x_orig
        preds[j,1] = y_orig
    return preds


def per_heatmap_stats(hm_tensor):
    # hm_tensor: torch tensor 1xJxHxW
    hm = hm_tensor.cpu().numpy()[0]
    J, H, W = hm.shape
    stats = []
    for j in range(J):
        h = hm[j]
        mx = float(h.max())
        mean = float(h.mean())
        std = float(h.std())
        # fraction of mass in top 1% pixels
        thresh = np.percentile(h, 99)
        top_frac = float((h >= thresh).sum()) / h.size
        stats.append({'max': mx, 'mean': mean, 'std': std, 'top1%': top_frac})
    return stats


def normalized_mean_error(preds, gts, bbox):
    x,y,w,h = bbox
    diag = np.sqrt(w*w + h*h)
    if diag <= 0:
        diag = 1.0
    errs = np.linalg.norm(preds - gts, axis=1)
    return np.mean(errs) / diag, errs / diag


def pck_score(norm_errs, thresh=0.05):
    return float((norm_errs <= thresh).sum()) / len(norm_errs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_file', type=str, required=True)
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--max_images', type=int, default=50)
    args = parser.parse_args()

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

    processed = 0
    all_stats = []
    raw_nm_es = []
    soft_nm_es = []
    raw_pcks = []
    soft_pcks = []

    for img_id, img_info in images.items():
        if img_id not in img_to_ann:
            continue
        ann = img_to_ann[img_id][0]
        file_name = img_info['file_name']
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
        keypoints = np.array(ann.get('keypoints', [])).reshape(-1,3)[:,:2]
        inp_tensor, meta = crop_and_preprocess(img, bbox, config.input_size)

        with torch.no_grad():
            out = model(inp_tensor)
            if isinstance(out, (list, tuple)):
                hm = out[0]
            else:
                hm = out

        stats = per_heatmap_stats(hm)
        all_stats.append(stats)

        preds_raw = heatmaps_to_preds_raw(hm, config, meta)
        preds_soft = heatmaps_to_preds_softmax(hm, config, meta)

        nme_raw, norm_errs_raw = normalized_mean_error(preds_raw, keypoints, bbox)
        nme_soft, norm_errs_soft = normalized_mean_error(preds_soft, keypoints, bbox)
        raw_nm_es.append(nme_raw)
        soft_nm_es.append(nme_soft)
        raw_pcks.append(pck_score(norm_errs_raw, thresh=0.05))
        soft_pcks.append(pck_score(norm_errs_soft, thresh=0.05))

        processed += 1
        if processed >= args.max_images:
            break

    # Aggregate stats
    J = len(all_stats[0]) if all_stats else 0
    # per-joint aggregated
    per_joint = [{'max':[], 'mean':[], 'std':[], 'top1%':[]} for _ in range(J)]
    for s_list in all_stats:
        for j, s in enumerate(s_list):
            per_joint[j]['max'].append(s['max'])
            per_joint[j]['mean'].append(s['mean'])
            per_joint[j]['std'].append(s['std'])
            per_joint[j]['top1%'].append(s['top1%'])

    print(f"Processed {processed} images")
    print(f"Overall NME raw: mean={np.mean(raw_nm_es):.4f}, median={np.median(raw_nm_es):.4f}, PCK@0.05={np.mean(raw_pcks):.4f}")
    print(f"Overall NME softmax: mean={np.mean(soft_nm_es):.4f}, median={np.median(soft_nm_es):.4f}, PCK@0.05={np.mean(soft_pcks):.4f}")

    # Print per-joint mean stats
    for j in range(J):
        print(f"Joint {j+1}: max={np.mean(per_joint[j]['max']):.3f}, mean={np.mean(per_joint[j]['mean']):.3f}, top1%={np.mean(per_joint[j]['top1%']):.4f}")

if __name__ == '__main__':
    main()
