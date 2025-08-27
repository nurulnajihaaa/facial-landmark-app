"""
Diagnostic evaluation script
- Loads model via `load_model_safely`
- Uses COCO annotations to crop images with GT bbox (same as training preprocessing)
- Runs model on crop, maps predicted keypoints back to original image coords
- Computes per-image NME (normalized by bbox diagonal) and PCK@0.05
- Saves side-by-side GT vs Pred visualizations to outputs/diagnostics

Usage:
    python evaluate_inference.py --coco_file coco_annotations.json --image_root dataset --model hrnet_w18_small_model_v1.pth --max_images 50

"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from model_utils import load_model_safely, get_model_info
from train_hrnet_facial import HRNetConfig


def crop_and_preprocess(img, bbox, input_size):
    x, y, w, h = bbox
    padding = 0.3
    x = max(0, x - w * padding / 2)
    y = max(0, y - h * padding / 2)
    w = w * (1 + padding)
    h = h * (1 + padding)

    # Ensure inside image
    x = max(0, min(x, img.shape[1] - 1))
    y = max(0, min(y, img.shape[0] - 1))
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)

    crop = img[int(y):int(y+h), int(x):int(x+w)].copy()
    resized = cv2.resize(crop, tuple(input_size))

    # normalize like inference
    inp = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std
    inp = inp.transpose(2,0,1)
    inp_tensor = torch.from_numpy(inp).unsqueeze(0).float()
    meta = {'crop_xywh': (x, y, w, h), 'crop_shape': crop.shape, 'resized_shape': resized.shape}
    return inp_tensor, meta


def heatmaps_to_keypoints(heatmaps, config, meta):
    # heatmaps: torch tensor BxJxHwxW
    batch_size, num_joints, H, W = heatmaps.shape
    hm = heatmaps.cpu().numpy()
    preds = np.zeros((num_joints,2), dtype=np.float32)
    confs = np.zeros((num_joints,), dtype=np.float32)
    for j in range(num_joints):
        h = hm[0,j]
        ind = np.unravel_index(np.argmax(h), h.shape)
        py, px = ind
        conf = h[py,px]
        # scale to resized input
        scale_x = config.input_size[0] / W
        scale_y = config.input_size[1] / H
        x_in = px * scale_x
        y_in = py * scale_y
        # map back to original image coordinates via meta crop
        x_crop, y_crop, w_crop, h_crop = meta['crop_xywh']
        # note: resized image corresponds to crop; scaling factor from resized to crop
        sx = w_crop / config.input_size[0]
        sy = h_crop / config.input_size[1]
        x_orig = x_crop + x_in * sx
        y_orig = y_crop + y_in * sy
        preds[j,0] = x_orig
        preds[j,1] = y_orig
        confs[j] = conf
    return preds, confs


def draw_points(img, points, color=(0,255,0), radius=3, labels=None):
    out = img.copy()
    for i, p in enumerate(points):
        x = int(round(p[0])); y = int(round(p[1]))
        cv2.circle(out, (x,y), radius, color, -1)
        if labels:
            cv2.putText(out, str(i+1), (x+4,y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1)
    return out


def normalized_mean_error(preds, gts, bbox):
    # normalize by bbox diagonal
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
    parser.add_argument('--out_dir', type=str, default='outputs/diagnostics')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.coco_file, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco.get('images', [])}
    anns = coco.get('annotations', [])

    # map image->first annotation (assume single face per image)
    img_to_ann = {}
    for ann in anns:
        img_to_ann.setdefault(ann['image_id'], []).append(ann)

    # load model
    device = 'cpu'
    model = load_model_safely(args.model, target_keypoints=14, device=device)
    model.eval()

    config = HRNetConfig()

    nme_list = []
    pck_list = []

    processed = 0
    for img_id, img_info in images.items():
        if img_id not in img_to_ann:
            continue
        ann = img_to_ann[img_id][0]
        file_name = img_info['file_name']
        # find image file
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

        # gt keypoints
        gt_kps = np.array(ann.get('keypoints',[])).reshape(-1,3)[:,:2]
        bbox = ann.get('bbox')
        inp_tensor, meta = crop_and_preprocess(img, bbox, config.input_size)

        with torch.no_grad():
            out = model(inp_tensor)
            if isinstance(out, tuple) or isinstance(out, list):
                heatmaps = out[0]
            else:
                heatmaps = out
        preds, confs = heatmaps_to_keypoints(heatmaps, config, meta)

        # compute metrics
        nme, norm_errs = normalized_mean_error(preds, gt_kps, bbox)
        pck = pck_score(norm_errs, thresh=0.05)
        nme_list.append(nme)
        pck_list.append(pck)

        # create visualization: left GT, right Pred
        left = draw_points(img, gt_kps, color=(0,255,0))
        right = draw_points(img, preds, color=(0,0,255))
        vis = np.concatenate([left, right], axis=1)
        # annotate
        cv2.putText(vis, f"NME: {nme:.4f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
        cv2.putText(vis, f"PCK@0.05: {pck:.3f}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)

        out_path = out_dir / f"diag_img{img_id}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        processed += 1
        if processed >= args.max_images:
            break

    if nme_list:
        print(f"Processed: {processed} images")
        print(f"Mean NME: {np.mean(nme_list):.4f}")
        print(f"Median NME: {np.median(nme_list):.4f}")
        print(f"Mean PCK@0.05: {np.mean(pck_list):.4f}")
    else:
        print("No images processed")

if __name__ == '__main__':
    main()
