"""
Analyze predicted vs GT heatmaps and compute per-joint statistics and NME.
Saves JSON summary to outputs/compare_heatmaps/stats.json
"""
import json
from pathlib import Path
import numpy as np
import torch
import cv2
from train_hrnet_facial import FacialKeypointsDataset, HRNetConfig
from model_utils import load_model_safely
from scipy.spatial.distance import euclidean


def generate_gt_heatmaps_from_scaled(keypoints, config):
    hk, wk = config.output_size[1], config.output_size[0]
    heatmaps = np.zeros((config.num_keypoints, hk, wk), dtype=np.float32)
    sigma = config.sigma
    size = 6 * sigma + 3
    for i in range(config.num_keypoints):
        x = keypoints[i * 3]
        y = keypoints[i * 3 + 1]
        v = keypoints[i * 3 + 2]
        if v == 0:
            continue
        xs = x * wk / config.input_size[0]
        ys = y * hk / config.input_size[1]
        x_min, y_min = int(xs - size // 2), int(ys - size // 2)
        x_max, y_max = x_min + size, y_min + size

        x_range = np.arange(0, size, 1, np.float32)
        y_range = x_range[:, None]
        x0 = y0 = size // 2
        g = np.exp(-((x_range - x0) ** 2 + (y_range - y0) ** 2) / (2 * sigma ** 2))

        g_x_min = max(0, -x_min)
        g_y_min = max(0, -y_min)
        g_x_max = min(size, wk - x_min)
        g_y_max = min(size, hk - y_min)

        h_x_min = max(0, x_min)
        h_y_min = max(0, y_min)
        h_x_max = min(wk, x_max)
        h_y_max = min(hk, y_max)

        if h_x_max > h_x_min and h_y_max > h_y_min:
            heatmaps[i, h_y_min:h_y_max, h_x_min:h_x_max] = g[g_y_min:g_y_max, g_x_min:g_x_max]
    return heatmaps


def analyze_one(gt_heatmaps, pred_heatmaps, gt_keypoints_scaled, bbox, config):
    num_joints = config.num_keypoints
    stats = {}
    hk, wk = config.output_size[1], config.output_size[0]

    diag = np.sqrt(bbox[2] ** 2 + bbox[3] ** 2) if bbox is not None else 1.0

    nmess = []
    per_joint = []
    for j in range(num_joints):
        gt = gt_heatmaps[j]
        pr = pred_heatmaps[j]
        # Basic stats
        total_gt = gt.sum()
        total_pr = pr.sum()
        max_gt = float(gt.max())
        max_pr = float(pr.max())
        mean_gt = float(gt.mean())
        mean_pr = float(pr.mean())
        std_pr = float(pr.std())

        # Concentration: fraction of energy in top 1% and top 5%
        flat_pr = pr.flatten()
        if flat_pr.sum() > 0:
            k1 = max(1, int(0.01 * flat_pr.size))
            k5 = max(1, int(0.05 * flat_pr.size))
            sorted_idx = np.argsort(flat_pr)[::-1]
            top1_frac = float(flat_pr[sorted_idx[:k1]].sum() / flat_pr.sum())
            top5_frac = float(flat_pr[sorted_idx[:k5]].sum() / flat_pr.sum())
        else:
            top1_frac = 0.0
            top5_frac = 0.0

        # Argmax coords for pred
        idx = np.argmax(pr)
        py, px = divmod(idx, wk)
        # scale to input image size
        pred_x = px * config.input_size[0] / wk
        pred_y = py * config.input_size[1] / hk

        # GT keypoint scaled coords
        gx = gt_keypoints_scaled[j * 3]
        gy = gt_keypoints_scaled[j * 3 + 1]
        v = gt_keypoints_scaled[j * 3 + 2]

        nme = None
        if v != 0:
            nme = euclidean((pred_x, pred_y), (gx, gy)) / (diag + 1e-6)
            nmess.append(nme)

        per_joint.append({
            'joint': j,
            'visible': bool(v != 0),
            'max_pred': max_pr,
            'mean_pred': mean_pr,
            'std_pred': std_pr,
            'top1_frac': top1_frac,
            'top5_frac': top5_frac,
            'nme': nme
        })

    overall_nme = float(np.mean(nmess)) if len(nmess) > 0 else None
    stats['overall_nme'] = overall_nme
    stats['per_joint'] = per_joint
    return stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_file', required=True)
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--max_images', type=int, default=10)
    args = parser.parse_args()

    out_dir = Path('outputs/compare_heatmaps')
    out_dir.mkdir(parents=True, exist_ok=True)

    config = HRNetConfig()
    dataset = FacialKeypointsDataset(args.coco_file, args.image_root, config, is_train=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_safely(args.model, target_keypoints=config.num_keypoints, device=device)
    model.eval()

    summary = {
        'images': [],
        'aggregates': {}
    }

    nmes = []
    top1s = []
    top5s = []

    for i in range(min(len(dataset), args.max_images)):
        item = dataset[i]
        img_id = item['image_id']
        bbox = None
        # Try to get bbox from the dataset annotations
        try:
            ann = dataset.annotations[img_id][0]
            bbox = ann.get('bbox')
        except Exception:
            bbox = None

        gt_keypoints = item['keypoints'].numpy().tolist()
        gt_heatmaps = generate_gt_heatmaps_from_scaled(gt_keypoints, config)

        # Recreate resized input image
        # We will read the raw image to apply same crop as dataset
        img_info = dataset.images[i]
        image_file = img_info['file_name']
        image_path = Path(args.image_root) / image_file
        if not image_path.exists():
            image_path = None
            image_name = Path(image_file).name
            for split in ['train', 'valid', 'test']:
                alt = Path(args.image_root) / split / 'images' / image_name
                if alt.exists():
                    image_path = alt
                    break
        if image_path is None:
            # fallback: use zeros
            inp = torch.zeros((1,3,config.input_size[1], config.input_size[0]))
            pred_np = np.zeros_like(gt_heatmaps)
        else:
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # crop same as dataset
            if bbox is not None:
                x,y,w,h = bbox
                padding = 0.3
                x = max(0, x - w * padding / 2)
                y = max(0, y - h * padding / 2)
                w = w * (1 + padding)
                h = h * (1 + padding)
                x = max(0, min(x, img.shape[1] - 1))
                y = max(0, min(y, img.shape[0] - 1))
                w = min(w, img.shape[1] - x)
                h = min(h, img.shape[0] - y)
                cropped = img[int(y):int(y+h), int(x):int(x+w)]
                if cropped.size == 0:
                    cropped = img.copy()
                resized = cv2.resize(cropped, tuple(config.input_size))
            else:
                resized = cv2.resize(img, tuple(config.input_size))

            inp = torch.FloatTensor(resized.transpose(2,0,1) / 255.0).unsqueeze(0)
            inp = (inp - torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)) / torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
            inp = inp.to(device)

            with torch.no_grad():
                pred = model(inp)
                if isinstance(pred, torch.Tensor) and pred.dim() == 4:
                    pred_np = pred.squeeze(0).cpu().numpy()
                    # Ensure shape matches heatmap (num_joints, h, w)
                    if pred_np.shape[1:] != (config.output_size[1], config.output_size[0]):
                        resized_preds = np.zeros_like(gt_heatmaps)
                        for k in range(pred_np.shape[0]):
                            resized_preds[k] = cv2.resize(pred_np[k], (config.output_size[0], config.output_size[1]))
                        pred_np = resized_preds
                else:
                    pred_np = np.zeros_like(gt_heatmaps)

        stats = analyze_one(gt_heatmaps, pred_np, gt_keypoints, bbox, config)
        summary['images'].append({'image_index': i, 'image_id': img_id, 'stats': stats})
        if stats['overall_nme'] is not None:
            nmes.append(stats['overall_nme'])
        for pj in stats['per_joint']:
            top1s.append(pj['top1_frac'])
            top5s.append(pj['top5_frac'])

    summary['aggregates']['mean_nme'] = float(np.mean(nmes)) if len(nmes) > 0 else None
    summary['aggregates']['median_nme'] = float(np.median(nmes)) if len(nmes) > 0 else None
    summary['aggregates']['mean_top1_frac'] = float(np.mean(top1s)) if len(top1s) > 0 else None
    summary['aggregates']['mean_top5_frac'] = float(np.mean(top5s)) if len(top5s) > 0 else None

    out_file = out_dir / 'stats.json'
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print('Wrote', out_file)
    print('Mean NME:', summary['aggregates']['mean_nme'])
    print('Median NME:', summary['aggregates']['median_nme'])
    print('Mean top1% fraction:', summary['aggregates']['mean_top1_frac'])
    print('Mean top5% fraction:', summary['aggregates']['mean_top5_frac'])
