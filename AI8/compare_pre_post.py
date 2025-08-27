"""
Compare per-image and per-joint heatmap statistics between two models (pre vs post fine-tune).
Saves JSON and CSV summaries to outputs/compare_heatmaps/compare_pre_post.json and .csv
"""
import json
from pathlib import Path
import numpy as np
import torch
import cv2
from train_hrnet_facial import FacialKeypointsDataset, HRNetConfig
from model_utils import load_model_safely
from scipy.spatial.distance import euclidean

out_dir = Path('outputs/compare_heatmaps')
out_dir.mkdir(parents=True, exist_ok=True)


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


def stats_for_model(model, dataset, config, device, max_images=50):
    model.eval()
    stats = []
    for i in range(min(len(dataset), max_images)):
        item = dataset[i]
        img_id = item['image_id']
        gt_keypoints = item['keypoints'].numpy().tolist()
        gt_hm = generate_gt_heatmaps_from_scaled(gt_keypoints, config)

        # create input image as in analyze_heatmaps
        img_info = dataset.images[i]
        image_file = img_info['file_name']
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
            pred_np = np.zeros_like(gt_hm)
        else:
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # get bbox
            bbox = None
            try:
                ann = dataset.annotations[img_id][0]
                bbox = ann.get('bbox')
            except Exception:
                bbox = None
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

            inp = torch.FloatTensor(resized.transpose(2,0,1) / 255.0).unsqueeze(0)
            inp = (inp - torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)) / torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
            inp = inp.to(device)
            with torch.no_grad():
                pred = model(inp)
                if isinstance(pred, torch.Tensor) and pred.dim() == 4:
                    pred_np = pred.squeeze(0).cpu().numpy()
                    if pred_np.shape[1:] != (config.output_size[1], config.output_size[0]):
                        resized_preds = np.zeros_like(gt_hm)
                        for k in range(pred_np.shape[0]):
                            resized_preds[k] = cv2.resize(pred_np[k], (config.output_size[0], config.output_size[1]))
                        pred_np = resized_preds
                else:
                    pred_np = np.zeros_like(gt_hm)

        # compute per-joint stats and NME
        num_joints = config.num_keypoints
        hk, wk = config.output_size[1], config.output_size[0]
        diag = None
        try:
            ann = dataset.annotations[img_id][0]
            bbox = ann.get('bbox')
            diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
        except Exception:
            diag = 1.0

        per_joint = []
        nmes = []
        for j in range(num_joints):
            pr = pred_np[j]
            flat = pr.flatten()
            total = flat.sum()
            if total > 0:
                k1 = max(1, int(0.01 * flat.size))
                k5 = max(1, int(0.05 * flat.size))
                sorted_idx = np.argsort(flat)[::-1]
                top1 = float(flat[sorted_idx[:k1]].sum() / total)
                top5 = float(flat[sorted_idx[:k5]].sum() / total)
            else:
                top1 = 0.0
                top5 = 0.0

            idx = np.argmax(pr)
            py, px = divmod(idx, wk)
            pred_x = px * config.input_size[0] / wk
            pred_y = py * config.input_size[1] / hk

            gx = gt_keypoints[j*3]
            gy = gt_keypoints[j*3+1]
            v = gt_keypoints[j*3+2]
            nme = None
            if v != 0:
                nme = euclidean((pred_x, pred_y), (gx, gy)) / (diag + 1e-6)
                nmes.append(nme)

            per_joint.append({'joint': j, 'top1': top1, 'top5': top5, 'nme': nme})

        overall_nme = float(np.mean(nmes)) if len(nmes)>0 else None
        stats.append({'image_index': i, 'image_id': img_id, 'overall_nme': overall_nme, 'per_joint': per_joint})
    return stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pre', required=True)
    parser.add_argument('--model_post', required=True)
    parser.add_argument('--max_images', type=int, default=50)
    args = parser.parse_args()

    config = HRNetConfig()
    dataset = FacialKeypointsDataset('coco_annotations.json', 'dataset', config, is_train=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading pre model...')
    model_pre = load_model_safely(args.model_pre, target_keypoints=config.num_keypoints, device=device)
    print('Loading post model...')
    model_post = load_model_safely(args.model_post, target_keypoints=config.num_keypoints, device=device)

    stats_pre = stats_for_model(model_pre, dataset, config, device, max_images=args.max_images)
    stats_post = stats_for_model(model_post, dataset, config, device, max_images=args.max_images)

    # compute deltas
    rows = []
    nme_pre_list = []
    nme_post_list = []
    for a, b in zip(stats_pre, stats_post):
        pre = a['overall_nme']
        post = b['overall_nme']
        delta = None
        if pre is not None and post is not None:
            delta = post - pre
            nme_pre_list.append(pre)
            nme_post_list.append(post)
        rows.append({'image_index': a['image_index'], 'image_id': a['image_id'], 'nme_pre': pre, 'nme_post': post, 'nme_delta': delta})

    agg = {
        'mean_nme_pre': float(np.mean(nme_pre_list)) if nme_pre_list else None,
        'mean_nme_post': float(np.mean(nme_post_list)) if nme_post_list else None,
        'delta_mean_nme': float(np.mean(nme_post_list) - np.mean(nme_pre_list)) if nme_pre_list and nme_post_list else None
    }

    out = {'per_image': rows, 'aggregates': agg}
    out_file = out_dir / 'compare_pre_post.json'
    with open(out_file, 'w') as f:
        json.dump(out, f, indent=2)

    # print compact report
    print('Wrote', out_file)
    print('Mean NME pre:', agg['mean_nme_pre'])
    print('Mean NME post:', agg['mean_nme_post'])
    print('Delta mean NME (post - pre):', agg['delta_mean_nme'])

    # print top 5 worst improvements
    diffs = [r for r in rows if r['nme_delta'] is not None]
    diffs_sorted = sorted(diffs, key=lambda x: x['nme_delta'], reverse=True)
    print('\nTop 5 images that worsened most (post - pre):')
    for r in diffs_sorted[:5]:
        print(r)

    improved_sorted = sorted(diffs, key=lambda x: x['nme_delta'])
    print('\nTop 5 images that improved most (post - pre):')
    for r in improved_sorted[:5]:
        print(r)

    print('\nDone')
