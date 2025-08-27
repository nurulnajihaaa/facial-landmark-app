"""
Compare GT heatmaps vs model predicted heatmaps for a COCO dataset.
Saves per-image visualization grids to outputs/compare_heatmaps/.
"""

import os
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from train_hrnet_facial import FacialKeypointsDataset, HRNetConfig
from model_utils import load_model_safely


def make_heatmap_grid(gt_heatmaps, pred_heatmaps, keypoint_names=None, vmax=None):
    num_joints = gt_heatmaps.shape[0]
    cols = 4
    rows = (num_joints + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 3, rows * 2.5))
    axes = axes.reshape(rows, cols * 2)

    for j in range(num_joints):
        r = j // cols
        c = j % cols
        ax_gt = axes[r, c * 2]
        ax_pred = axes[r, c * 2 + 1]

        ax_gt.imshow(gt_heatmaps[j], cmap='hot', vmax=vmax)
        ax_gt.set_title(f'GT {j+1}' if not keypoint_names else f'GT {keypoint_names[j]}')
        ax_gt.axis('off')

        ax_pred.imshow(pred_heatmaps[j], cmap='hot', vmax=vmax)
        ax_pred.set_title(f'Pred {j+1}' if not keypoint_names else f'Pred {keypoint_names[j]}')
        ax_pred.axis('off')

    # Hide unused axes
    total = rows * cols * 2
    used = num_joints * 2
    for k in range(used, total):
        r = k // (cols * 2)
        c = k % (cols * 2)
        axes[r, c].axis('off')

    plt.tight_layout()
    return fig


def generate_gt_heatmaps(keypoints, config):
    # keypoints is list [x1,y1,v1, x2,y2,v2, ...] in input_size coordinates
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
        # scale to heatmap size
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_file', required=True)
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out_dir', default='outputs/compare_heatmaps')
    parser.add_argument('--max_images', type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = HRNetConfig()

    # Load dataset (we will use the dataset to access bbox and GT keypoints and to follow the same crop rules)
    dataset = FacialKeypointsDataset(args.coco_file, args.image_root, config, is_train=False)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_safely(args.model, target_keypoints=config.num_keypoints, device=device)
    model.eval()

    for i in range(min(len(dataset), args.max_images)):
        item = dataset[i]
        img_id = item['image_id']

        # We need original image and bbox -> reconstruct by reading COCO
        img_info = dataset.images[i]
        ann = dataset.annotations[img_id][0]
        bbox = ann['bbox']

        # Recreate the cropped & resized image used during training
        # Use the dataset's preprocessing path by calling preprocess variant here
        # But since dataset.__getitem__ returns transformed tensor, we replicate crop logic
        image_path = None
        image_file = img_info['file_name']
        if not image_file.startswith('train/') and not image_file.startswith('valid/') and not image_file.startswith('test/'):
            image_path = Path(args.image_root) / image_file
        else:
            image_path = Path(args.image_root) / image_file
        if not image_path.exists():
            image_name = Path(image_file).name
            for split in ['train', 'valid', 'test']:
                alt = Path(args.image_root) / split / 'images' / image_name
                if alt.exists():
                    image_path = alt
                    break
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Use same crop logic as dataset
        x, y, w, h = bbox
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

        # Generate GT heatmaps (keypoints are available in dataset item but in scaled coords)
        gt_keypoints = item['keypoints'].numpy().tolist()
        gt_heatmaps = generate_gt_heatmaps(gt_keypoints, config)

        # Run model
        inp = torch.FloatTensor(resized.transpose(2,0,1) / 255.0).unsqueeze(0)
        # Normalize according to train transforms
        inp = (inp - torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)) / torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
        inp = inp.to(device)

        with torch.no_grad():
            pred = model(inp)
            if isinstance(pred, torch.Tensor) and pred.dim() == 4:
                pred_np = pred.squeeze(0).cpu().numpy()
                # Ensure heatmap dims are (num_joints, h, w)
                if pred_np.shape[1:] != tuple(config.output_size[::-1]):
                    # Resize each channel
                    resized_preds = np.zeros_like(gt_heatmaps)
                    for k in range(pred_np.shape[0]):
                        resized_preds[k] = cv2.resize(pred_np[k], (config.output_size[0], config.output_size[1]))
                    pred_np = resized_preds
            else:
                print(f"Unexpected model output for image {i}")
                continue

        # Normalize heatmaps for display
        vmax = max(gt_heatmaps.max(), pred_np.max())
        fig = make_heatmap_grid(gt_heatmaps, pred_np, vmax=vmax)
        fig.suptitle(f'Image {i} - ID {img_id}')
        out_file = out_dir / f'{i:04d}_img{img_id}_heatmaps.png'
        fig.savefig(str(out_file), dpi=150)
        plt.close(fig)
        print(f'Saved {out_file}')

    print('Done')
