"""
Diagnose whether predicted keypoints are permuted relative to GT.
For a few images, compute NME using index order and using Hungarian matching.
"""
import numpy as np
import json
from pathlib import Path
import cv2
import torch
from scipy.optimize import linear_sum_assignment
from train_hrnet_facial import FacialKeypointsDataset, HRNetConfig
from model_utils import load_model_safely


def compute_nme_pairwise(pred, gt, bbox_diag=1.0):
    # pred, gt: (N_joints, 2) arrays
    dists = np.linalg.norm(pred - gt, axis=1)
    return np.mean(dists) / (bbox_diag + 1e-6), dists


def extract_pred_and_gt_for_image(dataset, idx, model, device):
    item = dataset[idx]
    img_id = item['image_id']
    # reconstruct raw image
    img_info = dataset.images[idx]
    image_file = img_info['file_name']
    image_path = Path('dataset') / image_file
    if not image_path.exists():
        image_name = Path(image_file).name
        for split in ['train','valid','test']:
            alt = Path('dataset') / split / 'images' / image_name
            if alt.exists():
                image_path = alt
                break
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ann = dataset.annotations[item['image_id']][0]
    bbox = ann['bbox']
    # Prepare input resized same as dataset
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
    resized = cv2.resize(cropped, tuple((256,256)))
    inp = torch.FloatTensor(resized.transpose(2,0,1)/255.0).unsqueeze(0)
    inp = (inp - torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)) / torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    inp = inp.to(device)
    with torch.no_grad():
        out = model(inp)
        if isinstance(out, torch.Tensor) and out.dim()==4:
            pred_hm = out.squeeze(0).cpu().numpy()
        else:
            raise RuntimeError('Unexpected model output')
    # extract argmax coords in heatmap space
    C, H, W = pred_hm.shape
    pred = np.zeros((C,2), dtype=np.float32)
    for j in range(C):
        flat = pred_hm[j].flatten()
        idxm = np.argmax(flat)
        py, px = divmod(idxm, W)
        # scale to input_size
        pred_x = px * 256 / W
        pred_y = py * 256 / H
        pred[j] = [pred_x, pred_y]
    # GT from item scaled in dataset
    gt_scaled = item['keypoints'].numpy().reshape(-1,3)
    gt = gt_scaled[:, :2]
    # compute diag
    diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
    return pred, gt, diag, idx


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = HRNetConfig()
    dataset = FacialKeypointsDataset('coco_annotations.json', 'dataset', cfg, is_train=False)
    model = load_model_safely('checkpoints/finetune_head/checkpoint_epoch_11.pth', target_keypoints=cfg.num_keypoints, device=device)
    model.eval()

    results = []
    for i in range(min(len(dataset), 12)):
        try:
            pred, gt, diag, idx = extract_pred_and_gt_for_image(dataset, i, model, device)
        except Exception as e:
            print('skip', i, e)
            continue
        nme_index, dists = compute_nme_pairwise(pred, gt, diag)
        # Hungarian matching
        cost = np.linalg.norm(pred[:,None,:] - gt[None,:,:], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_pred = pred[row_ind]
        permuted_gt = gt[col_ind]
        # reorder matched_pred to match gt order for fair comparison: construct pred_reordered where pred_reordered[col_ind]=pred[row_ind]
        pred_reordered = np.zeros_like(pred)
        pred_reordered[col_ind] = pred[row_ind]
        nme_hungarian, _ = compute_nme_pairwise(pred_reordered, gt, diag)

        results.append({'index': i, 'image_id': int(dataset.images[i]['id']), 'nme_index': float(nme_index), 'nme_hungarian': float(nme_hungarian)})

    print('Results (first 12):')
    for r in results:
        print(r)
    # save
    with open('outputs/compare_heatmaps/permutation_diagnosis.json','w') as f:
        json.dump(results, f, indent=2)
    print('Wrote outputs/compare_heatmaps/permutation_diagnosis.json')
