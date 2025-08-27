"""
Load the app's FacialKeypointsInference with a chosen model and run visual inference on a few dataset images.
Saves matplotlib figures to outputs/streamlit_tests/ and JSON predictions.
"""
from pathlib import Path
import json
import os
import numpy as np
import cv2
import torch

from facial_keypoints_app import FacialKeypointsInference
from train_hrnet_facial import HRNetConfig, FacialKeypointsDataset

out_dir = Path('outputs/streamlit_tests')
out_dir.mkdir(parents=True, exist_ok=True)

MODEL = 'checkpoints/finetune_head/checkpoint_epoch_11.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM = 6

print('Loading inference class with model:', MODEL, 'device:', DEVICE)
inf = FacialKeypointsInference(MODEL, device=DEVICE)
config = HRNetConfig()

dataset = FacialKeypointsDataset('coco_annotations.json', 'dataset', config, is_train=False)

results = []
for i in range(min(NUM, len(dataset))):
    item = dataset[i]
    img_id = item['image_id']
    img_info = dataset.images[i]
    img_file = img_info['file_name']

    # find actual image path same logic as dataset
    image_path = Path('dataset') / img_file
    if not image_path.exists():
        image_path = None
        image_name = Path(img_file).name
        for split in ['train','valid','test']:
            alt = Path('dataset') / split / 'images' / image_name
            if alt.exists():
                image_path = alt
                break
    if image_path is None:
        print('Image path missing for', img_file)
        continue

    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get bbox from annotation
    ann = dataset.annotations[img_id][0]
    bbox = ann.get('bbox')

    # Run predict with bbox so app preprocessing is used
    kps, conf, proc_img = inf.predict(img, bbox=bbox)

    # Visualize using the app's method
    fig = inf.visualize_results(proc_img, kps, conf, inf.calculate_measurements(kps, conf))

    out_png = out_dir / f'vis_{i:02d}_img{img_id}.png'
    fig.savefig(str(out_png), dpi=150)
    fig.clf()

    # Save json
    out_json = out_dir / f'pred_{i:02d}_img{img_id}.json'
    with open(out_json, 'w') as f:
        json.dump({'image_id': int(img_id), 'keypoints': kps.tolist(), 'confidence': conf.tolist()}, f, indent=2)

    results.append({'image_id': int(img_id), 'png': str(out_png), 'json': str(out_json)})
    print('Saved', out_png, out_json)

print('Done. Wrote', len(results), 'results to', out_dir)
