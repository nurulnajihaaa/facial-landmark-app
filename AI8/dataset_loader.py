"""
Quick dataset loader / sanity-check for COCO keypoint annotations.
Saves preview images with overlaid keypoints to `outputs/previews/` and prints a short report.

Usage:
    python dataset_loader.py --coco_file coco_annotations.json --image_root dataset --num_preview 5

"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

KPT_COLOR = (0, 255, 0)
KPT_RADIUS = 3
BBOX_COLOR = (255, 0, 0)


def draw_keypoints(image, keypoints, kp_names=None, color=KPT_COLOR):
    # keypoints: [x,y,v, x,y,v, ...]
    img = image.copy()
    n = len(keypoints) // 3
    for i in range(n):
        x = int(round(keypoints[3 * i + 0]))
        y = int(round(keypoints[3 * i + 1]))
        v = int(keypoints[3 * i + 2])
        if v > 0:
            cv2.circle(img, (x, y), KPT_RADIUS, color, -1)
            if kp_names and i < len(kp_names):
                cv2.putText(img, str(i), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    return img


def draw_bbox(img, bbox, color=BBOX_COLOR):
    x, y, w, h = bbox
    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + w)), int(round(y + h))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_file', type=str, required=True)
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--include_labels', action='store_true', default=True,
                        help='If set, try to find and overlay label .txt files next to images')
    parser.add_argument('--num_preview', type=int, default=5)
    parser.add_argument('--preview_dir', type=str, default='outputs/previews')
    args = parser.parse_args()

    coco_path = Path(args.coco_file)
    image_root = Path(args.image_root)
    out_dir = Path(args.preview_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco.get('images', [])}
    anns = coco.get('annotations', [])

    total_images = len(images)
    total_anns = len(anns)

    missing_images = []
    missing_ann_images = set()

    # simple stats
    kpts_counts = []
    bbox_out_of_bounds = 0

    # Build image->annotations map
    img_to_anns = {}
    for ann in anns:
        img_id = ann['image_id']
        img_to_anns.setdefault(img_id, []).append(ann)

    # Check each image referenced in annotations
    checked = 0
    for img_id, img_info in images.items():
        file_name = img_info['file_name']
        # some COCOs use absolute or split paths; try several fallbacks
        cand_paths = [image_root / file_name, image_root / Path(file_name).name]
        if not any(p.exists() for p in cand_paths):
            # try common splits
            for split in ['train', 'valid', 'test']:
                p = image_root / split / 'images' / Path(file_name).name
                cand_paths.append(p)
        real_path = None
        for p in cand_paths:
            if p.exists():
                real_path = p
                break
        if real_path is None:
            missing_images.append(str(image_root / file_name))
            if img_id in img_to_anns:
                missing_ann_images.add(img_id)
            continue

        # if there are annotations for this image, inspect them
        anns_for_image = img_to_anns.get(img_id, [])
        if anns_for_image:
            img = cv2.imread(str(real_path))
            if img is None:
                missing_images.append(str(real_path))
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            for ai, ann in enumerate(anns_for_image):
                kps = ann.get('keypoints', [])
                bbox = ann.get('bbox', None)
                kpts_counts.append(len(kps)//3)
                if bbox is not None:
                    x, y, w, h = bbox
                    h_img, w_img = img.shape[0], img.shape[1]
                    # check bbox inside image
                    if x < 0 or y < 0 or x + w > w_img or y + h > h_img:
                        bbox_out_of_bounds += 1
                # save preview for first few annotated images
                if checked < args.num_preview:
                    vis = draw_keypoints(img, kps)
                    if bbox is not None:
                        vis = draw_bbox(vis, bbox)

                    # Optionally overlay label file text if available
                    if args.include_labels:
                        # label may be in same folder with .txt extension or in labels/ subfolders
                        label_candidates = []
                        img_basename = Path(file_name).name
                        name_stem = Path(img_basename).stem
                        # common places
                        label_candidates.append(image_root / Path(file_name).with_suffix('.txt'))
                        label_candidates.append(image_root / 'labels' / (name_stem + '.txt'))
                        for split in ['train', 'valid', 'test']:
                            label_candidates.append(image_root / split / 'labels' / (name_stem + '.txt'))

                        label_text = None
                        for lp in label_candidates:
                            if lp.exists():
                                try:
                                    label_text = lp.read_text(encoding='utf-8')
                                except Exception:
                                    label_text = lp.read_text(encoding='latin-1')
                                break

                        if label_text:
                            # render label text onto a padded image area at bottom
                            # wrap into lines of ~60 chars
                            max_chars = 60
                            lines = []
                            for raw_line in label_text.splitlines():
                                while len(raw_line) > max_chars:
                                    lines.append(raw_line[:max_chars])
                                    raw_line = raw_line[max_chars:]
                                lines.append(raw_line)

                            # create canvas with extra space at bottom
                            pad_h = 14 * (len(lines) + 1)
                            h0, w0 = vis.shape[0], vis.shape[1]
                            canvas = np.ones((h0 + pad_h, w0, 3), dtype=np.uint8) * 30
                            canvas[0:h0, :, :] = vis
                            y_text = h0 + 12
                            for ln in lines[:10]:
                                cv2.putText(canvas, ln, (6, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220,220,220), 1)
                                y_text += 14
                            vis_out = canvas
                        else:
                            vis_out = vis
                    else:
                        vis_out = vis

                    out_path = out_dir / f"preview_img{img_id}_ann{ai}.png"
                    # convert back to BGR for cv2.imwrite
                    vis_bgr = cv2.cvtColor(vis_out, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(out_path), vis_bgr)
                    checked += 1

    # Print report
    print("Dataset check report:")
    print(f"  Total images in JSON: {total_images}")
    print(f"  Total annotations: {total_anns}")
    print(f"  Images referenced by annotations: {len(img_to_anns)}")
    print(f"  Missing image files: {len(missing_images)} (first 5): {missing_images[:5]}")
    print(f"  Missing annotated image ids: {list(missing_ann_images)[:5]}")
    if kpts_counts:
        from statistics import mean, median
        print(f"  Keypoints per annotation: mean={mean(kpts_counts):.2f}, median={median(kpts_counts)}")
    print(f"  BBoxes out of bounds count: {bbox_out_of_bounds}")
    print(f"  Preview images written to: {out_dir}")

if __name__ == '__main__':
    main()
