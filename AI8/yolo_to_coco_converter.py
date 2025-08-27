"""
YOLO to COCO Format Converter for Facial Keypoints
Converts YOLO keypoint format to COCO format for use with HRNet/MMPose
"""

import json
import os
import cv2
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
from datetime import datetime


class YoloToCocoConverter:
    def __init__(self, dataset_root: str, output_path: str):
        self.dataset_root = Path(dataset_root)
        self.output_path = Path(output_path)
        self.num_keypoints = 14
        
        # COCO format structure
        self.coco_data = {
            "info": {
                "description": "Lateral Face Keypoints Dataset",
                "version": "1.0",
                "year": 2025,
                "contributor": "Facial Landmarks Detection",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "CC BY 4.0",
                    "url": "https://creativecommons.org/licenses/by/4.0/"
                }
            ],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "face",
                    "supercategory": "person",
                    "keypoints": [
                        "kp1", "kp2", "kp3", "kp4", "kp5", "kp6", "kp7",
                        "kp8", "kp9", "kp10", "kp11", "kp12", "kp13", "kp14"
                    ],
                    "skeleton": [
                        [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
                        [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14]
                    ]
                }
            ]
        }
        
        self.image_id = 1
        self.annotation_id = 1
    
    def convert_visibility(self, yolo_vis: int) -> int:
        """Convert YOLO visibility to COCO visibility
        YOLO: 2=visible, 1=labeled but not visible, 0=not labeled/missing
        COCO: 2=visible, 1=occluded, 0=not labeled
        """
        return yolo_vis
    
    def parse_yolo_line(self, line: str) -> Tuple[int, List[float], List[float]]:
        """Parse a YOLO format line and extract bbox and keypoints"""
        parts = line.strip().split()
        
        class_id = int(parts[0])
        bbox_cx, bbox_cy, bbox_w, bbox_h = map(float, parts[1:5])
        
        # Extract keypoints (x, y, visibility) for 14 keypoints
        keypoints = []
        for i in range(self.num_keypoints):
            start_idx = 5 + i * 3
            if start_idx + 2 < len(parts):
                x = float(parts[start_idx])
                y = float(parts[start_idx + 1])
                v = int(parts[start_idx + 2])
                keypoints.extend([x, y, self.convert_visibility(v)])
            else:
                # Missing keypoint
                keypoints.extend([0.0, 0.0, 0])
        
        bbox = [bbox_cx, bbox_cy, bbox_w, bbox_h]
        return class_id, bbox, keypoints
    
    def convert_bbox_to_coco(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """Convert YOLO bbox (cx, cy, w, h) normalized to COCO bbox (x, y, w, h) in pixels"""
        cx, cy, w, h = bbox
        
        # Convert from normalized to pixel coordinates
        cx_px = cx * img_width
        cy_px = cy * img_height
        w_px = w * img_width
        h_px = h * img_height
        
        # Convert from center format to top-left format
        x_min = cx_px - w_px / 2
        y_min = cy_px - h_px / 2
        
        return [x_min, y_min, w_px, h_px]
    
    def convert_keypoints_to_coco(self, keypoints: List[float], img_width: int, img_height: int) -> List[float]:
        """Convert normalized keypoints to pixel coordinates"""
        coco_keypoints = []
        for i in range(0, len(keypoints), 3):
            x_norm = keypoints[i]
            y_norm = keypoints[i + 1]
            visibility = keypoints[i + 2]
            
            # Convert to pixel coordinates
            x_px = x_norm * img_width
            y_px = y_norm * img_height
            
            coco_keypoints.extend([x_px, y_px, visibility])
        
        return coco_keypoints
    
    def count_visible_keypoints(self, keypoints: List[float]) -> int:
        """Count number of labeled keypoints (visibility > 0)"""
        count = 0
        for i in range(2, len(keypoints), 3):
            if keypoints[i] > 0:
                count += 1
        return count
    
    def process_split(self, split: str) -> None:
        """Process a dataset split (train/valid/test)"""
        images_dir = self.dataset_root / split / "images"
        labels_dir = self.dataset_root / split / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"Warning: {split} split not found, skipping...")
            return
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
            image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
        
        print(f"Processing {len(image_files)} images in {split} split...")
        
        for img_path in image_files:
            # Find corresponding label file
            label_path = labels_dir / (img_path.stem + ".txt")
            
            if not label_path.exists():
                print(f"Warning: No label file for {img_path.name}")
                continue
            
            # Read image to get dimensions
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                img_height, img_width = img.shape[:2]
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
                continue
            
            # Add image info
            image_info = {
                "id": self.image_id,
                "file_name": f"{split}/images/{img_path.name}",
                "width": img_width,
                "height": img_height
            }
            self.coco_data["images"].append(image_info)
            
            # Process labels
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        class_id, bbox, keypoints = self.parse_yolo_line(line)
                        
                        # Convert to COCO format
                        coco_bbox = self.convert_bbox_to_coco(bbox, img_width, img_height)
                        coco_keypoints = self.convert_keypoints_to_coco(keypoints, img_width, img_height)
                        
                        # Calculate area
                        area = coco_bbox[2] * coco_bbox[3]
                        
                        # Count visible keypoints
                        num_keypoints = self.count_visible_keypoints(coco_keypoints)
                        
                        # Create annotation
                        annotation = {
                            "id": self.annotation_id,
                            "image_id": self.image_id,
                            "category_id": 1,
                            "bbox": coco_bbox,
                            "area": area,
                            "keypoints": coco_keypoints,
                            "num_keypoints": num_keypoints,
                            "iscrowd": 0
                        }
                        
                        self.coco_data["annotations"].append(annotation)
                        self.annotation_id += 1
            
            except Exception as e:
                print(f"Error processing label {label_path}: {e}")
                continue
            
            self.image_id += 1
    
    def convert(self):
        """Convert the entire dataset"""
        print("Converting YOLO to COCO format...")
        
        # Process each split
        for split in ["train", "valid", "test"]:
            self.process_split(split)
        
        # Save COCO format JSON
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.coco_data, f, indent=2)
        
        print(f"Conversion completed!")
        print(f"Total images: {len(self.coco_data['images'])}")
        print(f"Total annotations: {len(self.coco_data['annotations'])}")
        print(f"Output saved to: {self.output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO keypoint format to COCO format")
    parser.add_argument("--dataset_root", type=str, default="dataset",
                        help="Path to dataset root directory")
    parser.add_argument("--output", type=str, default="coco_annotations.json",
                        help="Output COCO JSON file path")
    
    args = parser.parse_args()
    
    converter = YoloToCocoConverter(args.dataset_root, args.output)
    converter.convert()


if __name__ == "__main__":
    main()
