"""
HRNet Facial Keypoints Evaluation and Inference Script
Computes PCK, NME, orthodontic metrics, and visualizations
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import json
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.distance import euclidean
import math

from train_hrnet_facial import SimpleHRNet, HRNetConfig, FacialKeypointsDataset

try:
    from hrnet_model import get_pose_net, hrnet_w18_small_config
    ENHANCED_HRNET_AVAILABLE = True
except ImportError:
    ENHANCED_HRNET_AVAILABLE = False


class FacialKeypointsEvaluator:
    """Evaluator for facial keypoints detection"""
    
    def __init__(self, config):
        self.config = config
        self.num_keypoints = config.num_keypoints
        
        # Define keypoint names for better understanding
        self.keypoint_names = [
            'forehead', 'eyebrow_outer', 'eyebrow_inner', 'eye_outer',
            'eye_inner', 'nose_bridge', 'nose_tip', 'nose_bottom',
            'lip_top', 'lip_corner', 'lip_bottom', 'chin_tip',
            'jaw_mid', 'jaw_angle'
        ]
        
        # Define orthodontic landmark pairs for distance measurements
        self.orthodontic_pairs = [
            ('nose_tip', 'lip_top'),      # Nasolabial distance
            ('lip_top', 'lip_bottom'),     # Lip height
            ('nose_tip', 'chin_tip'),      # Lower face height
            ('eye_outer', 'jaw_angle'),    # Facial width
            ('forehead', 'chin_tip'),      # Total face height
        ]
        
        # Define triplets for angle measurements
        self.orthodontic_angles = [
            ('forehead', 'nose_tip', 'chin_tip'),  # Facial convexity
            ('nose_tip', 'lip_top', 'chin_tip'),   # Lower facial angle
        ]
    
    def get_max_preds(self, batch_heatmaps):
        """Get predictions from heatmaps"""
        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.max(heatmaps_reshaped, 2)
        
        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))
        
        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
        
        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)
        
        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)
        
        preds *= pred_mask
        return preds, maxvals
    
    def calc_dists(self, preds, target, normalize):
        """Calculate distances between predictions and targets"""
        preds = preds.astype(np.float32)
        target = target.astype(np.float32)
        dists = np.zeros((preds.shape[1], preds.shape[0]))
        
        for n in range(preds.shape[0]):
            for c in range(preds.shape[1]):
                if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                    normed_preds = preds[n, c, :] / normalize[n]
                    normed_targets = target[n, c, :] / normalize[n]
                    dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                else:
                    dists[c, n] = -1
        return dists
    
    def dist_acc(self, dists, thr=0.5):
        """Calculate accuracy based on distance threshold"""
        dist_cal = np.not_equal(dists, -1)
        num_dist_cal = dist_cal.sum()
        if num_dist_cal > 0:
            return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
        else:
            return -1
    
    def compute_pck(self, predictions, targets, bbox_sizes, threshold=0.05):
        """Compute Percentage of Correct Keypoints (PCK)"""
        # Normalize by bbox diagonal
        normalize = np.array([np.sqrt(bbox_sizes[:, 0] * bbox_sizes[:, 1])]).T
        
        dists = self.calc_dists(predictions, targets, normalize)
        acc = self.dist_acc(dists, threshold)
        
        return acc, dists
    
    def compute_nme(self, predictions, targets, bbox_sizes):
        """Compute Normalized Mean Error (NME)"""
        # Normalize by bbox diagonal
        normalize = np.array([np.sqrt(bbox_sizes[:, 0] * bbox_sizes[:, 1])]).T
        
        dists = self.calc_dists(predictions, targets, normalize)
        
        # Calculate mean error for valid keypoints
        valid_dists = dists[dists != -1]
        if len(valid_dists) > 0:
            nme = np.mean(valid_dists)
        else:
            nme = -1
        
        return nme
    
    def compute_orthodontic_distances(self, keypoints):
        """Compute orthodontic distance measurements"""
        distances = {}
        
        for pair_name, (kp1_name, kp2_name) in zip(
            ['nasolabial', 'lip_height', 'lower_face', 'facial_width', 'total_face'],
            self.orthodontic_pairs
        ):
            try:
                kp1_idx = self.keypoint_names.index(kp1_name)
                kp2_idx = self.keypoint_names.index(kp2_name)
                
                kp1 = keypoints[kp1_idx * 2:(kp1_idx * 2) + 2]
                kp2 = keypoints[kp2_idx * 2:(kp2_idx * 2) + 2]
                
                # Check if both keypoints are valid
                if (kp1[0] > 0 and kp1[1] > 0 and 
                    kp2[0] > 0 and kp2[1] > 0):
                    distance = euclidean(kp1, kp2)
                    distances[pair_name] = distance
                else:
                    distances[pair_name] = -1
            except (ValueError, IndexError):
                distances[pair_name] = -1
        
        return distances
    
    def compute_orthodontic_angles(self, keypoints):
        """Compute orthodontic angle measurements"""
        angles = {}
        
        for angle_name, (kp1_name, kp2_name, kp3_name) in zip(
            ['facial_convexity', 'lower_facial_angle'],
            self.orthodontic_angles
        ):
            try:
                kp1_idx = self.keypoint_names.index(kp1_name)
                kp2_idx = self.keypoint_names.index(kp2_name)
                kp3_idx = self.keypoint_names.index(kp3_name)
                
                kp1 = keypoints[kp1_idx * 2:(kp1_idx * 2) + 2]
                kp2 = keypoints[kp2_idx * 2:(kp2_idx * 2) + 2]
                kp3 = keypoints[kp3_idx * 2:(kp3_idx * 2) + 2]
                
                # Check if all keypoints are valid
                if (kp1[0] > 0 and kp1[1] > 0 and 
                    kp2[0] > 0 and kp2[1] > 0 and
                    kp3[0] > 0 and kp3[1] > 0):
                    
                    # Calculate angle using vectors
                    v1 = np.array(kp1) - np.array(kp2)
                    v2 = np.array(kp3) - np.array(kp2)
                    
                    # Calculate angle in degrees
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                    angles[angle_name] = angle
                else:
                    angles[angle_name] = -1
            except (ValueError, IndexError, ZeroDivisionError):
                angles[angle_name] = -1
        
        return angles
    
    def visualize_keypoints(self, image, keypoints, predictions, save_path, 
                           orthodontic_distances=None, orthodontic_angles=None):
        """Visualize keypoints and metrics on image"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Display image
        ax.imshow(image)
        
        # Plot ground truth keypoints (green circles)
        for i in range(0, len(keypoints), 3):
            x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
            if v > 0:
                ax.plot(x, y, 'go', markersize=8, alpha=0.7, label='Ground Truth' if i == 0 else '')
        
        # Plot predicted keypoints (red crosses)
        for i in range(len(predictions)):
            x, y = predictions[i]
            if x > 0 and y > 0:
                ax.plot(x, y, 'rx', markersize=10, markeredgewidth=2, 
                       label='Predictions' if i == 0 else '')
        
        # Draw facial outline polygon
        valid_predictions = [(x, y) for x, y in predictions if x > 0 and y > 0]
        if len(valid_predictions) > 3:
            polygon = patches.Polygon(valid_predictions, linewidth=2, 
                                    edgecolor='blue', facecolor='none', alpha=0.7)
            ax.add_patch(polygon)
        
        # Add text with metrics
        text_lines = []
        if orthodontic_distances:
            text_lines.append("Orthodontic Distances:")
            for name, dist in orthodontic_distances.items():
                if dist > 0:
                    text_lines.append(f"  {name}: {dist:.2f}px")
        
        if orthodontic_angles:
            text_lines.append("\nOrthodontic Angles:")
            for name, angle in orthodontic_angles.items():
                if angle > 0:
                    text_lines.append(f"  {name}: {angle:.1f}°")
        
        if text_lines:
            ax.text(0.02, 0.98, '\n'.join(text_lines), transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10)
        
        ax.set_title('Facial Keypoints Detection Results')
        ax.legend()
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def evaluate_model(self, model, dataloader, device, save_dir):
        """Evaluate model on dataset"""
        model.eval()
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        all_predictions = []
        all_targets = []
        all_bbox_sizes = []
        all_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images = batch['image'].to(device)
                heatmaps_gt = batch['heatmaps'].to(device)
                keypoints_gt = batch['keypoints'].numpy()
                image_ids = batch['image_id'].numpy()
                
                # Forward pass
                outputs = model(images)
                
                # Get predictions from heatmaps
                preds, maxvals = self.get_max_preds(outputs.cpu().numpy())
                
                # Scale predictions back to input image size
                preds[:, :, 0] *= self.config.input_size[0] / self.config.output_size[0]
                preds[:, :, 1] *= self.config.input_size[1] / self.config.output_size[1]
                
                batch_size = images.shape[0]
                
                for i in range(batch_size):
                    pred_keypoints = preds[i].reshape(-1, 2)
                    gt_keypoints = keypoints_gt[i].reshape(-1, 3)
                    
                    # Convert to pixel coordinates for visualization
                    gt_kpts_2d = gt_keypoints[:, :2]
                    
                    # Calculate bbox size (assuming square input)
                    bbox_size = [self.config.input_size[0], self.config.input_size[1]]
                    
                    all_predictions.append(pred_keypoints)
                    all_targets.append(gt_kpts_2d)
                    all_bbox_sizes.append(bbox_size)
                    
                    # Compute orthodontic metrics
                    pred_flat = pred_keypoints.flatten()
                    orthodontic_distances = self.compute_orthodontic_distances(pred_flat)
                    orthodontic_angles = self.compute_orthodontic_angles(pred_flat)
                    
                    # Store metrics
                    metrics = {
                        'image_id': image_ids[i],
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        **orthodontic_distances,
                        **orthodontic_angles
                    }
                    all_metrics.append(metrics)
                    
                    # Visualize every 10th image
                    if batch_idx % 10 == 0 and i == 0:
                        # Convert tensor back to numpy for visualization
                        img_np = images[i].cpu().numpy().transpose(1, 2, 0)
                        # Denormalize
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img_np = std * img_np + mean
                        img_np = np.clip(img_np, 0, 1)
                        
                        gt_flat = keypoints_gt[i].flatten()
                        
                        vis_path = save_dir / f'visualization_batch_{batch_idx}_sample_{i}.png'
                        self.visualize_keypoints(
                            img_np, gt_flat, pred_keypoints, vis_path,
                            orthodontic_distances, orthodontic_angles
                        )
        
        # Convert to numpy arrays for metrics computation
        predictions_np = np.array([pred.reshape(1, -1, 2) for pred in all_predictions])
        targets_np = np.array([target.reshape(1, -1, 2) for target in all_targets])
        bbox_sizes_np = np.array(all_bbox_sizes)
        
        # Reshape for metrics computation
        predictions_np = predictions_np.squeeze(1)
        targets_np = targets_np.squeeze(1)
        
        # Compute PCK and NME
        pck_05, _ = self.compute_pck(predictions_np, targets_np, bbox_sizes_np, threshold=0.05)
        pck_10, _ = self.compute_pck(predictions_np, targets_np, bbox_sizes_np, threshold=0.10)
        nme = self.compute_nme(predictions_np, targets_np, bbox_sizes_np)
        
        print(f"Evaluation Results:")
        print(f"PCK@0.05: {pck_05:.4f}")
        print(f"PCK@0.10: {pck_10:.4f}")
        print(f"NME: {nme:.4f}")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(all_metrics)
        metrics_csv_path = save_dir / 'evaluation_metrics.csv'
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Detailed metrics saved to: {metrics_csv_path}")
        
        # Save summary metrics
        summary_metrics = {
            'PCK_0.05': pck_05,
            'PCK_0.10': pck_10,
            'NME': nme,
            'num_samples': len(all_predictions)
        }
        
        summary_path = save_dir / 'summary_metrics.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_metrics, f, indent=2)
        
        return summary_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate HRNet facial keypoints model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--coco_file', type=str, required=True,
                        help='Path to COCO annotation file')
    parser.add_argument('--image_root', type=str, required=True,
                        help='Root directory containing images')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    args = parser.parse_args()
    
    # Configuration
    config = HRNetConfig()
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    if ENHANCED_HRNET_AVAILABLE:
        model = get_pose_net(hrnet_w18_small_config, is_train=False)
    else:
        model = SimpleHRNet(config)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print(f"Model loaded from: {args.model_path}")
    
    # Dataset
    dataset = FacialKeypointsDataset(args.coco_file, args.image_root, config, is_train=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                            shuffle=False, num_workers=4)
    
    # Evaluator
    evaluator = FacialKeypointsEvaluator(config)
    
    # Evaluate
    metrics = evaluator.evaluate_model(model, dataloader, device, args.save_dir)
    
    print("\nEvaluation completed!")
    print(f"Results saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
