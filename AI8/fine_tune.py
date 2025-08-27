"""
Quick fine-tune runner
- Loads model with `load_model_safely`
- Fine-tunes for a small number of epochs on the training set
- Saves checkpoints to `checkpoints/finetune`

Usage:
    python fine_tune.py --coco_file coco_annotations.json --image_root dataset --pretrained checkpoints/checkpoint_epoch_99.pth --epochs 3 --batch_size 8
"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from train_hrnet_facial import HRNetConfig, FacialKeypointsDataset, train_model
from model_utils import load_model_safely


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_file', type=str, required=True)
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default='checkpoints/finetune')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    config = HRNetConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dataset = FacialKeypointsDataset(args.coco_file, args.image_root, config, is_train=True)
    # Use a small validation split by reusing the dataset (optional)
    val_dataset = None

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = None

    # Load model (uses adaptive loader)
    model = load_model_safely(args.pretrained, target_keypoints=config.num_keypoints, device='cpu')

    # Fine-tune
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    train_model(config, train_loader, val_loader, model, device, args.save_dir)

if __name__ == '__main__':
    main()
