"""
Fine-tune only HRNet head (final layers) with a peak-preserving loss:
Loss = MSE(heatmaps, GT) + alpha * mean((1 - max_per_channel(pred_heatmap))^2)

Saves checkpoints to checkpoints/finetune_head/
"""
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_hrnet_facial import FacialKeypointsDataset, HRNetConfig, JointsMSELoss
from model_utils import load_model_safely
import numpy as np


def set_requires_grad(model, patterns_to_unfreeze):
    for name, p in model.named_parameters():
        p.requires_grad = False
        for pat in patterns_to_unfreeze:
            if pat in name:
                p.requires_grad = True
                break


def max_per_channel(heatmaps):
    # heatmaps: tensor (N, C, H, W)
    with torch.no_grad():
        return heatmaps.view(heatmaps.shape[0], heatmaps.shape[1], -1).max(dim=2)[0]


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = HRNetConfig()

    dataset = FacialKeypointsDataset(args.coco_file, args.image_root, cfg, is_train=True)
    val_dataset = FacialKeypointsDataset(args.coco_file, args.image_root, cfg, is_train=False)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = load_model_safely(args.pretrained, target_keypoints=cfg.num_keypoints, device=device)
    model.train()

    # Freeze everything except head-related patterns
    # Default patterns chosen to match reconstructed HRNet naming found in the checkpoint
    patterns = args.unfreeze_patterns.split(',') if args.unfreeze_patterns else ['final', 'upsample']
    print('Unfreeze patterns:', patterns)
    set_requires_grad(model, patterns)

    learnable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in learnable)
    print(f'Fine-tuning parameter tensors: {len(learnable)}, total params: {total}')

    optimizer = torch.optim.Adam(learnable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.6), int(args.epochs*0.9)], gamma=0.1)

    criterion = JointsMSELoss(use_target_weight=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    alpha = args.alpha

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch_idx, batch in enumerate(train_loader):
            imgs = batch['image'].to(device)
            heatmaps = batch['heatmaps'].to(device)
            target_weight = batch['target_weight'].to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            # Ensure outputs spatial size matches target
            if isinstance(outputs, torch.Tensor) and outputs.dim() == 4 and outputs.shape[2:] != heatmaps.shape[2:]:
                outputs = F.interpolate(outputs, size=heatmaps.shape[2:], mode='bilinear', align_corners=False)

            loss_mse = criterion(outputs, heatmaps, target_weight)
            # peak preserving term
            maxvals = max_per_channel(outputs)
            peak_loss = torch.mean((1.0 - maxvals) ** 2)
            loss = loss_mse + alpha * peak_loss

            loss.backward()
            optimizer.step()

            running += loss.item()
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}, MSE: {loss_mse.item():.6f}, Peak: {peak_loss.item():.6f}')

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                heatmaps = batch['heatmaps'].to(device)
                target_weight = batch['target_weight'].to(device)
                outputs = model(imgs)
                if isinstance(outputs, torch.Tensor) and outputs.dim() == 4 and outputs.shape[2:] != heatmaps.shape[2:]:
                    outputs = F.interpolate(outputs, size=heatmaps.shape[2:], mode='bilinear', align_corners=False)
                loss_mse = criterion(outputs, heatmaps, target_weight)
                maxvals = max_per_channel(outputs)
                peak_loss = torch.mean((1.0 - maxvals) ** 2)
                val_loss += (loss_mse + alpha * peak_loss).item()
            val_loss /= len(val_loader)
        print(f'Epoch {epoch} Train Loss: {running/len(train_loader):.6f}, Val Loss: {val_loss:.6f}')

        # Save checkpoint
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_dir / f'checkpoint_epoch_{epoch}.pth')

    print('Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_file', required=True)
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--save_dir', default='checkpoints/finetune_head')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--unfreeze_patterns', type=str, default='final,upsample',
                        help='Comma-separated list of parameter name substrings to unfreeze')
    args = parser.parse_args()
    train(args)
