"""
Train the simpler SimpleHRNet from scratch on the dataset for quick recovery.
"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from train_hrnet_facial import HRNetConfig, FacialKeypointsDataset, SimpleHRNet, JointsMSELoss
import torch.optim as optim
import torch.nn.functional as F


def train_simple(args):
    config = HRNetConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_dataset = FacialKeypointsDataset(args.coco_file, args.image_root, config, is_train=True)
    val_dataset = FacialKeypointsDataset(args.coco_file, args.image_root, config, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    model = SimpleHRNet(config)
    model = model.to(device)

    criterion = JointsMSELoss(use_target_weight=config.use_target_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(train_loader):
            imgs = batch['image'].to(device)
            heatmaps = batch['heatmaps'].to(device)
            target_weight = batch['target_weight'].to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            # upsample if necessary
            if outputs.dim() == 4 and outputs.shape[2:] != heatmaps.shape[2:]:
                outputs = F.interpolate(outputs, size=heatmaps.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, heatmaps, target_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch}/{config.epochs-1}, Train Loss: {avg_loss:.6f}')

        # quick validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                heatmaps = batch['heatmaps'].to(device)
                target_weight = batch['target_weight'].to(device)
                outputs = model(imgs)
                if outputs.dim() == 4 and outputs.shape[2:] != heatmaps.shape[2:]:
                    outputs = F.interpolate(outputs, size=heatmaps.shape[2:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, heatmaps, target_weight)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'Epoch {epoch}, Val Loss: {val_loss:.6f}')

        # save checkpoint
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, save_dir / f'simple_epoch_{epoch}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_file', required=True)
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default='checkpoints/simple')
    args = parser.parse_args()
    train_simple(args)
