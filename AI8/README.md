# HRNet Facial Keypoints Detection

A complete PyTorch implementation for facial keypoint detection using HRNet (High-Resolution Network) on lateral face images. This system converts YOLO keypoint format to COCO format and provides training, evaluation, and visualization capabilities.

## Features

- ✅ YOLO to COCO format converter for 14-point facial keypoints
- ✅ HRNet-W18-C-Small-v2 implementation optimized for facial landmarks
- ✅ Comprehensive evaluation metrics (PCK, NME, orthodontic measurements)
- ✅ Visualization with keypoint overlay and facial outline
- ✅ Pretrained model support with fine-tuning capabilities
- ✅ Automated pipeline for easy setup and execution

## Dataset Format

### Input (YOLO Format)
Each label file contains one line per face:
```
class_id bbox_cx bbox_cy bbox_w bbox_h kp1_x kp1_y v1 kp2_x kp2_y v2 ... kp14_x kp14_y v14
```

- Coordinates and bbox are normalized (0-1) relative to image dimensions
- Visibility values: 2=visible, 1=labeled but not visible, 0=not labeled/missing

### Output (COCO Format)
Standard COCO keypoint format with:
- Bbox format: [x_min, y_min, width, height] in pixels
- Keypoints format: [x1, y1, v1, x2, y2, v2, ...] flattened array
- Proper area and num_keypoints fields

## Installation

### Requirements
- Python 3.7+
- PyTorch 1.9+
- CUDA-capable GPU with 4GB+ VRAM (recommended)

### Quick Setup
```bash
# Install dependencies
python -m pip install -r requirements.txt

# Run complete pipeline
python run_pipeline.py --step all
```

### Manual Installation
```bash
pip install torch torchvision opencv-python numpy pandas matplotlib scipy scikit-image pillow tqdm tensorboard pycocotools seaborn albumentations
```

## Usage

### 1. Dataset Conversion
Convert YOLO format to COCO format:
```bash
python yolo_to_coco_converter.py --dataset_root dataset --output coco_annotations.json
```

### 2. Model Training
Train HRNet model with pretrained weights:
```bash
python train_hrnet_facial.py \
    --coco_file coco_annotations.json \
    --image_root dataset \
    --pretrained hrnet_w18_small_model_v1.pth \
    --save_dir checkpoints \
    --batch_size 16 \
    --epochs 100
```

#### Training Configuration
- **Input Size**: 256x256 pixels
- **Output Size**: 64x64 heatmaps
- **Batch Size**: 16-32 (adjust based on GPU memory)
- **Learning Rate**: 0.001 with MultiStepLR scheduler
- **Epochs**: 100 (early stopping recommended)

### 3. Model Evaluation
Evaluate trained model and generate metrics:
```bash
python evaluate_hrnet_facial.py \
    --model_path checkpoints/best_model.pth \
    --coco_file coco_annotations.json \
    --image_root dataset \
    --save_dir evaluation_results \
    --batch_size 16
```

### 4. Automated Pipeline
Run the complete pipeline:
```bash
# Full pipeline
python run_pipeline.py --step all

# Individual steps
python run_pipeline.py --step install
python run_pipeline.py --step convert
python run_pipeline.py --step train
python run_pipeline.py --step evaluate
```

## Model Architecture

### HRNet-W18-C-Small-v2
- **Backbone**: HRNet-W18 with reduced channels for efficiency
- **Input**: 3×256×256 RGB images
- **Output**: 14×64×64 heatmaps
- **Parameters**: ~4M (optimized for speed and memory)

### Key Features
- Multi-scale feature fusion
- High-resolution representation maintenance
- Gaussian heatmap generation for keypoint localization
- Pretrained weight initialization

## Evaluation Metrics

### Standard Metrics
- **PCK@0.05**: Percentage of Correct Keypoints at 5% threshold
- **PCK@0.10**: Percentage of Correct Keypoints at 10% threshold  
- **NME**: Normalized Mean Error (normalized by bbox diagonal)

### Orthodontic Metrics
- **Distance Measurements**:
  - Nasolabial distance (nose tip to upper lip)
  - Lip height (upper lip to lower lip)
  - Lower face height (nose tip to chin)
  - Facial width (eye outer to jaw angle)
  - Total face height (forehead to chin)

- **Angle Measurements**:
  - Facial convexity angle
  - Lower facial angle

## Output Files

### Training Outputs
- `checkpoints/`: Model checkpoints and best model
- `checkpoints/best_model.pth`: Best performing model
- Training logs with loss curves

### Evaluation Outputs
- `evaluation_results/summary_metrics.json`: Overall performance metrics
- `evaluation_results/evaluation_metrics.csv`: Per-image detailed metrics
- `evaluation_results/visualization_*.png`: Sample visualizations

### File Structure
```
project/
├── dataset/
│   ├── train/images/ & labels/
│   ├── valid/images/ & labels/
│   └── test/images/ & labels/
├── checkpoints/
├── evaluation_results/
├── coco_annotations.json
└── hrnet_w18_small_model_v1.pth
```

## Hardware Requirements

### Minimum Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB
- **GPU**: 4GB VRAM (GTX 1650 or better)
- **Storage**: 5GB free space

### Recommended Requirements
- **CPU**: 8+ cores (Intel i7 or AMD Ryzen 7)
- **RAM**: 16GB
- **GPU**: 8GB+ VRAM (RTX 3070 or better)
- **Storage**: 10GB+ free space

### Performance Estimates
- **Training Time**: ~2-4 hours for 100 epochs (RTX 3070)
- **Inference Speed**: ~50-100 FPS (batch size 1)
- **Memory Usage**: ~2-4GB GPU memory during training

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size to 8 or 4
   - Use gradient accumulation
   - Enable mixed precision training

2. **Slow Training**
   - Increase num_workers in DataLoader
   - Use SSD storage for dataset
   - Check GPU utilization

3. **Poor Convergence**
   - Verify data augmentation settings
   - Check learning rate schedule
   - Ensure proper data normalization

4. **Import Errors**
   - Install missing packages: `pip install -r requirements.txt`
   - Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`

### Performance Optimization

1. **Memory Optimization**
   ```python
   # In training script, add:
   torch.backends.cudnn.benchmark = True
   torch.backends.cudnn.deterministic = False
   ```

2. **Speed Optimization**
   ```python
   # Use mixed precision training
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

## Model Interpretation

### Understanding Results

1. **PCK Scores**
   - PCK@0.05 > 0.85: Excellent performance
   - PCK@0.05 > 0.75: Good performance
   - PCK@0.05 < 0.65: Needs improvement

2. **NME Scores**
   - NME < 0.05: Excellent accuracy
   - NME < 0.08: Good accuracy
   - NME > 0.10: Poor accuracy

3. **Orthodontic Metrics**
   - Values in pixels, useful for clinical applications
   - Compare ratios rather than absolute distances
   - Angle measurements should be anatomically reasonable (30-180°)

### Visualization Guide

- **Green circles**: Ground truth keypoints
- **Red crosses**: Predicted keypoints
- **Blue polygon**: Facial outline
- **Text overlay**: Orthodontic measurements

## References

- [HRNet: Deep High-Resolution Representation Learning](https://arxiv.org/abs/1908.07919)
- [HRNet-Image-Classification GitHub](https://github.com/HRNet/HRNet-Image-Classification)
- [COCO Keypoint Detection Format](https://cocodataset.org/#keypoints-2020)

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

For issues and questions, please open a GitHub issue with detailed information about your problem.
