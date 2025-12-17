# P2 Head Guide for YOLOv8 Training

## Overview

The P2 head feature adds an additional detection head to YOLOv8 for improved small object detection in aerial imagery. This is particularly beneficial for detecting small or distant objects like birds and drones.

## Architecture Comparison

### Baseline YOLOv8x (3-head)
- **Detection heads**: P3, P4, P5
- **Feature map scales**: 8x, 16x, 32x downsampling
- **Parameters**: 68.2M
- **GFLOPs**: 258.1
- **GPU Memory** (batch=4, 640px): ~4GB

### YOLOv8x-P2 (4-head)
- **Detection heads**: P2, P3, P4, P5
- **Feature map scales**: 4x, 8x, 16x, 32x downsampling
- **Parameters**: 66.6M
- **GFLOPs**: 317.2 (+23%)
- **GPU Memory** (batch=4, 640px): ~5.3GB (+33%)

## Usage

### Baseline Training (3-head)
```bash
# Default configuration
python train_yolov8_weighted.py

# With custom parameters
python train_yolov8_weighted.py --epochs 300 --batch 32 --imgsz 896
```

### P2 Head Training (4-head)
```bash
# Enable P2 head
python train_yolov8_weighted.py --p2head

# With custom parameters
python train_yolov8_weighted.py --p2head --epochs 300 --batch 24 --imgsz 896
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--p2head` | flag | False | Enable P2 head for small object detection |
| `--epochs` | int | 300 | Number of training epochs |
| `--batch` | int | Auto | Batch size (auto-adjusted for P2) |
| `--imgsz` | int | 896 | Input image size |
| `--device` | int | 0 | GPU device index (-1 for CPU) |
| `--data` | str | yolov8_config.yaml | Dataset configuration file |

## Performance Expectations

### Training Time
- **P2 head**: ~25-30% longer per epoch due to higher resolution feature maps
- **Baseline**: 100s/epoch (example)
- **P2**: 125-130s/epoch (example)

### Memory Usage
With 896px images and batch size 32:
- **Baseline**: ~60-65GB VRAM
- **P2**: Recommended batch size 24 (~60-65GB VRAM)

### Detection Performance
Research on aerial datasets shows:
- **Small objects**: 4-6% mAP improvement
- **Medium/Large objects**: Similar or slightly better
- **Overall mAP50**: 2-4% improvement expected

## Best Practices

### When to Use P2 Head
✅ **Use P2 when:**
- Detecting small objects (birds, drones, distant aircraft)
- Working with high-resolution aerial imagery
- Small object detection is critical for your application

❌ **Skip P2 when:**
- Only detecting large objects
- Training time is critical
- Limited GPU memory (<40GB)

### Batch Size Recommendations

| GPU Memory | Baseline Batch | P2 Batch | Image Size |
|------------|----------------|----------|------------|
| 80GB (H100) | 32 | 24 | 896px |
| 40GB (A100) | 16 | 12 | 896px |
| 24GB (RTX 4090) | 8 | 6 | 640px |
| 16GB (RTX 4080) | 4 | 4 | 640px |

## Output Directories

Training results are saved to separate directories:
- **Baseline**: `runs/detect/train`
- **P2 Head**: `runs/detect/train_p2`

This allows easy comparison between models.

## Validation

After training, validate both models on the test set:

```bash
# Baseline model
/home/ilaha/bitirmeprojesi/venv/bin/yolo val \
  model=runs/detect/train/weights/best.pt \
  data=/home/ilaha/bitirmeprojesi/yolov8_config.yaml \
  split=test

# P2 model
/home/ilaha/bitirmeprojesi/venv/bin/yolo val \
  model=runs/detect/train_p2/weights/best.pt \
  data=/home/ilaha/bitirmeprojesi/yolov8_config.yaml \
  split=test
```

Compare mAP scores, especially for small object classes (Bird, Drone).

## TensorBoard Visualization

View training curves for both models:

```bash
tensorboard --logdir /home/ilaha/bitirmeprojesi/runs/detect
```

This will show both `train` and `train_p2` runs for easy comparison.

## Technical Details

### P2 Head Architecture
The P2 head adds an extra upsampling layer in the neck to create a 4x downsampled feature map:
- **P2 (4x)**: 224x224 feature map (for 896px input)
- **P3 (8x)**: 112x112 feature map
- **P4 (16x)**: 56x56 feature map
- **P5 (32x)**: 28x28 feature map

### Configuration File
The P2 architecture is defined in `yolov8x-p2-custom.yaml` with 4 classes optimized for aerial detection.

## Troubleshooting

### Out of Memory Error
Reduce batch size:
```bash
python train_yolov8_weighted.py --p2head --batch 16
```

### Slow Training
- Reduce image size: `--imgsz 640`
- Use fewer workers (already optimized in script)
- Consider using baseline model if speed is critical

### Model Not Loading
Ensure you're using the correct model path:
- Baseline: `yolov8x.pt` (pretrained)
- P2: `yolov8x-p2-custom.yaml` (custom config)
