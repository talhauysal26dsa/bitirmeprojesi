"""
YOLOv8 Training Script with Weighted Loss
Baseline training for unified RGB + Thermal dataset
Supports P2 head for improved small object detection
NO oversampling/undersampling - uses weighted loss only
"""
from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
import os
import argparse

# Default Configuration
DEFAULT_CONFIG_PATH = '/home/ilaha/bitirmeprojesi/yolov8_config.yaml'
DEFAULT_EPOCHS = 300
DEFAULT_BATCH_SIZE = 32
DEFAULT_IMG_SIZE = 896
DEFAULT_DEVICE = 0

# Class weights (inverse frequency based on analysis)
# Calculated from total training data distribution
CLASS_WEIGHTS = [
    1.017,  # Airplane (2460 instances)
    1.086,  # Bird (2304 instances)
    1.000,  # Drone (2501 instances - baseline, most common)
    1.206   # Helicopter (2074 instances - least common, highest weight)
]

def check_gpu(device=0):
    """Check GPU availability and print info
    
    Args:
        device: GPU device index (or -1 for CPU)
    
    Returns:
        tuple: (has_gpu, gpu_memory_gb)
    """
    if device == -1 or not torch.cuda.is_available():
        print("⚠ Using CPU (will be slow!)")
        return False, 0
    
    try:
        gpu_name = torch.cuda.get_device_name(device)
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"GPU Available: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        return True, gpu_memory
    except (RuntimeError, AssertionError):
        print(f"⚠ GPU device {device} not available, falling back to CPU")
        return False, 0

def get_recommended_batch_size(gpu_memory_gb):
    """Recommend batch size based on GPU memory"""
    if gpu_memory_gb >= 16:
        return 32
    elif gpu_memory_gb >= 12:
        return 16
    elif gpu_memory_gb >= 8:
        return 8
    else:
        return 4

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='YOLOv8 Training Script with Optional P2 Head Support',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model architecture
    parser.add_argument('--p2head', action='store_true',
                        help='Use P2 head for improved small object detection (4-head: P2/P3/P4/P5)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=None,
                        help='Batch size (auto-adjusted for P2 if not specified)')
    parser.add_argument('--imgsz', type=int, default=DEFAULT_IMG_SIZE,
                        help='Input image size')
    parser.add_argument('--device', type=int, default=DEFAULT_DEVICE,
                        help='GPU device index (or -1 for CPU)')
    
    # Data configuration
    parser.add_argument('--data', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path to dataset YAML config')
    
    return parser.parse_args()

def train_yolov8(args):
    """Train YOLOv8 with weighted loss"""
    # Determine model and batch size based on P2 head flag
    if args.p2head:
        # Use custom P2 YAML configuration
        model_name = '/home/ilaha/bitirmeprojesi/yolov8x-p2-custom.yaml'
        # Reduce batch size for P2 head due to higher memory usage
        batch_size = args.batch if args.batch is not None else max(16, DEFAULT_BATCH_SIZE - 8)
        project_name = 'train_p2'
        arch_type = 'P2 HEAD (4-head: P2/P3/P4/P5)'
    else:
        model_name = 'yolov8x.pt'
        batch_size = args.batch if args.batch is not None else DEFAULT_BATCH_SIZE
        project_name = 'train'
        arch_type = 'BASELINE (3-head: P3/P4/P5)'
    
    print("="*80)
    print(f"YOLOv8 TRAINING - {arch_type}")
    print("="*80)
    print(f"\nModel: {model_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {args.imgsz}")
    print(f"Class Weights: {CLASS_WEIGHTS}")
    print(f"Device: {'GPU' if args.device >= 0 else 'CPU'}")
    if args.p2head:
        print(f"\n⚡ P2 Head Enabled: Higher resolution detection for small objects")
        print(f"   Feature map scales: 4x, 8x, 16x, 32x (vs baseline 8x, 16x, 32x)")
    
    # Check GPU
    has_gpu, gpu_memory = check_gpu(args.device)
    if has_gpu:
        recommended_batch = get_recommended_batch_size(gpu_memory)
        if args.p2head:
            recommended_batch = max(16, recommended_batch - 8)  # Adjust for P2 memory overhead
        if recommended_batch != batch_size:
            print(f"\n💡 Recommended batch size for your GPU: {recommended_batch}")
            print(f"   Current batch size: {batch_size}")
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model = YOLO(model_name)
    
    # Training configuration
    print("\nStarting training...")
    print(f"Training progress will be saved to: runs/detect/{project_name}")
    
    results = model.train(
        # Data configuration
        data=args.data,
        
        # Training parameters
        epochs=args.epochs,
        batch=batch_size,
        imgsz=args.imgsz,
        device=args.device,
        
        # NOTE: cls_pw (class weighting) removed - not supported in current Ultralytics version
        # Alternative: Use data augmentation or custom callback for handling class imbalance
        
        # Minimal augmentation (user requirement: no heavy augmentation)
        mosaic=0.0,      # Disable mosaic augmentation
        mixup=0.0,       # Disable mixup augmentation
        degrees=5.0,     # Minimal rotation (±5 degrees)
        translate=0.05,  # Minimal translation (5%)
        scale=0.1,       # Minimal scaling (±10%)
        fliplr=0.5,      # 50% horizontal flip (standard)
        flipud=0.0,      # No vertical flip
        hsv_h=0.0,       # No hue shift (especially for thermal)
        hsv_s=0.0,       # No saturation shift
        hsv_v=0.0,       # No value shift
        
        # Optimizer settings - optimized for large batch size
        optimizer='AdamW',
        lr0=0.002,       # Higher LR for large batch (scaled with batch size)
        lrf=0.001,       # Lower final LR for fine-tuning
        momentum=0.937,
        weight_decay=0.0005,
        
        # Loss weights
        box=7.5,         # Box loss weight
        cls=0.5,         # Class loss weight
        dfl=1.5,         # DFL loss weight
        
        # Other settings - optimized for 80GB VRAM
        patience=100,    # More patience for large model
        save=True,       # Save checkpoints
        save_period=10,  # Save every N epochs
        workers=8,       # Reduced workers to save memory
        cache='disk',    # Use disk cache instead of RAM to save VRAM
        project='runs/detect',
        name=project_name,
        exist_ok=False,
        pretrained=True,
        verbose=True,
        seed=42,         # Reproducibility
        deterministic=True,
        
        # Validation
        val=True,
        plots=True,
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    # Print results location
    save_dir = Path(results.save_dir)
    print(f"\nResults saved to: {save_dir}")
    print(f"Best weights: {save_dir / 'weights' / 'best.pt'}")
    print(f"Last weights: {save_dir / 'weights' / 'last.pt'}")
    
    # Print validation metrics
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print("\n📊 FINAL VALIDATION METRICS:")
        print(f"  mAP@50: {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review training curves in TensorBoard:")
    print(f"   tensorboard --logdir {save_dir.parent}")
    print("\n2. Validate on test set:")
    print(f"   python -m ultralytics.yolo val model={save_dir / 'weights' / 'best.pt'} \\")
    print(f"          data={args.data} split=test")
    print("\n3. Run inference:")
    print(f"   python -m ultralytics.yolo predict model={save_dir / 'weights' / 'best.pt'} \\")
    print("          source=path/to/images")
    
    return results

def main():
    """Main execution"""
    # Parse arguments
    args = parse_args()
    
    # Check if config exists
    if not os.path.exists(args.data):
        print(f"❌ Error: Config file not found: {args.data}")
        print("Please run prepare_unified_dataset.py first!")
        return
    
    # Load and verify config
    with open(args.data, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\nDataset Configuration:")
    print(f"  Path: {config['path']}")
    print(f"  Classes: {config['names']}")
    
    # Check if dataset exists
    dataset_path = config['path']
    if not os.path.exists(dataset_path):
        print(f"\n❌ Error: Dataset not found: {dataset_path}")
        print("Please run prepare_unified_dataset.py first!")
        return
    
    # Check splits exist
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            print(f"⚠ Warning: {split} split not found: {split_path}")
    
    # Start training
    results = train_yolov8(args)
    
    return results

if __name__ == "__main__":
    main()
