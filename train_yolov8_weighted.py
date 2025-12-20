"""
YOLOv8 Baseline Training Script with Full Augmentation
Optimized training with comprehensive augmentation for small object detection
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
DEFAULT_BATCH_SIZE = 16  # Reduced for multi-scale + higher resolution
DEFAULT_IMG_SIZE = 1280  # Higher base resolution for small objects
DEFAULT_DEVICE = 0
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
        description='YOLOv8 Baseline Training Script with Full Augmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training parameters
        parser.add_argument('--epochs', type=int, default=300,
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
    
    # Project configuration
    parser.add_argument('--name', type=str, default='train',
                        help='Training run name')
    
    return parser.parse_args()

def train_yolov8(args):
    """Train YOLOv8 baseline with full augmentation and transfer learning"""
    # Use baseline YOLOv8x model with pretrained weights
    model_name = 'yolov8x.pt'
    batch_size = args.batch if args.batch is not None else DEFAULT_BATCH_SIZE
    project_name = args.name
    arch_type = 'BASELINE (3-head: P3/P4/P5)'
    
    print("="*80)
    print(f"YOLOv8 TRAINING - {arch_type}")
    print("="*80)
    print(f"\nModel: {model_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {args.imgsz}")
    print(f"Device: {'GPU' if args.device >= 0 else 'CPU'}")
    print(f"\n✅ Weight Transfer: Pretrained weights from COCO")
    print(f"✅ Optimizer: AdamW with Cosine LR")
    print(f"✅ AMP: Enabled (automatic mixed precision)")
    print(f"✅ Augmentation: Balanced (FAIR COMPARISON)")
    
    # Check GPU
    has_gpu, gpu_memory = check_gpu(args.device)
    if has_gpu:
        recommended_batch = get_recommended_batch_size(gpu_memory)
        if recommended_batch != batch_size:
            print(f"\n💡 Recommended batch size for your GPU: {recommended_batch}")
            print(f"   Current batch size: {batch_size}")
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model = YOLO(model_name)
    
    # MULTI-SCALE TRAINING: Critical for multi-resolution dataset
    print("\n🔄 MULTI-SCALE Training Configuration:")
    print("   - Base imgsz: 1280 (small object detection)")
    print("   - Multi-scale range: 640-1280 (training dynamics)")
    print("   - Mosaic: 0.5 ✅ (multi-scale composition)")
    print("   - MixUp: 0.1 ✅ (background robustness)")
    print("   - Copy-Paste: 0.2 ✅ (object variety)")
    print("   - Scale: ±40% ✅ (scale invariance)")
    print("   - Rotation: ±15° ✅")
    print("   - Translation: ±15% ✅")
    print("   - Mixed Precision: bf16 (H100 optimized)")
    print("   - Close mosaic at epoch 290")
    
    aug_params = {
        'mosaic': 0.5,          # Multi-scale composition
        'mixup': 0.1,           # Background robustness
        'copy_paste': 0.2,      # Increased for small objects
        'scale': 0.4,           # ±40% scale invariance for multi-resolution
        'degrees': 15.0,        # ±15° rotation for aerial views
        'translate': 0.15,      # ±15% translation
        'fliplr': 0.5,
        'flipud': 0.0,
        'hsv_h': 0.015,         # Slightly increased for variety
        'hsv_s': 0.4,           # Increased saturation augmentation
        'hsv_v': 0.4,           # Increased value augmentation
        'close_mosaic': 10,     # Close mosaic 10 epochs before end
    }
    
    optimizer_params = {
        'optimizer': 'AdamW',      # AdamW optimizer ✅
        'lr0': 0.001,              # Initial LR (FAIR - same as P2H) ✅
        'lrf': 0.01,               # Final LR multiplier (FAIR - same as P2H)
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'cos_lr': True,            # Cosine LR schedule
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
    }
    
    training_params = {
        'amp': True,               # Mixed precision (H100 optimized)
        'nbs': 64,                 # Nominal batch size for gradient accumulation
    }
    
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
        
        # Augmentation parameters - Full small object augmentation ✅
        **aug_params,
        
        # Optimizer settings - AdamW with ReduceLROnPlateau behavior ✅
        **optimizer_params,
        
        # Loss weights
        box=7.5,         # Box loss weight
        cls=0.5,         # Class loss weight
        dfl=1.5,         # DFL loss weight
        
        # Training parameters - EMA + gradient clipping ✅
        **training_params,
        
        # Other settings
            patience=25,     # Early stopping patience (tolerant for plateau recovery)
        save=True,       # Save checkpoints
        save_period=10,  # Save every N epochs
        workers=8,       # Data loading workers
        cache='disk',    # Disk cache for memory efficiency
        project='runs/detect',
        name=project_name,
        exist_ok=False,
        pretrained=True,
        verbose=True,
        seed=42,         # Reproducibility
        deterministic=True,
        
        # MULTI-SCALE TRAINING (Critical for multi-resolution dataset)
        rect=False,      # Square images for multi-scale (NOT rectangular)
        # Note: Ultralytics automatically applies multi-scale when rect=False
        # Training will use random sizes between [imgsz*0.5, imgsz*1.0]
        
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
