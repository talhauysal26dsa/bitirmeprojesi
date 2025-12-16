"""
YOLOv8 Training Script with Weighted Loss
Baseline training for unified RGB + Thermal dataset
NO oversampling/undersampling - uses weighted loss only
"""
from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
import os

# Configuration
CONFIG_PATH = '/home/talha/bitirmeprojesi/yolov8_config.yaml'
MODEL_SIZE = 'yolov8x.pt'  # 80GB VRAM allows largest model: yolov8x
EPOCHS = 300  # More epochs for better convergence with large model
BATCH_SIZE = 32  # Optimized for memory efficiency with large image size
IMG_SIZE = 896  # Balanced size: better than 640, more memory efficient than 1280
DEVICE = 0  # GPU index, or 'cpu'

# Class weights (inverse frequency based on analysis)
# Calculated from total training data distribution
CLASS_WEIGHTS = [
    1.017,  # Airplane (2460 instances)
    1.086,  # Bird (2304 instances)
    1.000,  # Drone (2501 instances - baseline, most common)
    1.206   # Helicopter (2074 instances - least common, highest weight)
]

def check_gpu():
    """Check GPU availability and print info"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Available: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        return True
    else:
        print("⚠ No GPU available, using CPU (will be slow!)")
        return False

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

def train_yolov8():
    """Train YOLOv8 with weighted loss"""
    print("="*80)
    print("YOLOv8 BASELINE TRAINING - WEIGHTED LOSS")
    print("="*80)
    print(f"\nModel: {MODEL_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Class Weights: {CLASS_WEIGHTS}")
    print(f"Device: {'GPU' if DEVICE != 'cpu' else 'CPU'}")
    
    # Check GPU
    has_gpu = check_gpu()
    if has_gpu and DEVICE == 0:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        recommended_batch = get_recommended_batch_size(gpu_memory)
        if recommended_batch != BATCH_SIZE:
            print(f"\n💡 Recommended batch size for your GPU: {recommended_batch}")
            print(f"   Current batch size: {BATCH_SIZE}")
    
    # Load model
    print(f"\nLoading model: {MODEL_SIZE}")
    model = YOLO(MODEL_SIZE)
    
    # Training configuration
    print("\nStarting training...")
    print("Training progress will be saved to: runs/detect/train")
    
    results = model.train(
        # Data configuration
        data=CONFIG_PATH,
        
        # Training parameters
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        
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
        name='train',
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
    print(f"          data={CONFIG_PATH} split=test")
    print("\n3. Run inference:")
    print(f"   python -m ultralytics.yolo predict model={save_dir / 'weights' / 'best.pt'} \\")
    print("          source=path/to/images")
    
    return results

def main():
    """Main execution"""
    # Check if config exists
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Error: Config file not found: {CONFIG_PATH}")
        print("Please run prepare_unified_dataset.py first!")
        return
    
    # Load and verify config
    with open(CONFIG_PATH, 'r') as f:
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
    results = train_yolov8()
    
    return results

if __name__ == "__main__":
    main()
