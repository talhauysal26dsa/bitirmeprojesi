"""
Ultra-Optimized P2H Training with Adaptive Learning
Combines all advanced techniques for maximum performance
"""
import os
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
from train_p2h_optimized import transfer_weights_to_p2
from train_p2h_adaptive_callbacks import (
    ReduceLROnPlateauCallback,
    EMACallback,
    GradientClippingCallback,
    OneCycleLRCallback,
    WarmRestartCallback
)
from train_p2h_callbacks import P2HDifferentialLRCallback

def parse_args():
    parser = argparse.ArgumentParser(
        description='Ultra-Optimized P2H Training with Adaptive Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument('--baseline-weights', type=str, 
                        default='runs/detect/train/weights/best.pt',
                        help='Baseline YOLOv8x weights for transfer learning')
    parser.add_argument('--p2-yaml', type=str, 
                        default='/home/ilaha/bitirmeprojesi/yolov8x-p2-custom.yaml',
                        help='P2 architecture YAML')
    parser.add_argument('--data', type=str, 
                        default='/home/ilaha/bitirmeprojesi/yolov8_config.yaml',
                        help='Dataset YAML')
    
    # Training parameters (optimized for multi-scale + small objects)
    parser.add_argument('--epochs', type=int, default=70,
                        help='Max 70 epochs with early stopping (patience=15). '
                             'If model plateaus (baseline did at epoch 40), will stop ~epoch 55. '
        parser.add_argument('--epochs', type=int, default=300,
    parser.add_argument('--batch', type=int, default=16,
                        help='P2H batch size (reduced for multi-scale + 1280 resolution + P2 head)')
    parser.add_argument('--imgsz', type=int, default=1280,
                        help='Base image size for multi-scale training (640-1280 range)')
    parser.add_argument('--device', type=str, default='0')
    
    # Adaptive LR strategy (choose one primary strategy)
    parser.add_argument('--lr-strategy', type=str, default='plateau',
                        choices=['cosine', 'onecycle', 'plateau', 'warm_restart'],
                        help='Primary LR scheduling strategy (default: plateau - most reliable)')
    
    # LR parameters (optimized for P2H stability)
    parser.add_argument('--lr0', type=float, default=0.005,
                        help='Initial learning rate (optimal for P2H: 0.005, slower than baseline for stability)')
    parser.add_argument('--lrf', type=float, default=0.05,
                        help='Final LR fraction (optimal: 0.05 for better fine-tuning)')
    parser.add_argument('--lr-min', type=float, default=1e-6,
                        help='Minimum learning rate')
    
    # Plateau-specific (optimized for 40 epochs)
    parser.add_argument('--plateau-patience', type=int, default=8,
                        help='Patience for ReduceLROnPlateau (optimal for 40 epochs: 8)')
    parser.add_argument('--plateau-factor', type=float, default=0.5,
                        help='LR reduction factor (optimal: 0.5 = halve LR)')
    
    # OneCycle-specific
    parser.add_argument('--onecycle-pct-start', type=float, default=0.3,
                        help='Percentage of cycle for warmup')
    parser.add_argument('--onecycle-div-factor', type=float, default=25,
                        help='Initial LR divisor')
    
    # Warm Restart-specific
    parser.add_argument('--restart-period', type=int, default=50,
                        help='Initial restart period (epochs)')
    parser.add_argument('--restart-mult', type=int, default=2,
                        help='Period multiplier after restart')
    
    # Advanced features (optional - use flags to enable)
    parser.add_argument('--differential-lr', action='store_true', default=False,
                        help='Use differential learning rates for P2H layers')
    parser.add_argument('--lr-backbone', type=float, default=0.0005,
                        help='Learning rate for backbone (optimal: 0.0005)')
    parser.add_argument('--lr-neck', type=float, default=0.001,
                        help='Learning rate for neck (optimal: 0.001)')
    parser.add_argument('--lr-p2', type=float, default=0.005,
                        help='Learning rate for P2 head (optimal: 0.005, 10x backbone)')
    parser.add_argument('--ema', action='store_true', default=False,
                        help='Use Exponential Moving Average')
    parser.add_argument('--ema-decay', type=float, default=0.9999,
                        help='EMA decay rate (optimal: 0.9999)')
    parser.add_argument('--gradient-clip', type=float, default=5.0,
                        help='Max gradient norm (optimal for P2H: 5.0, more aggressive clipping)')
    
    # Optimizer (optimal: AdamW)
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['SGD', 'Adam', 'AdamW', 'RAdam', 'NAdam'],
                        help='Optimizer (optimal: AdamW for best balance)')
    
    # Augmentation (Multi-scale optimized for small objects)
    parser.add_argument('--mosaic', type=float, default=0.5,
                        help='Mosaic augmentation (multi-scale composition)')
    parser.add_argument('--copy-paste', type=float, default=0.2,
                        help='Copy-paste augmentation (increased for small objects)')
    parser.add_argument('--mixup', type=float, default=0.1,
                        help='MixUp augmentation (background robustness)')
    parser.add_argument('--scale', type=float, default=0.35,
                        help='Scale augmentation (±35% for multi-resolution invariance)')
    parser.add_argument('--degrees', type=float, default=15.0,
                        help='Rotation augmentation (±15° for aerial views)')
    parser.add_argument('--translate', type=float, default=0.12,
                        help='Translation augmentation (±12%)')
    

    # Output
    parser.add_argument('--project', type=str, default='runs/detect')
    parser.add_argument('--name', type=str, default='p2h_optimized_final')
    
    return parser.parse_args()

def setup_callbacks(model, args, total_steps=None):
    """Setup adaptive learning callbacks based on strategy"""
    
    print("\\n" + "="*80)
    print("ADAPTIVE LEARNING CONFIGURATION")
    print("="*80)
    
    callbacks_info = []
        # Differential LR for P2H (always enabled by default)
    if args.differential_lr:
        print(f"\n🎯 DIFFERENTIAL LEARNING RATES (P2H Optimization)")
        print(f"   Backbone:  {args.lr_backbone:.6f} (preserve pretrained knowledge)")
        print(f"   Neck:      {args.lr_neck:.6f} (adapt features)")
        print(f"   P2 Head:   {args.lr_p2:.6f} (learn new features, {args.lr_p2/args.lr_backbone:.0f}x backbone)")
        print(f"   Strategy: New P2 layers learn faster than pretrained layers")
        
        diff_lr_cb = P2HDifferentialLRCallback(
            lr_backbone=args.lr_backbone,
            lr_neck=args.lr_neck,
            lr_p2=args.lr_p2
        )
        model.add_callback('on_train_start', diff_lr_cb.on_train_start)
        callbacks_info.append("DifferentialLR")
        # Primary LR strategy
    if args.lr_strategy == 'onecycle':
        print(f"\\n🚀 PRIMARY: OneCycle LR Policy")
        print(f"   Max LR: {args.lr0:.6f}")
        print(f"   Min LR: {args.lr0/args.onecycle_div_factor:.6f} → {args.lr_min:.6f}")
        print(f"   Warmup: {args.onecycle_pct_start*100:.0f}% of training")
        
        onecycle_cb = OneCycleLRCallback(
            max_lr=args.lr0,
            total_steps=total_steps,
            pct_start=args.onecycle_pct_start,
            div_factor=args.onecycle_div_factor,
            final_div_factor=args.lr0/args.lr_min
        )
        model.add_callback('on_train_start', onecycle_cb.on_train_start)
        model.add_callback('on_train_batch_end', onecycle_cb.on_train_batch_end)
        callbacks_info.append("OneCycle")
        
    elif args.lr_strategy == 'plateau':
        print(f"\\n📉 PRIMARY: ReduceLROnPlateau")
        print(f"   Monitor: metrics/mAP50(B)")
        print(f"   Patience: {args.plateau_patience} epochs")
        print(f"   Factor: {args.plateau_factor}x reduction")
        print(f"   Min LR: {args.lr_min:.6f}")
        
        plateau_cb = ReduceLROnPlateauCallback(
            monitor='metrics/mAP50(B)',
            patience=args.plateau_patience,
            factor=args.plateau_factor,
            min_lr=args.lr_min,
            verbose=True
        )
        model.add_callback('on_fit_epoch_end', plateau_cb.on_fit_epoch_end)
        callbacks_info.append("Plateau")
        
    elif args.lr_strategy == 'warm_restart':
        print(f"\\n🔄 PRIMARY: Warm Restarts (SGDR)")
        print(f"   Initial period: {args.restart_period} epochs")
        print(f"   Multiplier: {args.restart_mult}x")
        print(f"   Min LR: {args.lr_min:.6f}")
        
        restart_cb = WarmRestartCallback(
            T_0=args.restart_period,
            T_mult=args.restart_mult,
            eta_min=args.lr_min
        )
        model.add_callback('on_train_start', restart_cb.on_train_start)
        model.add_callback('on_train_epoch_end', restart_cb.on_train_epoch_end)
        callbacks_info.append("WarmRestart")
        
    else:  # cosine (default, built-in to YOLO)
        print(f"\\n📊 PRIMARY: Cosine Annealing (built-in)")
        print(f"   Initial LR: {args.lr0:.6f}")
        print(f"   Final LR: {args.lr0 * 0.01:.6f}")
        callbacks_info.append("Cosine")
    
    # EMA (recommended for all strategies)
    if args.ema:
        print(f"\\n✨ EMA: Exponential Moving Average")
        print(f"   Decay: {args.ema_decay:.4f}")
        print(f"   Benefit: Smoother predictions, better generalization")
        
        ema_cb = EMACallback(decay=args.ema_decay)
        model.add_callback('on_train_start', ema_cb.on_train_start)
        model.add_callback('on_train_batch_end', ema_cb.on_train_batch_end)
        model.add_callback('on_fit_epoch_end', ema_cb.on_fit_epoch_end)
        callbacks_info.append("EMA")
    
    # Gradient Clipping
    if args.gradient_clip > 0:
        print(f"\\n✂️ Gradient Clipping")
        print(f"   Max norm: {args.gradient_clip}")
        print(f"   Benefit: Prevents exploding gradients, stable training")
        
        clip_cb = GradientClippingCallback(max_norm=args.gradient_clip)
        model.add_callback('on_before_optimizer_step', clip_cb.on_before_optimizer_step)
        callbacks_info.append("GradClip")
    
    print(f"\\n📋 Active callbacks: {', '.join(callbacks_info)}")
    print("="*80)
    
    return callbacks_info

def main():
    args = parse_args()
    
    print("\\n" + "="*80)
    print("ULTRA-OPTIMIZED P2H TRAINING")
    print("="*80)
    print(f"\\nStrategy: Transfer Learning + {args.lr_strategy.upper()} LR + Adaptive Optimizer")
    
    # Validate files
    if not os.path.exists(args.baseline_weights):
        print(f"\\n❌ Baseline weights not found: {args.baseline_weights}")
        return
    
    # Transfer weights
    print("\\n" + "="*80)
    print("STEP 1: WEIGHT TRANSFER")
    print("="*80)
    model = transfer_weights_to_p2(
        baseline_weights=args.baseline_weights,
        p2_yaml=args.p2_yaml,
        device=args.device
    )
    
    # Estimate total steps for OneCycle
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Rough estimate (will be refined during training)
    # Assuming ~5000 training images, batch=24 → ~208 steps/epoch
    total_steps = args.epochs * 200  # Conservative estimate
    
    # Setup callbacks
    print("\\n" + "="*80)
    print("STEP 2: ADAPTIVE LEARNING SETUP")
    print("="*80)
    callbacks_info = setup_callbacks(model, args, total_steps)
    
    # Training configuration
    print("\\n" + "="*80)
    print("STEP 3: TRAINING CONFIGURATION")
    print("="*80)
    print(f"\\nOptimizer: {args.optimizer}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Epochs: {args.epochs}")
    print(f"\\nAugmentation:")
    print(f"  Mosaic: {args.mosaic}")
    print(f"  Copy-Paste: {args.copy_paste}")
    print(f"  MixUp: {args.mixup}")
    print(f"  Scale: ±{args.scale*100:.0f}%")
    
    # Start training
    print("\\n" + "="*80)
    print("STEP 4: TRAINING")
    print("="*80)
    
    results = model.train(
        # Data
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        
        # Learning rate (optimized for P2H convergence)
        lr0=args.lr0 if args.lr_strategy != 'onecycle' else args.lr0/args.onecycle_div_factor,
        lrf=args.lrf,  # Final LR fraction for better fine-tuning
        
        # Cosine LR (only if using built-in)
        cos_lr=(args.lr_strategy == 'cosine'),
        
        # Warmup
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Optimizer
        optimizer=args.optimizer,
        momentum=0.937 if args.optimizer == 'SGD' else 0.9,
        weight_decay=0.0005,
        
        # Augmentation (Multi-scale optimized for small objects)
        mosaic=args.mosaic,
        copy_paste=args.copy_paste,
        mixup=args.mixup,
        degrees=args.degrees,      # ±15° for aerial views
        translate=args.translate,  # ±15%
        scale=args.scale,          # ±40% for multi-resolution
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,               # Slightly increased variety
        hsv_s=0.4,                 # Increased saturation
        hsv_v=0.4,
        auto_augment='',     # FAIR: disabled (same as baseline)
        erasing=0.0,         # FAIR: disabled (same as baseline)
        
        # Loss weights (optimized for aerial detection - emphasis on localization)
        box=7.5,  # box_loss_gain: Higher weight for precise bounding boxes (critical for small aerial objects)
        cls=0.5,
        dfl=1.5,
        
        # Training settings
            patience=25,  # Early stopping: stops if no improvement for 25 epochs (tolerant for plateau recovery)
        save=True,
        save_period=10,
        workers=8,  # Disable multiprocessing (fixes connection reset error)
        cache='disk',  # Disable cache for stability
        project=args.project,
        name=args.name,
        exist_ok=False,
        pretrained=False,
        verbose=True,
        seed=42,
        deterministic=True,
        val=True,
        plots=True,
        close_mosaic=10,
        rect=False,
        iou=0.7,
        amp=True,
        nbs=64,
    )
    
    # Results summary
    print("\\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    
    save_dir = Path(results.save_dir)
    print(f"\\nResults: {save_dir}")
    print(f"Best weights: {save_dir / 'weights' / 'best.pt'}")
    
    # Print metrics
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\\n📊 FINAL METRICS:")
        print(f"  mAP@50:    {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall:    {metrics.get('metrics/recall(B)', 0):.4f}")
    
    print(f"\\n🎯 Adaptive Strategy Used: {args.lr_strategy.upper()}")
    print(f"📋 Active Features: {', '.join(callbacks_info)}")
    
    print("\\n" + "="*80)
    print("NEXT: Evaluate with evaluate_models.py")
    print("="*80)
    
    return results

if __name__ == '__main__':
    main()
