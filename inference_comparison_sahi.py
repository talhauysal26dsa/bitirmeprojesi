"""
SAHI Inference Comparison Script
Compare all model combinations:
1. Baseline (no SAHI)
2. Baseline + SAHI
3. P2H (no SAHI)
4. P2H + SAHI

Generates comprehensive comparison metrics and visualizations
"""
import os
import argparse
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction
from ultralytics import YOLO
import cv2
import json
import time
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Compare Baseline vs P2H with/without SAHI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model paths
    parser.add_argument('--baseline-model', type=str, 
                        default='runs/detect/train/weights/best.pt',
                        help='Path to baseline model weights')
    parser.add_argument('--p2h-model', type=str, 
                        default='runs/detect/p2h_fair/weights/best.pt',
                        help='Path to P2H model weights (trained with fair comparison)')
    
    # Inference parameters
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (cuda:0 or cpu)')
    
    # SAHI parameters (optimized for 896x896 training)
    parser.add_argument('--slice-height', type=int, default=640,
                        help='Slice height in pixels (0.7x training size)')
    parser.add_argument('--slice-width', type=int, default=640,
                        help='Slice width in pixels (0.7x training size)')
    parser.add_argument('--overlap-height', type=float, default=0.3,
                        help='Overlap ratio for height (0.3 better for small objects)')
    parser.add_argument('--overlap-width', type=float, default=0.3,
                        help='Overlap ratio for width (0.3 better for small objects)')
    
    # Input/Output
    parser.add_argument('--source', type=str, required=True,
                        help='Path to test images directory')
    parser.add_argument('--output', type=str, default='runs/sahi/comparison',
                        help='Output directory for comparison results')
    
    return parser.parse_args()

def setup_models(args):
    """Initialize all model configurations"""
    print(f"\n{'='*80}")
    print("MODEL COMPARISON SETUP")
    print(f"{'='*80}\n")
    
    models = {}
    
    # 1. Baseline (no SAHI)
    print("Loading Baseline model (no SAHI)...")
    models['baseline'] = YOLO(args.baseline_model)
    
    # 2. Baseline + SAHI
    print("Loading Baseline model (with SAHI)...")
    models['baseline_sahi'] = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=args.baseline_model,
        confidence_threshold=args.conf,
        device=args.device
    )
    
    # 3. P2H (no SAHI)
    print("Loading P2H model (no SAHI)...")
    models['p2h'] = YOLO(args.p2h_model)
    
    # 4. P2H + SAHI
    print("Loading P2H model (with SAHI)...")
    models['p2h_sahi'] = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=args.p2h_model,
        confidence_threshold=args.conf,
        device=args.device
    )
    
    print("\n✅ All models loaded successfully\n")
    return models

def run_inference(image_path, models, args):
    """Run inference with all model configurations"""
    results = {}
    
    # 1. Baseline (no SAHI)
    start_time = time.time()
    pred = models['baseline'].predict(str(image_path), conf=args.conf, verbose=False)[0]
    num_boxes = len(pred.boxes.data) if hasattr(pred.boxes, 'data') else 0
    results['baseline'] = {
        'detections': num_boxes,
        'time': time.time() - start_time,
        'boxes': pred.boxes.data.cpu().numpy() if num_boxes > 0 else []
    }
    
    # 2. Baseline + SAHI
    start_time = time.time()
    sahi_result = get_sliced_prediction(
        image=str(image_path),
        detection_model=models['baseline_sahi'],
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_height,
        overlap_width_ratio=args.overlap_width,
        verbose=0
    )
    results['baseline_sahi'] = {
        'detections': len(sahi_result.object_prediction_list),
        'time': time.time() - start_time,
        'result': sahi_result
    }
    
    # 3. P2H (no SAHI)
    start_time = time.time()
    pred = models['p2h'].predict(str(image_path), conf=args.conf, verbose=False)[0]
    num_boxes = len(pred.boxes.data) if hasattr(pred.boxes, 'data') else 0
    results['p2h'] = {
        'detections': num_boxes,
        'time': time.time() - start_time,
        'boxes': pred.boxes.data.cpu().numpy() if num_boxes > 0 else []
    }
    
    # 4. P2H + SAHI
    start_time = time.time()
    sahi_result = get_sliced_prediction(
        image=str(image_path),
        detection_model=models['p2h_sahi'],
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_height,
        overlap_width_ratio=args.overlap_width,
        verbose=0
    )
    results['p2h_sahi'] = {
        'detections': len(sahi_result.object_prediction_list),
        'time': time.time() - start_time,
        'result': sahi_result
    }
    
    return results

def generate_comparison_report(all_results, output_dir):
    """Generate comprehensive comparison report"""
    # Aggregate statistics
    stats = {
        'baseline': {'detections': [], 'time': []},
        'baseline_sahi': {'detections': [], 'time': []},
        'p2h': {'detections': [], 'time': []},
        'p2h_sahi': {'detections': [], 'time': []}
    }
    
    for result in all_results:
        for model_name in stats.keys():
            stats[model_name]['detections'].append(result[model_name]['detections'])
            stats[model_name]['time'].append(result[model_name]['time'])
    
    # Create DataFrame with safe division
    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0.0
    
    df = pd.DataFrame({
        'Model': ['Baseline', 'Baseline+SAHI', 'P2H', 'P2H+SAHI'],
        'Avg Detections': [
            safe_avg(stats['baseline']['detections']),
            safe_avg(stats['baseline_sahi']['detections']),
            safe_avg(stats['p2h']['detections']),
            safe_avg(stats['p2h_sahi']['detections'])
        ],
        'Avg Time (s)': [
            safe_avg(stats['baseline']['time']),
            safe_avg(stats['baseline_sahi']['time']),
            safe_avg(stats['p2h']['time']),
            safe_avg(stats['p2h_sahi']['time'])
        ],
        'Total Detections': [
            sum(stats['baseline']['detections']),
            sum(stats['baseline_sahi']['detections']),
            sum(stats['p2h']['detections']),
            sum(stats['p2h_sahi']['detections'])
        ]
    })
    
    # Save CSV
    csv_path = output_dir / 'comparison_results.csv'
    df.to_csv(csv_path, index=False)
    
    # Generate plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Average Detections
    axes[0].bar(df['Model'], df['Avg Detections'], color=['blue', 'cyan', 'green', 'lime'])
    axes[0].set_title('Average Detections per Image')
    axes[0].set_ylabel('Detections')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Average Inference Time
    axes[1].bar(df['Model'], df['Avg Time (s)'], color=['blue', 'cyan', 'green', 'lime'])
    axes[1].set_title('Average Inference Time')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}\n")
    print(df.to_string(index=False))
    print(f"\nResults saved to: {output_dir}")
    print(f"  - CSV: {csv_path}")
    print(f"  - Plots: {output_dir / 'comparison_plots.png'}")
    print(f"\n{'='*80}")
    print("NOTE: These are detection counts only.")
    print("For proper evaluation metrics (mAP, Precision, Recall),")
    print("use: yolo val model=<model_path> data=<data_yaml> split=test")
    print(f"{'='*80}")

def main():
    """Main execution"""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup models
    models = setup_models(args)
    
    # Get image list
    source_path = Path(args.source)
    image_paths = list(source_path.glob('*.jpg')) + \
                 list(source_path.glob('*.png')) + \
                 list(source_path.glob('*.jpeg'))
    
    print(f"Found {len(image_paths)} images for comparison\n")
    
    # Run comparison
    all_results = []
    for image_path in tqdm(image_paths, desc="Running comparison"):
        results = run_inference(image_path, models, args)
        all_results.append(results)
    
    # Generate report
    generate_comparison_report(all_results, output_dir)

if __name__ == "__main__":
    main()
