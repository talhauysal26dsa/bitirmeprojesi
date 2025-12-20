"""
Comprehensive Model Evaluation Script
Evaluates multiple YOLO models and saves detailed metrics to CSV
"""
import os
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO
import pandas as pd

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Comprehensive Model Evaluation with CSV Output',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='Paths to model weights (e.g., best.pt)')
    parser.add_argument('--names', type=str, nargs='+', required=True,
                        help='Model names for identification')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data.yaml')
    
    # Evaluation parameters
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--imgsz', type=int, default=896,
                        help='Image size')
    parser.add_argument('--batch', type=int, default=24,
                        help='Batch size')
    parser.add_argument('--conf', type=float, default=0.001,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='IoU threshold')
    parser.add_argument('--device', type=str, default='0',
                        help='Device (cuda:0 or cpu)')
    
    # Output
    parser.add_argument('--output', type=str, default='evaluation_results',
                        help='Output directory')
    parser.add_argument('--save-json', action='store_true',
                        help='Save detailed JSON results')
    
    return parser.parse_args()

def evaluate_model(model_path, data_yaml, args):
    """Evaluate a single model and return metrics"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_path}")
    print(f"{'='*80}\n")
    
    # Load model
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(
        data=data_yaml,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_json=args.save_json,
        plots=True,
        verbose=True
    )
    
    return results

def extract_metrics(results, model_name, model_path):
    """Extract all relevant metrics from validation results"""
    metrics = {
        'model_name': model_name,
        'model_path': model_path,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    # Overall metrics
    box_metrics = results.box
    metrics['precision'] = float(box_metrics.mp)  # mean precision
    metrics['recall'] = float(box_metrics.mr)  # mean recall
    metrics['mAP50'] = float(box_metrics.map50)
    metrics['mAP50-95'] = float(box_metrics.map)
    metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + 1e-8)
    
    # Per-class metrics
    if hasattr(box_metrics, 'ap_class_index'):
        class_names = results.names
        for i, class_idx in enumerate(box_metrics.ap_class_index):
            class_name = class_names[int(class_idx)]
            
            # Precision, Recall, mAP50, mAP50-95 per class
            if hasattr(box_metrics, 'p') and len(box_metrics.p) > i:
                metrics[f'{class_name}_precision'] = float(box_metrics.p[i])
            if hasattr(box_metrics, 'r') and len(box_metrics.r) > i:
                metrics[f'{class_name}_recall'] = float(box_metrics.r[i])
            if hasattr(box_metrics, 'ap50') and len(box_metrics.ap50) > i:
                metrics[f'{class_name}_mAP50'] = float(box_metrics.ap50[i])
            if hasattr(box_metrics, 'ap') and len(box_metrics.ap) > i:
                metrics[f'{class_name}_mAP50-95'] = float(box_metrics.ap[i])
    
    # Speed metrics
    if hasattr(results, 'speed'):
        speed = results.speed
        metrics['speed_preprocess_ms'] = speed.get('preprocess', 0)
        metrics['speed_inference_ms'] = speed.get('inference', 0)
        metrics['speed_postprocess_ms'] = speed.get('postprocess', 0)
        metrics['speed_total_ms'] = sum([v for v in speed.values()])
    
    # Model info
    if hasattr(results, 'results_dict'):
        metrics['fitness'] = results.results_dict.get('fitness', 0)
    
    return metrics

def save_results(all_metrics, output_dir, args):
    """Save results to CSV and JSON"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save to CSV
    csv_file = output_dir / f'evaluation_{args.split}_{timestamp}.csv'
    
    # Convert to DataFrame for better formatting
    df = pd.DataFrame(all_metrics)
    
    # Reorder columns: model info first, then overall metrics, then per-class
    priority_cols = ['model_name', 'model_path', 'timestamp', 
                     'precision', 'recall', 'mAP50', 'mAP50-95', 'f1_score']
    other_cols = [col for col in df.columns if col not in priority_cols]
    df = df[priority_cols + other_cols]
    
    df.to_csv(csv_file, index=False, float_format='%.6f')
    print(f"\n✅ CSV saved: {csv_file}")
    
    # Save to JSON
    json_file = output_dir / f'evaluation_{args.split}_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"✅ JSON saved: {json_file}")
    
    # Create comparison summary
    create_comparison_summary(all_metrics, output_dir, timestamp, args)
    
    return csv_file, json_file

def create_comparison_summary(all_metrics, output_dir, timestamp, args):
    """Create a comparison summary between models"""
    summary_file = output_dir / f'comparison_summary_{args.split}_{timestamp}.txt'
    
    with open(summary_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"MODEL COMPARISON SUMMARY - {args.split.upper()} SET\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
        
        # Overall metrics table
        f.write("OVERALL METRICS:\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Model':<30} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10} {'F1':>10}\n")
        f.write("-"*100 + "\n")
        
        for metrics in all_metrics:
            f.write(f"{metrics['model_name']:<30} "
                   f"{metrics['precision']:>10.4f} "
                   f"{metrics['recall']:>10.4f} "
                   f"{metrics['mAP50']:>10.4f} "
                   f"{metrics['mAP50-95']:>10.4f} "
                   f"{metrics['f1_score']:>10.4f}\n")
        
        f.write("-"*100 + "\n\n")
        
        # Per-class metrics
        if len(all_metrics) > 0:
            # Extract class names
            class_metrics = {}
            for key in all_metrics[0].keys():
                if '_mAP50' in key and key != 'mAP50':
                    class_name = key.replace('_mAP50', '')
                    class_metrics[class_name] = []
            
            if class_metrics:
                f.write("PER-CLASS mAP50:\n")
                f.write("-"*100 + "\n")
                f.write(f"{'Class':<20}")
                for metrics in all_metrics:
                    f.write(f"{metrics['model_name']:<25}")
                f.write("\n")
                f.write("-"*100 + "\n")
                
                for class_name in class_metrics.keys():
                    f.write(f"{class_name:<20}")
                    for metrics in all_metrics:
                        map50_key = f'{class_name}_mAP50'
                        if map50_key in metrics:
                            f.write(f"{metrics[map50_key]:<25.4f}")
                    f.write("\n")
                f.write("-"*100 + "\n\n")
        
        # Speed comparison
        f.write("INFERENCE SPEED:\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Model':<30} {'Preprocess (ms)':>18} {'Inference (ms)':>18} {'Postprocess (ms)':>18} {'Total (ms)':>15}\n")
        f.write("-"*100 + "\n")
        
        for metrics in all_metrics:
            f.write(f"{metrics['model_name']:<30} "
                   f"{metrics.get('speed_preprocess_ms', 0):>18.2f} "
                   f"{metrics.get('speed_inference_ms', 0):>18.2f} "
                   f"{metrics.get('speed_postprocess_ms', 0):>18.2f} "
                   f"{metrics.get('speed_total_ms', 0):>15.2f}\n")
        
        f.write("-"*100 + "\n\n")
        
        # Best model
        best_map50 = max(all_metrics, key=lambda x: x['mAP50'])
        best_map5095 = max(all_metrics, key=lambda x: x['mAP50-95'])
        fastest = min(all_metrics, key=lambda x: x.get('speed_inference_ms', float('inf')))
        
        f.write("BEST PERFORMERS:\n")
        f.write("-"*100 + "\n")
        f.write(f"Best mAP50:     {best_map50['model_name']} ({best_map50['mAP50']:.4f})\n")
        f.write(f"Best mAP50-95:  {best_map5095['model_name']} ({best_map5095['mAP50-95']:.4f})\n")
        f.write(f"Fastest:        {fastest['model_name']} ({fastest.get('speed_inference_ms', 0):.2f} ms)\n")
        f.write("-"*100 + "\n")
    
    print(f"✅ Summary saved: {summary_file}")

def main():
    args = parse_args()
    
    # Validate inputs
    if len(args.models) != len(args.names):
        raise ValueError("Number of models must match number of names")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*100)
    print("MODEL EVALUATION SCRIPT")
    print("="*100)
    print(f"\nDataset: {args.data}")
    print(f"Split: {args.split}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device}")
    print(f"Models to evaluate: {len(args.models)}")
    for name, path in zip(args.names, args.models):
        print(f"  - {name}: {path}")
    print()
    
    # Evaluate all models
    all_metrics = []
    for model_path, model_name in zip(args.models, args.names):
        try:
            results = evaluate_model(model_path, args.data, args)
            metrics = extract_metrics(results, model_name, model_path)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            continue
    
    if not all_metrics:
        print("❌ No models were successfully evaluated")
        return
    
    # Save results
    csv_file, json_file = save_results(all_metrics, output_dir, args)
    
    print("\n" + "="*100)
    print("✅ EVALUATION COMPLETE")
    print("="*100)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - CSV: {csv_file.name}")
    print(f"  - JSON: {json_file.name}")
    print(f"  - Summary: comparison_summary_{args.split}_*.txt")
    print()

if __name__ == '__main__':
    main()
