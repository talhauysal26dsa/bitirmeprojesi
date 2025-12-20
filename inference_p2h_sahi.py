"""
SAHI Inference Script - P2H Model
Slicing Aided Hyper Inference for improved small object detection
Uses P2H YOLOv8x model (4-head detection) trained on aerial dataset
"""
import os
import argparse
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.file import download_from_url
import cv2
import json
import csv
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='SAHI Inference with P2H YOLOv8x Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument('--model', type=str, 
                        default='runs/detect/p2h_fair/weights/best.pt',
                        help='Path to P2H model weights')
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
                        help='Path to image or directory of images')
    parser.add_argument('--output', type=str, default='runs/sahi/p2h',
                        help='Output directory for results')
    parser.add_argument('--save-json', action='store_true',
                        help='Save results as JSON')
    parser.add_argument('--save-vis', action='store_true',
                        help='Save visualized predictions')
    
    return parser.parse_args()

def setup_model(args):
    """Initialize SAHI detection model"""
    print(f"\n{'='*80}")
    print("P2H MODEL + SAHI INFERENCE")
    print(f"{'='*80}")
    print(f"\nModel: {args.model}")
    print(f"Architecture: YOLOv8x-P2 (4-head detection)")
    print(f"Confidence: {args.conf}")
    print(f"Device: {args.device}")
    print(f"Slice size: {args.slice_height}x{args.slice_width}")
    print(f"Overlap: {args.overlap_height}x{args.overlap_width}")
    
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=args.model,
        confidence_threshold=args.conf,
        device=args.device
    )
    
    return detection_model

def process_image(image_path, detection_model, args):
    """Process a single image with SAHI"""
    result = get_sliced_prediction(
        image=str(image_path),
        detection_model=detection_model,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_height,
        overlap_width_ratio=args.overlap_width,
        verbose=0
    )
    
    return result

def save_results(result, image_path, output_dir, args):
    """Save prediction results"""
    image_name = Path(image_path).stem
    
    # Save JSON with proper detection data
    if args.save_json:
        json_path = output_dir / 'json' / f'{image_name}.json'
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract detection data manually
        detections = []
        for obj in result.object_prediction_list:
            detection = {
                'bbox': obj.bbox.to_xyxy(),  # [x1, y1, x2, y2]
                'score': obj.score.value,
                'category_id': obj.category.id,
                'category_name': obj.category.name
            }
            detections.append(detection)
        
        # Save to JSON
        with open(json_path, 'w') as f:
            json.dump({
                'image': str(image_path),
                'detections': detections,
                'num_detections': len(detections)
            }, f, indent=2)
    
    # Save visualization
    if args.save_vis:
        vis_path = output_dir / 'visualizations' / f'{image_name}.jpg'
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        result.export_visuals(export_dir=str(vis_path.parent))
    
    return result

def main():
    """Main execution"""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup model
    detection_model = setup_model(args)
    
    # Get image list
    source_path = Path(args.source)
    if source_path.is_file():
        image_paths = [source_path]
    else:
        image_paths = list(source_path.glob('*.jpg')) + \
                     list(source_path.glob('*.png')) + \
                     list(source_path.glob('*.jpeg'))
    
    print(f"\nFound {len(image_paths)} images")
    print(f"Output directory: {output_dir}")
    print(f"\nProcessing images...\n")
    
    # Process images
    results = []
    per_class_stats = defaultdict(int)
    
    for image_path in tqdm(image_paths, desc="SAHI Inference (P2H)"):
        result = process_image(image_path, detection_model, args)
        save_results(result, image_path, output_dir, args)
        
        # Collect statistics
        detections_info = []
        for obj in result.object_prediction_list:
            class_name = obj.category.name
            per_class_stats[class_name] += 1
            detections_info.append({
                'class': class_name,
                'confidence': obj.score.value,
                'bbox': obj.bbox.to_xyxy()
            })
        
        results.append({
            'image': image_path.name,
            'image_path': str(image_path),
            'total_detections': len(result.object_prediction_list),
            'detections': detections_info
        })
    
    # Save summary CSV
    save_summary_csv(results, per_class_stats, output_dir, args)
    
    # Summary
    print(f"\n{'='*80}")
    print("INFERENCE COMPLETE")
    print(f"{'='*80}")
    print(f"Total images: {len(image_paths)}")
    
    if results:
        total_dets = sum(r['total_detections'] for r in results)
        avg_dets = total_dets / len(results)
        print(f"Total detections: {total_dets}")
        print(f"Average detections per image: {avg_dets:.2f}")
        
        # Per-class summary
        print(f"\nPer-class detections:")
        for class_name, count in sorted(per_class_stats.items()):
            print(f"  {class_name}: {count}")
    else:
        print("No results to summarize")
    
    print(f"\nResults saved to: {output_dir}")

def save_summary_csv(results, per_class_stats, output_dir, args):
    """Save detailed results to CSV files"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Per-image CSV
    csv_path = output_dir / f'inference_results_{timestamp}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Total_Detections', 'Airplane', 'Bird', 'Drone', 'Helicopter'])
        
        for result in results:
            class_counts = defaultdict(int)
            for det in result['detections']:
                class_counts[det['class']] += 1
            
            writer.writerow([
                result['image'],
                result['total_detections'],
                class_counts.get('Airplane', 0),
                class_counts.get('Bird', 0),
                class_counts.get('Drone', 0),
                class_counts.get('Helicopter', 0)
            ])
    
    print(f"\n✅ Per-image results saved: {csv_path}")
    
    # 2. Detailed detections CSV (all bboxes)
    detailed_csv = output_dir / f'detailed_detections_{timestamp}.csv'
    with open(detailed_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])
        
        for result in results:
            for det in result['detections']:
                bbox = det['bbox']
                writer.writerow([
                    result['image'],
                    det['class'],
                    f"{det['confidence']:.4f}",
                    f"{bbox[0]:.2f}",
                    f"{bbox[1]:.2f}",
                    f"{bbox[2]:.2f}",
                    f"{bbox[3]:.2f}"
                ])
    
    print(f"✅ Detailed detections saved: {detailed_csv}")
    
    # 3. Summary statistics
    summary_csv = output_dir / f'summary_statistics_{timestamp}.csv'
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Images', len(results)])
        writer.writerow(['Total Detections', sum(r['total_detections'] for r in results)])
        writer.writerow(['Avg Detections/Image', f"{sum(r['total_detections'] for r in results) / len(results):.2f}"])
        writer.writerow(['Model', args.model])
        writer.writerow(['Confidence Threshold', args.conf])
        writer.writerow(['Slice Size', f"{args.slice_height}x{args.slice_width}"])
        writer.writerow(['Overlap', f"{args.overlap_height}x{args.overlap_width}"])
        writer.writerow(['Timestamp', timestamp])
        writer.writerow([])
        writer.writerow(['Class', 'Count'])
        for class_name, count in sorted(per_class_stats.items()):
            writer.writerow([class_name, count])
    
    print(f"✅ Summary statistics saved: {summary_csv}")

if __name__ == "__main__":
    main()
