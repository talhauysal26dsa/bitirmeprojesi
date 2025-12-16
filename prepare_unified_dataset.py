"""
Dataset Preparation Script for YOLOv8 Baseline Training
Merges RGB and Thermal datasets into unified structure
NO oversampling/undersampling - uses all available data
"""
import os
import shutil
from pathlib import Path
from collections import defaultdict
import yaml

# Dataset configuration
DATASETS_ROOT = '/home/talha/bitirmeprojesi/datasets'
OUTPUT_ROOT = '/home/talha/bitirmeprojesi/unified_dataset'

# RGB datasets to merge
RGB_DATASETS = [
    'rgb/Anti2(rgb)',
    'rgb/flyingobject(rbg)'
]

# Thermal datasets to merge  
THERMAL_DATASETS = [
    'thermal/AoD(white-hot-thermal)',
    'thermal/termal_drone(white-hot-thermal)',
    'thermal/IVFlyingObjects. (white-hot-thermal)'
]

# Class names
CLASS_NAMES = ['Airplane', 'Bird', 'Drone', 'Helicopter']

def copy_split(dataset_path, split_name, output_dir, modality_prefix):
    """
    Copy images and labels from a dataset split to unified structure
    
    Args:
        dataset_path: Path to source dataset
        split_name: 'train', 'test', 'valid', or 'val'
        output_dir: Output directory for this split
        modality_prefix: 'rgb' or 'thermal' prefix for filenames
    """
    # Handle both 'valid' and 'val' naming
    split_names = [split_name]
    if split_name == 'valid':
        split_names.append('val')
    elif split_name == 'val':
        split_names.append('valid')
    
    copied_count = 0
    for sname in split_names:
        split_path = os.path.join(dataset_path, sname)
        if not os.path.exists(split_path):
            continue
            
        images_path = os.path.join(split_path, 'images')
        labels_path = os.path.join(split_path, 'labels')
        
        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            continue
        
        # Create output directories
        output_images = os.path.join(output_dir, 'images')
        output_labels = os.path.join(output_dir, 'labels')
        os.makedirs(output_images, exist_ok=True)
        os.makedirs(output_labels, exist_ok=True)
        
        # Get dataset name for unique prefixing
        dataset_name = Path(dataset_path).name.replace(' ', '_').replace('(', '').replace(')', '')
        
        # Copy all images and labels
        for img_file in os.listdir(images_path):
            if not (img_file.endswith('.jpg') or img_file.endswith('.png')):
                continue
                
            # Create unique filename
            base_name = os.path.splitext(img_file)[0]
            ext = os.path.splitext(img_file)[1]
            new_name = f"{modality_prefix}_{dataset_name}_{base_name}"
            
            # Copy image
            src_img = os.path.join(images_path, img_file)
            dst_img = os.path.join(output_images, new_name + ext)
            shutil.copy2(src_img, dst_img)
            
            # Copy corresponding label
            label_file = base_name + '.txt'
            src_label = os.path.join(labels_path, label_file)
            if os.path.exists(src_label):
                dst_label = os.path.join(output_labels, new_name + '.txt')
                shutil.copy2(src_label, dst_label)
                copied_count += 1
    
    return copied_count

def merge_datasets():
    """Merge all RGB and thermal datasets into unified structure"""
    print("="*80)
    print("MERGING DATASETS INTO UNIFIED STRUCTURE")
    print("="*80)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, split, 'labels'), exist_ok=True)
    
    stats = defaultdict(lambda: {'rgb': 0, 'thermal': 0})
    
    # Merge RGB datasets
    print("\nMerging RGB datasets...")
    for dataset in RGB_DATASETS:
        dataset_path = os.path.join(DATASETS_ROOT, dataset)
        dataset_name = Path(dataset).name
        print(f"  Processing {dataset_name}...")
        
        for split in ['train', 'val', 'test']:
            output_split = 'val' if split == 'valid' else split
            output_dir = os.path.join(OUTPUT_ROOT, output_split)
            count = copy_split(dataset_path, split, output_dir, 'rgb')
            stats[output_split]['rgb'] += count
            if count > 0:
                print(f"    {split}: {count} images")
    
    # Merge Thermal datasets
    print("\nMerging Thermal datasets...")
    for dataset in THERMAL_DATASETS:
        dataset_path = os.path.join(DATASETS_ROOT, dataset)
        dataset_name = Path(dataset).name
        print(f"  Processing {dataset_name}...")
        
        for split in ['train', 'val', 'test']:
            output_split = 'val' if split == 'valid' else split
            output_dir = os.path.join(OUTPUT_ROOT, output_split)
            count = copy_split(dataset_path, split, output_dir, 'thermal')
            stats[output_split]['thermal'] += count
            if count > 0:
                print(f"    {split}: {count} images")
    
    # Print summary
    print("\n" + "="*80)
    print("MERGE SUMMARY")
    print("="*80)
    for split in ['train', 'val', 'test']:
        rgb_count = stats[split]['rgb']
        thermal_count = stats[split]['thermal']
        total = rgb_count + thermal_count
        rgb_pct = (rgb_count / total * 100) if total > 0 else 0
        thermal_pct = (thermal_count / total * 100) if total > 0 else 0
        
        print(f"\n{split.upper()}:")
        print(f"  RGB:     {rgb_count:5d} ({rgb_pct:.1f}%)")
        print(f"  Thermal: {thermal_count:5d} ({thermal_pct:.1f}%)")
        print(f"  TOTAL:   {total:5d}")
    
    return stats

def create_yaml_config(stats):
    """Create YOLOv8 configuration YAML file"""
    config = {
        'path': OUTPUT_ROOT,
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'names': {i: name for i, name in enumerate(CLASS_NAMES)}
    }
    
    config_path = os.path.join('/home/talha/bitirmeprojesi', 'yolov8_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n✓ Created YOLOv8 config: {config_path}")
    return config_path

def verify_dataset():
    """Verify the merged dataset structure"""
    print("\n" + "="*80)
    print("VERIFYING DATASET")
    print("="*80)
    
    issues = []
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(OUTPUT_ROOT, split, 'images')
        labels_dir = os.path.join(OUTPUT_ROOT, split, 'labels')
        
        if not os.path.exists(images_dir):
            issues.append(f"Missing images directory: {images_dir}")
            continue
        if not os.path.exists(labels_dir):
            issues.append(f"Missing labels directory: {labels_dir}")
            continue
        
        images = set(os.path.splitext(f)[0] for f in os.listdir(images_dir) 
                    if f.endswith(('.jpg', '.png')))
        labels = set(os.path.splitext(f)[0] for f in os.listdir(labels_dir) 
                    if f.endswith('.txt'))
        
        # Check for orphaned files
        orphan_images = images - labels
        orphan_labels = labels - images
        
        if orphan_images:
            issues.append(f"{split}: {len(orphan_images)} images without labels")
        if orphan_labels:
            issues.append(f"{split}: {len(orphan_labels)} labels without images")
    
    if issues:
        print("\n⚠ VERIFICATION ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ All checks passed!")
    
    return len(issues) == 0

def main():
    """Main execution"""
    print("\nYOLOv8 Dataset Preparation")
    print("Unified RGB + Thermal Dataset Creation")
    print("No oversampling/undersampling - using all available data\n")
    
    # Merge datasets
    stats = merge_datasets()
    
    # Create YAML config
    create_yaml_config(stats)
    
    # Verify
    verify_dataset()
    
    print("\n" + "="*80)
    print("DATASET PREPARATION COMPLETE!")
    print("="*80)
    print(f"\nUnified dataset location: {OUTPUT_ROOT}")
    print("Next step: Run training script")
    print("  python train_yolov8_weighted.py")

if __name__ == "__main__":
    main()
