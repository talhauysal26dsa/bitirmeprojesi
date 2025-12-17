# SAHI Inference Scripts - Usage Guide

## Scripts Created

1. **inference_baseline_sahi.py** - Baseline model with SAHI
2. **inference_p2h_sahi.py** - P2H model with SAHI
3. **inference_comparison_sahi.py** - Compare all combinations

## Installation

```bash
# Install SAHI
pip install sahi

# Or in your venv
/home/ilaha/bitirmeprojesi/venv/bin/pip install sahi
```

## Usage Examples

### 1. Baseline + SAHI
```bash
python inference_baseline_sahi.py \
  --source /path/to/test/images \
  --slice-height 512 \
  --slice-width 512 \
  --overlap-height 0.2 \
  --overlap-width 0.2 \
  --save-vis \
  --save-json
```

### 2. P2H + SAHI
```bash
python inference_p2h_sahi.py \
  --source /path/to/test/images \
  --slice-height 512 \
  --slice-width 512 \
  --overlap-height 0.2 \
  --overlap-width 0.2 \
  --save-vis \
  --save-json
```

### 3. Full Comparison
```bash
python inference_comparison_sahi.py \
  --source /path/to/test/images \
  --slice-height 512 \
  --slice-width 512 \
  --overlap-height 0.2 \
  --overlap-width 0.2
```

## Parameters to Experiment

### Slice Size
- Small (256x256): More slices, slower, better for tiny objects
- Medium (512x512): **Recommended** - balanced
- Large (896x896): Fewer slices, faster, may miss small objects

### Overlap Ratio
- Low (0.1): Faster, may miss objects at boundaries
- Medium (0.2): **Recommended** - balanced
- High (0.3): Slower, better coverage

## Output Structure

```
runs/sahi/
├── baseline/
│   ├── json/           # Detection results
│   └── visualizations/ # Annotated images
├── p2h/
│   ├── json/
│   └── visualizations/
└── comparison/
    ├── comparison_results.csv
    └── comparison_plots.png
```

## Next Steps

1. Wait for P2H training to complete
2. Install SAHI: `pip install sahi`
3. Run comparison on test set
4. Analyze results and optimize parameters
