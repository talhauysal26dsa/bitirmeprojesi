# Hyperparameter Selection Guide

Quick guide to choosing optimal hyperparameters for YOLOv8 training.

## 🚀 Quick Start

### 1. Find Optimal Batch Size (2-3 minutes)

```bash
# For baseline model
python optimize_batch_size.py

# For P2 head model
python optimize_batch_size.py --p2head
```

**What it does:** Binary search to find max batch size that fits in GPU memory

### 2. Quick Parameter Test (40 minutes)

```bash
# Test baseline parameters
python quick_test.py

# Test P2 head parameters
python quick_test.py --p2head
```

**What it does:** 20-epoch training with analysis and 300-epoch prediction

### 3. Full Training (10-12 hours)

```bash
# Use recommended batch size from step 1
python train_yolov8_weighted.py --epochs 300 --batch <optimal_batch>
```

---

## 📊 Hyperparameter Decision Tree

### **Batch Size**

| GPU Memory | Baseline (3-head) | P2 Head (4-head) | Notes |
|------------|-------------------|------------------|-------|
| 80GB (A100) | 48-64 | 32-40 | Optimal |
| 40GB (A40) | 32-40 | 24-32 | Good |
| 24GB (RTX 3090) | 16-24 | 12-16 | Acceptable |
| 16GB (RTX 4060Ti) | 8-12 | 6-8 | Small batch |
| < 12GB | 4-8 | 4-6 | Too small, consider gradient accumulation |

**Rule:** P2 head uses ~30% more memory than baseline

**Auto-detect:**
```bash
python optimize_batch_size.py  # Finds optimal batch automatically
```

---

### **Learning Rate (lr0)**

Current defaults are optimized based on batch size:

- **Baseline:** `lr0=0.001` (with cosine scheduler)
- **P2 Head:** `lr0=0.0005` (backbone), `0.001` (neck), `0.005` (P2 head) - differential LR

**When to adjust:**

| Observation | Action | New lr0 |
|-------------|--------|---------|
| Loss drops >40% in 20 epochs | ✅ Keep default | - |
| Loss drops 25-40% | ✅ Good | - |
| Loss drops <25% | ⚠️ Increase LR | lr0 × 1.5 |
| Loss fluctuates wildly | ⚠️ Decrease LR | lr0 × 0.5 |
| Val loss >> Train loss (overfitting) | ⚠️ Increase weight_decay | 0.0005 → 0.001 |

**Check with quick test:**
```bash
python quick_test.py  # Will show loss reduction %
```

---

### **Augmentation Strength**

Current optimized defaults (same for both baseline and P2):

```yaml
mosaic: 1.0          # Full mosaic (best for small objects)
mixup: 0.15          # Moderate mixup
copy_paste: 0.3      # Good balance
scale: 0.5           # ±50% scale variation
close_mosaic: 10     # Disable in last 10 epochs
```

**When to adjust:**

| Issue | Solution | Parameter Change |
|-------|----------|------------------|
| Overfitting (val >> train loss) | ✅ Increase augmentation | `mixup: 0.2`, `copy_paste: 0.4` |
| Underfitting (both losses high) | ⚠️ Decrease augmentation | `mosaic: 0.8`, `mixup: 0.1` |
| Training unstable | ⚠️ Reduce augmentation | `scale: 0.3` |
| Small objects not detected | ✅ Keep mosaic high | `mosaic: 1.0` ✅ |

---

### **Image Size (imgsz)**

| Dataset | Baseline | P2 Head | Rationale |
|---------|----------|---------|-----------|
| Default | 640 | 640 | Standard YOLOv8 |
| Many small objects | 640 | 640 | P2 head handles small at 640 |
| Memory limited | 512 | 512 | Reduce memory 30% |
| Very large images | 800 | 800 | Better detail (slower) |

**Current:** `imgsz=640` (optimal for most cases)

**Memory impact:**
- 640 → 512: Save ~30% memory, allow 1.4x larger batch
- 640 → 800: Use ~50% more memory, reduce batch proportionally

---

### **Optimizer Choice**

Current defaults:

- **Baseline:** AdamW (adaptive, stable)
- **P2 Head:** AdamW (with differential LR callback)

**When to change:**

| Scenario | Optimizer | Config |
|----------|-----------|--------|
| Default (recommended) | AdamW ✅ | Current default |
| Very large batch (>64) | SGD with momentum | Add `optimizer='SGD'` |
| Memory constrained | AdamW (8-bit) | Requires separate setup |
| Faster convergence | RAdam | Add `optimizer='RAdam'` |

---

### **Loss Weights**

Current defaults:
```yaml
box: 7.5   # Localization loss
cls: 0.5   # Classification loss
dfl: 1.5   # Distribution focal loss
```

**When to adjust:**

| Problem | Solution | Change |
|---------|----------|--------|
| Poor localization (low IoU) | Increase box weight | `box: 10.0` |
| Misclassification | Increase cls weight | `cls: 1.0` |
| Class imbalance | Keep current + augmentation | Use current ✅ |

---

## 🎯 Decision Workflow

### Before Training:

1. **Run batch size optimizer** (2 mins)
   ```bash
   python optimize_batch_size.py --p2head  # or without --p2head
   ```

2. **Run quick parameter test** (40 mins)
   ```bash
   python quick_test.py --p2head  # or without --p2head
   ```

3. **Analyze results:**
   - ✅ Loss reduction >30% → Start full training
   - ⚠️ Loss reduction <30% → Adjust LR and retest
   - ⚠️ Val loss >> Train loss → Increase augmentation

### During Training:

Monitor these metrics in TensorBoard or results.csv:

```bash
# Monitor training
tensorboard --logdir runs/detect/train  # or train_p2
```

**Red flags:**

| Metric | Problem | Solution |
|--------|---------|----------|
| Loss plateaus early (<100 epochs) | LR too low | Restart with higher LR |
| Loss oscillates | LR too high or batch too small | Reduce LR or increase batch |
| mAP50 drops after 200 epochs | Overfitting | More augmentation next time |
| GPU utilization <80% | Batch too small | Increase batch size |

---

## 📈 Expected Performance Benchmarks

### Baseline (YOLOv8x, 3-head)

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| mAP50 | >0.805 | 0.810 | 0.815+ |
| mAP50-95 | >0.550 | 0.560 | 0.575+ |
| Training time (300 epochs) | ~10h | ~9h | ~8h |

### P2 Head (YOLOv8x-P2, 4-head)

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| mAP50 | >0.820 | 0.825 | 0.830+ |
| mAP50-95 | >0.570 | 0.580 | 0.595+ |
| Small object mAP | +5-7% vs baseline | +7-9% | +10%+ |
| Training time (300 epochs) | ~12h | ~11h | ~10h |

**Note:** P2 should beat baseline by +2-3% mAP50 minimum

---

## 🔧 Troubleshooting

### "CUDA out of memory"

1. Reduce batch size: `--batch 16` → `--batch 8`
2. Or reduce image size: `--imgsz 640` → `--imgsz 512`
3. Or run optimizer: `python optimize_batch_size.py`

### Training too slow

1. Check GPU utilization: `nvidia-smi -l 1`
2. If <80%: Increase batch size
3. If 100%: Hardware bottleneck, expected

### Loss not decreasing

1. Check learning rate: Should decay from 0.001 → 0.00001
2. Run quick test: `python quick_test.py`
3. If <25% reduction: Try `lr0=0.002`

### Overfitting (val >> train loss)

1. Increase augmentation: Already at good defaults
2. More data: Add more training samples
3. Early stopping: Use `patience=50` instead of 100

---

## 📝 Summary: What to Run

### First Time Setup:

```bash
# 1. Find optimal batch size (2 mins)
python optimize_batch_size.py

# 2. Test parameters (40 mins)  
python quick_test.py

# 3. If test looks good, full training (10-12 hours)
python train_yolov8_weighted.py --epochs 300 --batch <optimal>
```

### For P2 Head:

```bash
# 1. Find batch size for P2
python optimize_batch_size.py --p2head

# 2. Test P2 parameters
python quick_test.py --p2head

# 3. Full P2 training
python train_yolov8_weighted.py --p2head --epochs 300 --batch <optimal>
```

**Total time investment:**
- Setup: 42 mins (batch opt + quick test)
- Full training: 10-12 hours
- **Total: ~11-12 hours** (vs 12+ hours blind trial)

**Saves:** Avoids failed 12-hour trainings from bad hyperparameters!
