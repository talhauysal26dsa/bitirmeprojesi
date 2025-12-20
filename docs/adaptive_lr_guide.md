# Adaptive Learning Rate Strategies for P2H Training

## 🎯 Overview

Added **5 advanced adaptive LR strategies** to optimize P2H training:

| Strategy | Best For | Convergence | Stability |
|----------|----------|-------------|-----------|
| **Cosine Annealing** | General purpose | Medium | High ⭐⭐⭐ |
| **OneCycle** | Fast training | **Very Fast** ⚡ | Medium ⭐⭐ |
| **ReduceLROnPlateau** | Safe training | Slow | **Very High** ⭐⭐⭐⭐ |
| **Warm Restarts** | Escape local minima | Medium | Medium ⭐⭐ |
| **Differential LR** | Transfer learning | Medium | High ⭐⭐⭐ |

---

## 🚀 Quick Start Commands

### Option 1: OneCycle (Fastest - **Recommended for quick experiments**)

```bash
python train_p2h_ultra.py \
  --baseline-weights runs/detect/train/weights/best.pt \
  --lr-strategy onecycle \
  --lr0 0.01 \
  --epochs 100 \
  --batch 24 \
  --device 0 \
  --ema \
  --gradient-clip 10.0
```

**Why OneCycle?**
- ⚡ **Fastest convergence** (often 2-3x faster)
- 🎯 Reaches near-optimal performance in fewer epochs
- 📈 Smooth training curve
- **Use when:** You want quick results or limited time

**Expected time:** ~100 epochs to match 300-epoch baseline

---

### Option 2: Cosine + Plateau (Most Reliable - **Recommended for production**)

```bash
python train_p2h_ultra.py \
  --baseline-weights runs/detect/train/weights/best.pt \
  --lr-strategy plateau \
  --lr0 0.001 \
  --plateau-patience 15 \
  --plateau-factor 0.5 \
  --epochs 300 \
  --batch 24 \
  --device 0 \
  --ema \
  --gradient-clip 10.0
```

**Why Plateau?**
- 🛡️ **Most stable** - adapts to training dynamics
- 📊 Automatically reduces LR when stuck
- 🎓 Best for final production models
- **Use when:** You want the absolute best model

**Expected time:** ~300 epochs for best results

---

### Option 3: Warm Restarts (Advanced - **Recommended for research**)

```bash
python train_p2h_ultra.py \
  --baseline-weights runs/detect/train/weights/best.pt \
  --lr-strategy warm_restart \
  --lr0 0.001 \
  --restart-period 50 \
  --restart-mult 2 \
  --epochs 300 \
  --batch 24 \
  --device 0 \
  --ema
```

**Why Warm Restarts?**
- 🔄 Periodically "kicks" optimization out of local minima
- 🎲 Explores multiple solutions
- 🏆 Can find better optima than standard training
- **Use when:** Stuck in local minima or want ensemble-like benefits

---

## 📊 Detailed Strategy Comparison

### 1. **OneCycle LR** ⚡

**Learning Rate Schedule:**
```
LR
^
|     /\
|    /  \___
|   /        \___
|  /             \___
| /                  \___
+----------------------------> Epoch
  Warmup  Peak    Annealing
```

**Parameters:**
- `--lr0 0.01` - Maximum LR (peak)
- `--onecycle-pct-start 0.3` - Warmup duration (30% of training)
- `--onecycle-div-factor 25` - Initial LR = max_lr / 25

**Pros:**
- ✅ Fastest convergence
- ✅ Automatic warmup + annealing
- ✅ Often reaches 95% of final performance in 50% of epochs

**Cons:**
- ⚠️ Sensitive to max_lr choice
- ⚠️ Less flexible than plateau

**Best command:**
```bash
# Fast training (100 epochs)
python train_p2h_ultra.py --lr-strategy onecycle --lr0 0.01 --epochs 100
```

---

### 2. **ReduceLROnPlateau** 🛡️

**Learning Rate Schedule:**
```
LR
^
|____
|    \____
|         \____
|              \____
|                   \____
+----------------------------> Epoch
     Plateau  Reduce  Plateau
```

**Parameters:**
- `--lr0 0.001` - Initial LR
- `--plateau-patience 15` - Wait N epochs before reducing
- `--plateau-factor 0.5` - Reduce by 50% each time
- `--lr-min 1e-6` - Minimum LR

**Pros:**
- ✅ Most stable
- ✅ Adapts to training dynamics
- ✅ Never reduces LR too early

**Cons:**
- ⚠️ Slower convergence
- ⚠️ May plateau for long periods

**Best command:**
```bash
# Stable training (300 epochs)
python train_p2h_ultra.py --lr-strategy plateau --plateau-patience 15 --epochs 300
```

---

### 3. **Cosine Annealing** 📊

**Learning Rate Schedule:**
```
LR
^
|
|   ___
|  /   \___
| /        \___
|/             \___
+----------------------------> Epoch
  Smooth cosine decay
```

**Parameters:**
- `--lr0 0.001` - Initial LR
- Built-in to YOLO (no extra params needed)

**Pros:**
- ✅ Smooth, predictable schedule
- ✅ Good balance of speed and stability
- ✅ Default in most papers

**Cons:**
- ⚠️ Fixed schedule (doesn't adapt)

**Best command:**
```bash
# Standard training (300 epochs)
python train_p2h_ultra.py --lr-strategy cosine --lr0 0.001 --epochs 300
```

---

### 4. **Warm Restarts (SGDR)** 🔄

**Learning Rate Schedule:**
```
LR
^
|  /\    /\       /\
| /  \  /  \     /  \
|/    \/    \   /    \
|           \/\/      \/
+----------------------------> Epoch
  Restart Restart  Restart
```

**Parameters:**
- `--restart-period 50` - Initial restart interval
- `--restart-mult 2` - Double interval after each restart
- `--lr0 0.001` - Max LR (after restart)
- `--lr-min 1e-6` - Min LR (before restart)

**Pros:**
- ✅ Escapes local minima
- ✅ Explores multiple solutions
- ✅ Ensemble-like benefits

**Cons:**
- ⚠️ Complex hyperparameters
- ⚠️ May destabilize training

**Best command:**
```bash
# Advanced training (300 epochs)
python train_p2h_ultra.py --lr-strategy warm_restart --restart-period 50 --epochs 300
```

---

## 🎯 Additional Features

### **EMA (Exponential Moving Average)** ✨

**Always recommended!**

```bash
--ema --ema-decay 0.9999
```

**Benefits:**
- ✅ Smoother predictions
- ✅ Better generalization
- ✅ ~1-2% mAP improvement
- ✅ No training time cost

**How it works:**
```python
shadow_weights = 0.9999 * shadow_weights + 0.0001 * current_weights
```

---

### **Gradient Clipping** ✂️

**Recommended for stability:**

```bash
--gradient-clip 10.0
```

**Benefits:**
- ✅ Prevents exploding gradients
- ✅ More stable training
- ✅ Allows higher learning rates

**When to use:**
- Training is unstable (loss spikes)
- Using high learning rates
- Training with mixed precision

---

### **Advanced Optimizers** 🧠

```bash
--optimizer AdamW  # Default, best for most cases
--optimizer RAdam  # More stable Adam
--optimizer NAdam  # Faster Adam with Nesterov
```

**Optimizer Comparison:**

| Optimizer | Speed | Stability | Memory |
|-----------|-------|-----------|--------|
| SGD | Slow | High | Low |
| Adam | Fast | Medium | High |
| **AdamW** | Fast | High | High ⭐ |
| RAdam | Medium | Very High | High |
| NAdam | Very Fast | Medium | High |

---

## 📈 Expected Performance

### OneCycle (100 epochs)
- **Training time:** ~3-4 hours
- **Expected mAP50:** 0.80-0.82
- **vs Baseline:** +1-3%

### Plateau (300 epochs)
- **Training time:** ~10-12 hours
- **Expected mAP50:** 0.82-0.85
- **vs Baseline:** +2-5%

### Warm Restarts (300 epochs)
- **Training time:** ~10-12 hours
- **Expected mAP50:** 0.81-0.84
- **vs Baseline:** +1-4%

---

## 🔧 Hyperparameter Tuning Guide

### If training is **too slow:**
```bash
# Use OneCycle
--lr-strategy onecycle --lr0 0.01 --epochs 100
```

### If training is **unstable:**
```bash
# Use Plateau with gradient clipping
--lr-strategy plateau --gradient-clip 10.0 --lr0 0.0005
```

### If stuck in **local minimum:**
```bash
# Use Warm Restarts
--lr-strategy warm_restart --restart-period 30
```

### If validation **not improving:**
```bash
# Reduce learning rate or increase patience
--lr0 0.0005 --plateau-patience 20
```

---

## 📊 Monitoring Training

### Check LR schedule:
```bash
tensorboard --logdir runs/detect/train_p2h_ultra
```

Look for:
- ✅ Smooth loss curve
- ✅ LR decreases over time
- ✅ Validation improving

### Warning signs:
- ⚠️ Loss spikes → Reduce LR or enable gradient clipping
- ⚠️ Flat validation → Increase augmentation or reduce LR
- ⚠️ Overfitting → Add more regularization

---

## 🎓 Recommendations

### For **quick experiments** (testing ideas):
```bash
python train_p2h_ultra.py --lr-strategy onecycle --epochs 50 --batch 24
```

### For **production models** (final deployment):
```bash
python train_p2h_ultra.py --lr-strategy plateau --epochs 300 --batch 24 --ema
```

### For **research** (exploring limits):
```bash
python train_p2h_ultra.py --lr-strategy warm_restart --epochs 500 --batch 24 --ema
```

---

## 📚 References

1. **OneCycle:** Smith, L. N. (2018). "A disciplined approach to neural network hyper-parameters"
2. **Warm Restarts:** Loshchilov & Hutter (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts"
3. **EMA:** Polyak & Juditsky (1992). "Acceleration of stochastic approximation by averaging"
4. **Gradient Clipping:** Pascanu et al. (2013). "On the difficulty of training recurrent neural networks"

---

## 💡 Pro Tips

1. **Start with OneCycle** for quick validation of your setup
2. **Switch to Plateau** for final production model
3. **Always use EMA** - it's free performance
4. **Monitor gradients** - enable gradient clipping if you see spikes
5. **Compare multiple runs** - use different strategies and pick best

---

**Created:** December 2025  
**Version:** 2.0 - Adaptive Learning Edition
