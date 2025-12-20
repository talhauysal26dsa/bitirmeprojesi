"""
Advanced Training Callbacks for Adaptive Learning
Implements ReduceLROnPlateau, EMA, and other advanced techniques
"""
from ultralytics.utils import LOGGER
import torch
import torch.nn as nn
from copy import deepcopy

class ReduceLROnPlateauCallback:
    """
    Reduce learning rate when validation metric plateaus
    
    Similar to torch.optim.lr_scheduler.ReduceLROnPlateau but as YOLO callback
    """
    
    def __init__(self, monitor='metrics/mAP50(B)', patience=10, factor=0.5, min_lr=1e-7, verbose=True):
        """
        Args:
            monitor: Metric to monitor (e.g., 'metrics/mAP50(B)')
            patience: Epochs to wait before reducing LR
            factor: Factor to reduce LR by (new_lr = lr * factor)
            min_lr: Minimum learning rate
            verbose: Print when LR is reduced
        """
        self.monitor = monitor
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.best_metric = None
        self.wait = 0
        self.lr_reductions = 0
        
    def on_fit_epoch_end(self, trainer):
        """Check if we should reduce LR after validation"""
        metrics = trainer.metrics
        
        if self.monitor not in metrics:
            return
        
        current_metric = metrics[self.monitor]
        
        # Initialize best metric
        if self.best_metric is None:
            self.best_metric = current_metric
            return
        
        # Check if improved
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.wait = 0
        else:
            self.wait += 1
            
            # Reduce LR if patience exceeded
            if self.wait >= self.patience:
                self._reduce_lr(trainer)
                self.wait = 0
    
    def _reduce_lr(self, trainer):
        """Reduce learning rate for all parameter groups"""
        optimizer = trainer.optimizer
        
        for i, param_group in enumerate(optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            
            if new_lr < old_lr:
                param_group['lr'] = new_lr
                self.lr_reductions += 1
                
                if self.verbose:
                    group_name = param_group.get('name', f'group_{i}')
                    LOGGER.info(f"⚡ ReduceLROnPlateau: {group_name} LR {old_lr:.6f} → {new_lr:.6f} "
                               f"(no improvement for {self.patience} epochs)")


class EMACallback:
    """
    Exponential Moving Average of model weights
    
    Maintains shadow weights that are averaged over time
    Results in smoother, more stable predictions
    """
    
    def __init__(self, decay=0.9999, update_period=1):
        """
        Args:
            decay: EMA decay rate (higher = slower update)
            update_period: Update EMA every N batches
        """
        self.decay = decay
        self.update_period = update_period
        self.ema_model = None
        self.updates = 0
        
    def on_train_start(self, trainer):
        """Initialize EMA model"""
        self.ema_model = deepcopy(trainer.model).eval()
        
        # Freeze EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False
        
        LOGGER.info(f"✓ EMA initialized (decay={self.decay})")
    
    def on_train_batch_end(self, trainer):
        """Update EMA weights after each batch"""
        self.updates += 1
        
        if self.updates % self.update_period == 0:
            self._update_ema(trainer.model)
    
    def _update_ema(self, model):
        """Update EMA weights"""
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
    
    def on_fit_epoch_end(self, trainer):
        """Log EMA status"""
        if trainer.epoch % 10 == 0:
            LOGGER.info(f"📊 EMA updates: {self.updates} (decay={self.decay:.4f})")


class GradientClippingCallback:
    """
    Gradient clipping to prevent exploding gradients
    """
    
    def __init__(self, max_norm=10.0, norm_type=2.0):
        """
        Args:
            max_norm: Maximum gradient norm
            norm_type: Type of norm (2.0 = L2 norm)
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.clip_count = 0
        
    def on_before_optimizer_step(self, trainer):
        """Clip gradients before optimizer step"""
        if self.max_norm <= 0:
            return
        
        model = trainer.model
        
        # Compute gradient norm
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )
        
        # Track clipping
        if total_norm > self.max_norm:
            self.clip_count += 1
            
            if self.clip_count % 100 == 0:
                LOGGER.info(f"✂️ Gradient clipped {self.clip_count} times (norm={total_norm:.2f} > {self.max_norm})")


class OneCycleLRCallback:
    """
    OneCycle Learning Rate policy
    
    Fast convergence with:
    1. Warmup phase: LR increases from low to high
    2. Annealing phase: LR decreases to very low
    """
    
    def __init__(self, max_lr=0.01, total_steps=None, pct_start=0.3, div_factor=25, final_div_factor=1e4):
        """
        Args:
            max_lr: Maximum learning rate
            total_steps: Total training steps (epochs * steps_per_epoch)
            pct_start: Percentage of cycle spent increasing LR
            div_factor: Initial LR = max_lr / div_factor
            final_div_factor: Final LR = max_lr / final_div_factor
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.current_step = 0
        
    def on_train_start(self, trainer):
        """Initialize OneCycle scheduler"""
        if self.total_steps is None:
            self.total_steps = trainer.epochs * len(trainer.train_loader)
        
        LOGGER.info(f"✓ OneCycle LR: {self.max_lr/self.div_factor:.6f} → {self.max_lr:.6f} → {self.max_lr/self.final_div_factor:.6f}")
        LOGGER.info(f"  Total steps: {self.total_steps}, Warmup: {int(self.total_steps * self.pct_start)}")
    
    def on_train_batch_end(self, trainer):
        """Update LR according to OneCycle policy"""
        self.current_step += 1
        lr = self._get_lr()
        
        # Update all parameter groups
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self):
        """Compute current LR based on OneCycle policy"""
        step = self.current_step
        total = self.total_steps
        pct = step / total
        
        # Phase 1: Warmup (0 to pct_start)
        if pct < self.pct_start:
            # Linear increase
            lr_scale = pct / self.pct_start
            lr = self.max_lr / self.div_factor + lr_scale * (self.max_lr - self.max_lr / self.div_factor)
        
        # Phase 2: Annealing (pct_start to 1.0)
        else:
            # Cosine annealing
            pct_anneal = (pct - self.pct_start) / (1.0 - self.pct_start)
            lr_scale = (1 + torch.cos(torch.tensor(pct_anneal * 3.14159))) / 2
            lr = self.max_lr / self.final_div_factor + lr_scale * (self.max_lr - self.max_lr / self.final_div_factor)
        
        return float(lr)


class WarmRestartCallback:
    """
    Cosine Annealing with Warm Restarts (SGDR)
    
    Periodically restarts LR to escape local minima
    """
    
    def __init__(self, T_0=50, T_mult=2, eta_min=1e-6):
        """
        Args:
            T_0: Initial restart period (epochs)
            T_mult: Multiply restart period by this after each restart
            eta_min: Minimum learning rate
        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        
        self.T_cur = 0
        self.T_i = T_0
        self.restart_count = 0
        self.initial_lrs = {}
        
    def on_train_start(self, trainer):
        """Save initial learning rates"""
        for i, param_group in enumerate(trainer.optimizer.param_groups):
            self.initial_lrs[i] = param_group['lr']
        
        LOGGER.info(f"✓ Warm Restarts: T_0={self.T_0}, T_mult={self.T_mult}")
    
    def on_train_epoch_end(self, trainer):
        """Update LR with warm restarts"""
        self.T_cur += 1
        
        # Check if restart
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult
            self.restart_count += 1
            LOGGER.info(f"🔄 Warm Restart #{self.restart_count} (next in {self.T_i} epochs)")
        
        # Compute cosine annealed LR
        for i, param_group in enumerate(trainer.optimizer.param_groups):
            initial_lr = self.initial_lrs[i]
            lr = self.eta_min + (initial_lr - self.eta_min) * (1 + torch.cos(torch.tensor(3.14159 * self.T_cur / self.T_i))) / 2
            param_group['lr'] = float(lr)


# Usage documentation
USAGE_EXAMPLE = """
# Advanced Adaptive Learning Rate Usage

from train_p2h_adaptive_callbacks import (
    ReduceLROnPlateauCallback,
    EMACallback,
    GradientClippingCallback,
    OneCycleLRCallback,
    WarmRestartCallback
)
from ultralytics import YOLO

model = YOLO('yolov8x-p2-custom.yaml')

# Option 1: ReduceLROnPlateau (safe, proven)
plateau_cb = ReduceLROnPlateauCallback(
    monitor='metrics/mAP50(B)',
    patience=10,
    factor=0.5,
    min_lr=1e-7
)
model.add_callback('on_fit_epoch_end', plateau_cb.on_fit_epoch_end)

# Option 2: OneCycle (fast convergence)
onecycle_cb = OneCycleLRCallback(
    max_lr=0.01,
    total_steps=None,  # Auto-computed
    pct_start=0.3
)
model.add_callback('on_train_start', onecycle_cb.on_train_start)
model.add_callback('on_train_batch_end', onecycle_cb.on_train_batch_end)

# Option 3: Warm Restarts (escape local minima)
restart_cb = WarmRestartCallback(T_0=50, T_mult=2)
model.add_callback('on_train_start', restart_cb.on_train_start)
model.add_callback('on_train_epoch_end', restart_cb.on_train_epoch_end)

# EMA (always recommended)
ema_cb = EMACallback(decay=0.9999)
model.add_callback('on_train_start', ema_cb.on_train_start)
model.add_callback('on_train_batch_end', ema_cb.on_train_batch_end)

# Gradient Clipping (for stability)
clip_cb = GradientClippingCallback(max_norm=10.0)
model.add_callback('on_before_optimizer_step', clip_cb.on_before_optimizer_step)

# Train
model.train(data='yolov8_config.yaml', epochs=300)
"""

if __name__ == "__main__":
    print(USAGE_EXAMPLE)
