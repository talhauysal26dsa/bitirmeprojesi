"""
Custom Training Callbacks for P2H Optimization
Implements differential learning rates for optimal P2H training
"""
from ultralytics.utils import LOGGER
import torch

class P2HDifferentialLRCallback:
    """
    Callback for differential learning rates in P2H training
    
    Applies different learning rates to:
    - Backbone (pretrained): Lower LR
    - Neck (pretrained): Medium LR  
    - P2 Head (new): Higher LR
    """
    
    def __init__(self, lr_backbone=0.001, lr_neck=0.002, lr_p2=0.01):
        self.lr_backbone = lr_backbone
        self.lr_neck = lr_neck
        self.lr_p2 = lr_p2
        self.initial_setup = False
        
    def on_train_start(self, trainer):
        """Setup differential learning rates when training starts"""
        if self.initial_setup:
            return
            
        model = trainer.model
        optimizer = trainer.optimizer
        
        # Group parameters by layer type
        backbone_params = []
        neck_params = []
        p2_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Identify layer type by name pattern
            # Use startswith for exact matching (model.10 vs model.1)
            if (name.startswith('model.0.') or name.startswith('model.1.') or 
                name.startswith('model.2.')):
                # Early backbone layers
                backbone_params.append(param)
            elif (name.startswith('model.3.') or name.startswith('model.4.') or 
                  name.startswith('model.5.')):
                # Mid backbone layers
                backbone_params.append(param)
            elif (name.startswith('model.6.') or name.startswith('model.7.') or 
                  name.startswith('model.8.') or name.startswith('model.9.')):
                # Deep backbone layers
                backbone_params.append(param)
            elif (name.startswith('model.10.') or name.startswith('model.11.') or 
                  name.startswith('model.12.')):
                # Neck layers (P4/P5 path)
                neck_params.append(param)
            elif (name.startswith('model.13.') or name.startswith('model.14.') or 
                  name.startswith('model.15.')):
                # Neck layers (P3 path)
                neck_params.append(param)
            elif (name.startswith('model.16.') or name.startswith('model.17.') or 
                  name.startswith('model.18.')):
                # P2 head layers (NEW - these are the critical small object layers)
                p2_params.append(param)
            else:
                # Detection head and other layers
                other_params.append(param)
        
        # Update optimizer parameter groups
        if hasattr(optimizer, 'param_groups'):
            # Clear existing param groups
            optimizer.param_groups.clear()
            
            # Add new param groups with differential LRs
            if backbone_params:
                optimizer.add_param_group({
                    'params': backbone_params,
                    'lr': self.lr_backbone,
                    'initial_lr': self.lr_backbone,  # Required by YOLO
                    'name': 'backbone'
                })
                
            if neck_params:
                optimizer.add_param_group({
                    'params': neck_params,
                    'lr': self.lr_neck,
                    'initial_lr': self.lr_neck,  # Required by YOLO
                    'name': 'neck'
                })
                
            if p2_params:
                optimizer.add_param_group({
                    'params': p2_params,
                    'lr': self.lr_p2,
                    'initial_lr': self.lr_p2,  # Required by YOLO
                    'name': 'p2_head'
                })
                
            if other_params:
                optimizer.add_param_group({
                    'params': other_params,
                    'lr': self.lr_neck,  # Use neck LR for detection heads
                    'initial_lr': self.lr_neck,  # Required by YOLO
                    'name': 'other'
                })
            
            LOGGER.info(f"✓ Differential LR setup:")
            LOGGER.info(f"  Backbone: {len(backbone_params)} params @ LR={self.lr_backbone}")
            LOGGER.info(f"  Neck:     {len(neck_params)} params @ LR={self.lr_neck}")
            LOGGER.info(f"  P2 Head:  {len(p2_params)} params @ LR={self.lr_p2}")
            LOGGER.info(f"  Other:    {len(other_params)} params @ LR={self.lr_neck}")
            
            self.initial_setup = True


# Usage Example
if __name__ == "__main__":
    print("""
# Usage Example for P2HDifferentialLRCallback

from train_p2h_callbacks import P2HDifferentialLRCallback
from ultralytics import YOLO

# Create model
model = YOLO('yolov8x-p2-custom.yaml')

# Setup differential LR callback
diff_lr_callback = P2HDifferentialLRCallback(
    lr_backbone=0.0001,  # Low LR for pretrained backbone
    lr_neck=0.0003,      # Medium LR for neck
    lr_p2=0.0006         # Higher LR for new P2 head
)

# Register callback
model.add_callback('on_train_start', diff_lr_callback.on_train_start)

# Train
model.train(
    data='yolov8_config.yaml',
    epochs=300,
    # ... other params
)
""")
