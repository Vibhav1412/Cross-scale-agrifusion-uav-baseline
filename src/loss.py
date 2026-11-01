"""
Step 6.2: Loss Functions for Imbalanced Semantic Segmentation
Implements:
- Multi-Class Soft Dice Loss
- Combined Weighted Cross-Entropy + Dice Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassDiceLoss(nn.Module):
    """
    Multi-Class Soft Dice Loss with smooth factor.
    Operates on raw network logits and integer targets.
    """
    def __init__(self, num_classes=3, smooth=1.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw outputs.
            targets: (B, H, W) integer ground truth.
        """
        # Softmax over channel dimension
        probs = F.softmax(logits, dim=1) # (B, C, H, W)
        
        # Convert targets to one-hot encoding: (B, C, H, W)
        targets_one_hot = F.one_hot(targets.clamp(0, self.num_classes - 1), num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float() # (B, C, H, W)
        
        # Spatial dimensions flatten: (B, C, H*W)
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score # (C,)
        
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            return torch.sum(dice_loss * weights) / torch.sum(weights)
        return torch.mean(dice_loss)

class CombinedLoss(nn.Module):
    """
    Hybrid Loss: Weighted Cross-Entropy + Multi-Class Soft Dice Loss.
    Addresses extreme class imbalance in early-season agricultural UAV imagery.
    """
    def __init__(
        self,
        num_classes=3,
        ce_weight=0.5,
        dice_weight=0.5,
        class_weights=None
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
            self.dice_loss = MultiClassDiceLoss(num_classes=num_classes, class_weights=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
            self.dice_loss = MultiClassDiceLoss(num_classes=num_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        total = self.ce_weight * ce + self.dice_weight * dice
        return total, ce.item(), dice.item()

