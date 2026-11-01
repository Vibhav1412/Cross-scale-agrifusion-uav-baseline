"""
Step 6.1: Segmentation Metrics Suite
Calculates Intersection over Union (IoU), Dice Coefficient (F1-Score),
Precision, Recall, and Confusion Matrix across all semantic classes.
"""

import numpy as np
import torch

class SegmentationMetrics:
    """
    Accumulates confusion matrix over batches to compute macro and per-class:
    - IoU (Jaccard Index)
    - Dice Coefficient (F1 Score)
    - Precision
    - Recall
    """
    def __init__(self, num_classes=3, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or ["soil", "crop", "weed"]
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds (Tensor): Predicted class indices of shape (B, H, W) or logits (B, C, H, W).
            targets (Tensor): Ground truth class indices of shape (B, H, W).
        """
        if preds.dim() == 4:
            preds = torch.argmax(preds, dim=1)
            
        preds_np = preds.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()
        
        # Valid mask (0 <= target < num_classes)
        valid = (targets_np >= 0) & (targets_np < self.num_classes)
        p = preds_np[valid]
        t = targets_np[valid]
        
        # Fast 2D bincount for confusion matrix: rows=ground_truth, cols=prediction
        bins = self.num_classes * t + p
        counts = np.bincount(bins, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self):
        cm = self.confusion_matrix
        eps = 1e-7
        
        # True Positives: diagonal
        tp = np.diag(cm)
        # False Positives: sum along columns minus TP
        fp = np.sum(cm, axis=0) - tp
        # False Negatives: sum along rows minus TP
        fn = np.sum(cm, axis=1) - tp
        
        # Per-class metrics
        iou = tp / (tp + fp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        
        # Macro averages
        miou = float(np.mean(iou))
        mdice = float(np.mean(dice))
        mprecision = float(np.mean(precision))
        mrecall = float(np.mean(recall))
        
        results = {
            "miou": miou,
            "mdice": mdice,
            "mprecision": mprecision,
            "mrecall": mrecall,
            "iou_per_class": {name: float(iou[i]) for i, name in enumerate(self.class_names)},
            "dice_per_class": {name: float(dice[i]) for i, name in enumerate(self.class_names)},
            "precision_per_class": {name: float(precision[i]) for i, name in enumerate(self.class_names)},
            "recall_per_class": {name: float(recall[i]) for i, name in enumerate(self.class_names)},
            "confusion_matrix": cm.tolist()
        }
        return results

