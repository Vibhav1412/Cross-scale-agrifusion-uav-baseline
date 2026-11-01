"""
Step 9.2: Uncertainty Estimation for UAV Segmentation
Implements Monte Carlo (MC) Dropout and Pixel-Wise Predictive Entropy.
Quantifies epistemic and boundary uncertainty to separate high-confidence
predictions from ambiguous crop-weed decision zones.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def estimate_uncertainty_mc_dropout(model, image_tensor: torch.Tensor, num_passes: int = 8):
    """
    Computes pixel-wise predictive entropy using Monte Carlo Dropout.
    Args:
        model (nn.Module): Segmentation model with MC dropout support.
        image_tensor (Tensor): (1, 3, H, W) normalized image tensor.
        num_passes (int): Number of stochastic forward passes (default 8).
    Returns:
        mean_probs (np.ndarray): (H, W, num_classes) averaged probabilities.
        predictive_entropy (np.ndarray): (H, W) normalized entropy in [0, 1].
        variance_map (np.ndarray): (H, W) predictive variance.
        pred_mask (np.ndarray): (H, W) hard prediction mask.
    """
    model.eval()
    if hasattr(model, "enable_mc_dropout"):
        model.enable_mc_dropout()
    else:
        # Fallback: manually enable Dropout layers
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout2d) or isinstance(m, torch.nn.Dropout):
                m.train()

    prob_passes = []
    
    with torch.no_grad():
        for _ in range(num_passes):
            logits = model(image_tensor) # (1, num_classes, H, W)
            probs = F.softmax(logits, dim=1) # (1, num_classes, H, W)
            prob_passes.append(probs.squeeze(0).permute(1, 2, 0).cpu().numpy()) # (H, W, C)

    # Convert to array: (T, H, W, C)
    prob_passes = np.stack(prob_passes, axis=0)
    
    # 1. Mean probability across stochastic passes: (H, W, C)
    mean_probs = np.mean(prob_passes, axis=0)
    
    # 2. Hard prediction mask: argmax of mean probability
    pred_mask = np.argmax(mean_probs, axis=-1)
    
    # 3. Normalized Predictive Entropy: H(p) = - sum(p * log(p)) / log(num_classes)
    eps = 1e-8
    num_classes = mean_probs.shape[-1]
    entropy = -np.sum(mean_probs * np.log(mean_probs + eps), axis=-1)
    normalized_entropy = np.clip(entropy / np.log(num_classes), 0.0, 1.0)
    
    # 4. Variance across passes (epistemic uncertainty proxy)
    variance_map = np.mean(np.var(prob_passes, axis=0), axis=-1)

    return mean_probs, normalized_entropy, variance_map, pred_mask

def generate_uncertainty_visualization(
    rgb_display_np: np.ndarray,
    pred_mask: np.ndarray,
    entropy_map: np.ndarray,
    output_path: str = None,
    uncertainty_threshold: float = 0.35
):
    """
    Plots original image, predicted segmentation, predictive entropy heatmap,
    and high-uncertainty boundary zones.
    """
    palette = {0: [0, 0, 0], 1: [0, 255, 0], 2: [255, 0, 0]}
    pred_color = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for c, rgb_val in palette.items():
        pred_color[pred_mask == c] = rgb_val

    # High uncertainty binary mask
    uncertain_regions = entropy_map > uncertainty_threshold
    uncertainty_pct = (np.sum(uncertain_regions) / entropy_map.size) * 100.0

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(rgb_display_np)
        axes[0].set_title("Input UAV RGB Image")
        axes[0].axis("off")
        
        axes[1].imshow(pred_color)
        axes[1].set_title("Predicted Segmentation (Green:Crop, Red:Weed)")
        axes[1].axis("off")
        
        im_ent = axes[2].imshow(entropy_map, cmap="plasma", vmin=0, vmax=1.0)
        axes[2].set_title("Pixel-Wise Predictive Entropy [0, 1]")
        axes[2].axis("off")
        plt.colorbar(im_ent, ax=axes[2], fraction=0.046, pad=0.04)
        
        # High uncertainty overlay
        unc_overlay = rgb_display_np.copy()
        unc_overlay[uncertain_regions] = [255, 255, 0] # Yellow for high uncertainty
        axes[3].imshow(unc_overlay)
        axes[3].set_title(f"Uncertain Boundaries ({uncertainty_pct:.1f}% Yellow)")
        axes[3].axis("off")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    return {
        "mean_entropy": round(float(np.mean(entropy_map)), 4),
        "max_entropy": round(float(np.max(entropy_map)), 4),
        "high_uncertainty_pixel_pct": round(float(uncertainty_pct), 2),
        "uncertainty_threshold": uncertainty_threshold
    }

