"""
Step 6.4: Full-Resolution & Patch-Based Inference Engine
Supports:
- Individual and batch image inference.
- Arbitrary high-resolution UAV images via sliding-window patch inference.
- Seamless reconstruction of predictions into original image dimensions.
- Side-by-side visualization generation.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from models.unet import build_unet
from src.analytics import compute_field_coverage

def load_inference_model(checkpoint_path=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = checkpoint_path or os.path.join(PROJECT_ROOT, "outputs/checkpoints/unet_best.pth")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {ckpt_path}")
        
    model = build_unet(num_classes=3, pretrained=False)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    return model, device

def predict_patch(model, patch_tensor, device):
    """Predicts a single standardized (1, 3, 256, 256) patch."""
    with torch.no_grad():
        logits = model(patch_tensor.to(device))
        probs = F.softmax(logits, dim=1)
        return probs.squeeze(0).cpu().numpy() # (3, H, W)

def predict_large_image(
    model,
    image_pil: Image.Image,
    device,
    patch_size: int = 256,
    stride: int = 192,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
):
    """
    Performs sliding-window patch-based inference with overlap on high-resolution UAV imagery.
    Reconstructs full probability maps and prediction masks into original dimensions.
    """
    img_rgb = image_pil.convert("RGB")
    w, h = img_rgb.size
    
    # Probability accumulator and weight counter for overlapping patches
    prob_map = np.zeros((3, h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    
    # 2D Hanning window for smooth edge blending
    win_y = np.hanning(patch_size)
    win_x = np.hanning(patch_size)
    patch_weight = np.outer(win_y, win_x) + 1e-4 # Avoid divide by zero
    
    # Generate grid coordinates
    y_starts = list(range(0, max(1, h - patch_size + 1), stride))
    if y_starts[-1] + patch_size < h:
        y_starts.append(h - patch_size)
        
    x_starts = list(range(0, max(1, w - patch_size + 1), stride))
    if x_starts[-1] + patch_size < w:
        x_starts.append(w - patch_size)
        
    # If image is smaller than patch_size, pad or resize
    if h < patch_size or w < patch_size:
        resized = TF.resize(img_rgb, (patch_size, patch_size))
        t = TF.normalize(TF.to_tensor(resized), mean=mean, std=std).unsqueeze(0)
        p = predict_patch(model, t, device)
        # Upsample back
        p_tensor = torch.from_numpy(p).unsqueeze(0)
        p_up = F.interpolate(p_tensor, size=(h, w), mode="bilinear", align_corners=False).squeeze(0).numpy()
        pred_mask = np.argmax(p_up, axis=0)
        return pred_mask, p_up

    # Iterate over sliding windows
    for y in y_starts:
        for x in x_starts:
            crop = img_rgb.crop((x, y, x + patch_size, y + patch_size))
            crop_tensor = TF.normalize(TF.to_tensor(crop), mean=mean, std=std).unsqueeze(0)
            
            patch_probs = predict_patch(model, crop_tensor, device) # (3, patch_size, patch_size)
            
            prob_map[:, y:y + patch_size, x:x + patch_size] += patch_probs * patch_weight
            weight_map[y:y + patch_size, x:x + patch_size] += patch_weight

    # Normalize blended probabilities
    prob_map /= np.expand_dims(weight_map, axis=0)
    pred_mask = np.argmax(prob_map, axis=0).astype(np.uint8)
    
    return pred_mask, prob_map

def colorize_mask(mask: np.ndarray):
    """Maps class indices (0: Soil, 1: Crop, 2: Weed) to RGB colors."""
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color_mask[mask == 1] = [0, 255, 0] # Green for crop
    color_mask[mask == 2] = [255, 0, 0] # Red for weed
    return color_mask

def create_prediction_overlay(image_np: np.ndarray, pred_mask: np.ndarray, alpha=0.5):
    """Blends color segmentation mask onto original RGB image."""
    color_mask = colorize_mask(pred_mask)
    overlay = image_np.copy()
    has_veg = pred_mask > 0
    overlay[has_veg] = (overlay[has_veg] * (1 - alpha) + color_mask[has_veg] * alpha).astype(np.uint8)
    return overlay, color_mask

