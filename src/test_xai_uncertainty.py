"""
Verification script for Feature 5 (Grad-CAM XAI) and Feature 6 (Uncertainty Estimation).
Generates report figures in outputs/reports/.
"""

import os
import sys
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torchvision.transforms.functional as TF

from src.predict import load_inference_model
from src.explainability import generate_xai_visualization
from src.uncertainty import estimate_uncertainty_mc_dropout, generate_uncertainty_visualization

def run_verification():
    print("=" * 70)
    print("TESTING EXPLAINABLE AI & UNCERTAINTY ESTIMATION MODULES")
    print("=" * 70)
    
    img_path = "C:/Users/Admin/Downloads/Tiles/Tiles/RedEdge/003/tile/RGB/frame0008.png"
    ckpt_path = os.path.join(PROJECT_ROOT, "outputs/checkpoints/unet_best.pth")
    
    image_pil = Image.open(img_path).convert("RGB").resize((256, 256))
    img_display = np.array(image_pil)
    img_tensor = TF.normalize(TF.to_tensor(image_pil), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0)
    
    model, device = load_inference_model(ckpt_path)
    img_tensor = img_tensor.to(device)
    
    # 1. Grad-CAM
    xai_out = os.path.join(PROJECT_ROOT, "outputs/reports/gradcam_sample_explanation.png")
    crop_cam, weed_cam, crop_ov, weed_ov = generate_xai_visualization(model, img_tensor, img_display, output_path=xai_out)
    print(f"[PASS] Saved Grad-CAM XAI explanation to: {xai_out}")
    
    # 2. Monte Carlo Dropout Uncertainty
    unc_out = os.path.join(PROJECT_ROOT, "outputs/reports/mc_dropout_uncertainty_sample.png")
    mean_p, entropy, variance, pred_m = estimate_uncertainty_mc_dropout(model, img_tensor, num_passes=8)
    unc_metrics = generate_uncertainty_visualization(img_display, pred_m, entropy, output_path=unc_out)
    print(f"[PASS] Saved Uncertainty Visualization to: {unc_out}")
    print(f"  Mean Predictive Entropy: {unc_metrics['mean_entropy']}")
    print(f"  Uncertain Boundary Pixels: {unc_metrics['high_uncertainty_pixel_pct']}%")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()

