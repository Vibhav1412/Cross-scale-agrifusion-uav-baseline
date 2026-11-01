"""
Step 7: Model Evaluation Suite
Evaluates trained segmentation checkpoints on held-out test splits.
Computes mIoU, Dice, Precision, Recall, Confusion Matrix,
and generates side-by-side comparative visualizations (RGB, GT, Pred, Error).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F

from models.unet import build_unet
from models.deeplab import build_deeplabv3plus
from models.baseline_cnn import build_baseline_cnn
from src.dataset import get_dataloaders
from src.metrics import SegmentationMetrics

def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def evaluate_model(args):
    model_name = getattr(args, "model", "unet")
    batch_size = getattr(args, "batch_size", 8)
    img_size = getattr(args, "img_size", 256)
    num_classes = getattr(args, "num_classes", 3)
    checkpoint_arg = getattr(args, "checkpoint", None)

    print("=" * 70)
    print(f"STEP 7: EVALUATING {model_name.upper()} ON HELD-OUT TEST SPLIT")
    print("=" * 70)
    
    device = get_device()
    print(f"Evaluation Device: {device}")
    
    # 1. DataLoader for Test Split (Dataset 003 - Held-out flight)
    _, _, test_loader = get_dataloaders(
        splits_dir=os.path.join(PROJECT_ROOT, "data/splits"),
        batch_size=batch_size,
        image_size=(img_size, img_size),
        num_workers=0
    )
    print(f"Loaded {len(test_loader.dataset)} test tiles ({len(test_loader)} batches)")
    
    # 2. Build Model & Load Checkpoint
    if model_name == "unet":
        model = build_unet(num_classes=num_classes, pretrained=False)
    elif model_name == "deeplab":
        model = build_deeplabv3plus(num_classes=num_classes, pretrained=False)
    elif model_name == "baseline_cnn":
        model = build_baseline_cnn(num_classes=num_classes)
        
    ckpt_path = checkpoint_arg or os.path.join(PROJECT_ROOT, f"outputs/checkpoints/{model_name}_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        
    print(f"Loading checkpoint weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    
    # 3. Compute Test Metrics
    metrics_calc = SegmentationMetrics(num_classes=num_classes, class_names=["soil", "crop", "weed"])
    
    test_samples = []
    
    with torch.no_grad():
        for i, (images, masks, meta) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            metrics_calc.update(preds, masks)
            
            # Save first 5 batches for visualization
            if len(test_samples) < 10:
                for j in range(images.size(0)):
                    if len(test_samples) < 10:
                        test_samples.append({
                            "image": images[j].cpu(),
                            "mask": masks[j].cpu(),
                            "pred": preds[j].cpu(),
                            "meta": {
                                "dataset_id": meta["dataset_id"][j],
                                "frame_id": meta["frame_id"][j]
                            }
                        })
                        
    results = metrics_calc.compute()
    
    print("\n" + "=" * 70)
    print("HELD-OUT TEST SET EVALUATION RESULTS (CROSS-FIELD GENERALIZATION)")
    print("=" * 70)
    print(f"Mean IoU (mIoU):       {results['miou']*100:.2f}%")
    print(f"Mean Dice (F1):        {results['mdice']*100:.2f}%")
    print(f"Macro Precision:       {results['mprecision']*100:.2f}%")
    print(f"Macro Recall:          {results['mrecall']*100:.2f}%")
    print("\nPer-Class Breakdown:")
    for cls_name in ["soil", "crop", "weed"]:
        print(f"  {cls_name.capitalize():<10} | IoU: {results['iou_per_class'][cls_name]*100:.2f}% | "
              f"Dice: {results['dice_per_class'][cls_name]*100:.2f}% | "
              f"Prec: {results['precision_per_class'][cls_name]*100:.2f}% | "
              f"Rec: {results['recall_per_class'][cls_name]*100:.2f}%")
              
    print("\nConfusion Matrix (Rows: Ground Truth, Cols: Prediction):")
    cm = np.array(results["confusion_matrix"])
    print(f"               Pred Soil     Pred Crop     Pred Weed")
    print(f"  Actual Soil  {cm[0,0]:<13} {cm[0,1]:<13} {cm[0,2]:<13}")
    print(f"  Actual Crop  {cm[1,0]:<13} {cm[1,1]:<13} {cm[1,2]:<13}")
    print(f"  Actual Weed  {cm[2,0]:<13} {cm[2,1]:<13} {cm[2,2]:<13}")
    
    # Save Report CSV
    reports_dir = os.path.join(PROJECT_ROOT, "outputs/reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, f"{args.model}_test_evaluation.csv")
    
    summary_data = [
        {"Metric": "mIoU", "Value": round(results["miou"], 4)},
        {"Metric": "Mean Dice (mDice)", "Value": round(results["mdice"], 4)},
        {"Metric": "Macro Precision", "Value": round(results["mprecision"], 4)},
        {"Metric": "Macro Recall", "Value": round(results["mrecall"], 4)},
        {"Metric": "Soil IoU", "Value": round(results["iou_per_class"]["soil"], 4)},
        {"Metric": "Crop IoU", "Value": round(results["iou_per_class"]["crop"], 4)},
        {"Metric": "Weed IoU", "Value": round(results["iou_per_class"]["weed"], 4)},
        {"Metric": "Crop Dice", "Value": round(results["dice_per_class"]["crop"], 4)},
        {"Metric": "Weed Dice", "Value": round(results["dice_per_class"]["weed"], 4)}
    ]
    pd.DataFrame(summary_data).to_csv(report_path, index=False)
    print(f"\nSaved test evaluation metrics to: {report_path}")
    
    # Generate Visualizations
    save_test_visualizations(test_samples, args.model)
    return results

def save_test_visualizations(samples, model_name, num_plots=6):
    preds_dir = os.path.join(PROJECT_ROOT, "outputs/predictions")
    os.makedirs(preds_dir, exist_ok=True)
    
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    
    # Color palette: 0=Black, 1=Green (Crop), 2=Red (Weed)
    palette = {
        0: [0, 0, 0],
        1: [0, 255, 0],
        2: [255, 0, 0]
    }
    
    n = min(len(samples), num_plots)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for i in range(n):
        s = samples[i]
        # Denormalize RGB
        img_np = s["image"].permute(1, 2, 0).numpy()
        img_np = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)
        
        gt_np = s["mask"].numpy()
        pred_np = s["pred"].numpy()
        
        # Color masks
        gt_color = np.zeros((*gt_np.shape, 3), dtype=np.uint8)
        for c, rgb_val in palette.items():
            gt_color[gt_np == c] = rgb_val
            
        pred_color = np.zeros((*pred_np.shape, 3), dtype=np.uint8)
        for c, rgb_val in palette.items():
            pred_color[pred_np == c] = rgb_val
            
        # Error map: Red where incorrect
        error_map = np.zeros((*pred_np.shape, 3), dtype=np.uint8)
        error_map[pred_np != gt_np] = [255, 165, 0] # Orange for error
        error_map[(pred_np == gt_np) & (gt_np > 0)] = [0, 255, 0] # Green for correct vegetation
        
        # 1. Original RGB
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Test UAV RGB [{s['meta']['dataset_id']}/{s['meta']['frame_id']}]")
        axes[i, 0].axis("off")
        
        # 2. Ground Truth Mask
        axes[i, 1].imshow(gt_color)
        axes[i, 1].set_title("Ground Truth (Green:Crop, Red:Weed)")
        axes[i, 1].axis("off")
        
        # 3. Model Prediction
        axes[i, 2].imshow(pred_color)
        axes[i, 2].set_title(f"{model_name.upper()} Prediction")
        axes[i, 2].axis("off")
        
        # 4. Error / Agreement Map
        axes[i, 3].imshow(error_map)
        err_px = np.sum(pred_np != gt_np)
        axes[i, 3].set_title(f"Agreement Map (Orange: Misclassified {err_px}px)")
        axes[i, 3].axis("off")
        
    plt.tight_layout()
    viz_path = os.path.join(preds_dir, f"{model_name}_test_predictions_gallery.png")
    plt.savefig(viz_path, dpi=150)
    plt.close()
    print(f"Saved prediction visual comparison gallery to: {viz_path}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Model on Test Split")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "deeplab", "baseline_cnn"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=3)
    
    args = parser.parse_args()
    evaluate_model(args)

