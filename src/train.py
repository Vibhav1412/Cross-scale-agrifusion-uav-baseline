"""
Step 6.3: Model Training Engine (Robust getattr for all args)
Trains U-Net, DeepLabV3+, and Basic CNN on WeedMap UAV imagery.
Supports Apple Silicon MPS, CUDA, and CPU.
Logs metrics, saves checkpoints, and plots training curves.
"""

import os
import sys
import time
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.unet import build_unet
from models.deeplab import build_deeplabv3plus
from models.baseline_cnn import build_baseline_cnn
from src.dataset import get_dataloaders
from src.loss import CombinedLoss
from src.metrics import SegmentationMetrics

def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_class_weights(weights_file="data/splits/class_weights.yaml"):
    full_path = os.path.join(PROJECT_ROOT, weights_file)
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            data = yaml.safe_load(f)
            return torch.tensor(data.get("normalized_weights", [0.0087, 1.9017, 1.0896]), dtype=torch.float32)
    return torch.tensor([0.0087, 1.9017, 1.0896], dtype=torch.float32)

def train_model(args):
    # Safely extract all arguments with defaults
    model_name = getattr(args, "model", "unet")
    epochs = getattr(args, "epochs", 10)
    batch_size = getattr(args, "batch_size", 8)
    img_size = getattr(args, "img_size", 256)
    lr = getattr(args, "lr", 0.001)
    weight_decay = getattr(args, "weight_decay", 1e-4)
    ce_weight = getattr(args, "ce_weight", 0.5)
    dice_weight = getattr(args, "dice_weight", 0.5)
    num_classes = getattr(args, "num_classes", 3)
    pretrained = getattr(args, "pretrained", True)
    seed = getattr(args, "seed", 42)
    resume = getattr(args, "resume", None)

    print("=" * 70, flush=True)
    print(f"STEP 6: TRAINING {model_name.upper()} BASELINE MODEL ({epochs} Epochs)", flush=True)
    print("=" * 70, flush=True)
    
    device = get_device()
    print(f"Training Device: {device}", flush=True)
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        splits_dir=os.path.join(PROJECT_ROOT, "data/splits"),
        batch_size=batch_size,
        image_size=(img_size, img_size),
        num_workers=0
    )
    print(f"Loaded {len(train_loader.dataset)} training tiles ({len(train_loader)} batches)", flush=True)
    print(f"Loaded {len(val_loader.dataset)} validation tiles ({len(val_loader)} batches)", flush=True)
    
    # 2. Model Selection
    if model_name == "unet":
        model = build_unet(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "deeplab":
        model = build_deeplabv3plus(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "baseline_cnn":
        model = build_baseline_cnn(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    model = model.to(device)
    
    # 3. Loss & Optimizer
    class_weights = load_class_weights().to(device)
    print(f"Class Weights Applied to Loss: {class_weights.tolist()}", flush=True)
    
    criterion = CombinedLoss(
        num_classes=num_classes,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        class_weights=class_weights
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Metrics
    metrics_calc = SegmentationMetrics(num_classes=num_classes, class_names=["soil", "crop", "weed"])
    
    # Output directories
    ckpt_dir = os.path.join(PROJECT_ROOT, "outputs/checkpoints")
    reports_dir = os.path.join(PROJECT_ROOT, "outputs/reports")
    graphs_dir = os.path.join(PROJECT_ROOT, "outputs/graphs")
    for d in [ckpt_dir, reports_dir, graphs_dir]:
        os.makedirs(d, exist_ok=True)
        
    best_val_miou = 0.0
    start_epoch = 1
    history = []
    
    # Check if resuming
    if resume:
        resume_path = resume if os.path.isfile(resume) else os.path.join(ckpt_dir, f"{model_name}_best.pth")
        if os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}", flush=True)
            ckpt = torch.load(resume_path, map_location=device)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt.get("epoch", 0) + 1
                best_val_miou = ckpt.get("val_miou", 0.0)
                print(f"Loaded epoch {start_epoch-1} with best Val mIoU: {best_val_miou*100:.2f}%", flush=True)
            else:
                model.load_state_dict(ckpt)
                
    start_train_time = time.time()
    
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_ce = 0.0
        train_dice = 0.0
        
        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss, ce_val, dice_val = criterion(logits, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_ce += ce_val
            train_dice += dice_val
            
        train_loss /= len(train_loader)
        train_ce /= len(train_loader)
        train_dice /= len(train_loader)
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_ce = 0.0
        val_dice = 0.0
        metrics_calc.reset()
        
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                logits = model(images)
                loss, ce_val, dice_val = criterion(logits, masks)
                
                val_loss += loss.item()
                val_ce += ce_val
                val_dice += dice_val
                
                metrics_calc.update(logits, masks)
                
        val_loss /= len(val_loader)
        val_ce /= len(val_loader)
        val_dice /= len(val_loader)
        
        val_metrics = metrics_calc.compute()
        val_miou = val_metrics["miou"]
        crop_iou = val_metrics["iou_per_class"]["crop"]
        weed_iou = val_metrics["iou_per_class"]["weed"]
        soil_iou = val_metrics["iou_per_class"]["soil"]
        val_mdice = val_metrics["mdice"]
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        epoch_duration = time.time() - epoch_start
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val mIoU: {val_miou*100:.2f}% (Soil: {soil_iou*100:.1f}%, Crop: {crop_iou*100:.1f}%, Weed: {weed_iou*100:.1f}%) | "
              f"Val mDice: {val_mdice*100:.2f}% | LR: {current_lr:.6f}", flush=True)
              
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_ce": round(train_ce, 4),
            "train_dice": round(train_dice, 4),
            "val_loss": round(val_loss, 4),
            "val_ce": round(val_ce, 4),
            "val_dice": round(val_dice, 4),
            "val_miou": round(val_miou, 4),
            "crop_iou": round(crop_iou, 4),
            "weed_iou": round(weed_iou, 4),
            "soil_iou": round(soil_iou, 4),
            "val_mdice": round(val_mdice, 4),
            "lr": current_lr,
            "epoch_duration_s": round(epoch_duration, 2)
        })
        
        # Checkpoint Best Model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_path = os.path.join(ckpt_dir, f"{model_name}_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": val_miou,
                "crop_iou": crop_iou,
                "weed_iou": weed_iou,
                "soil_iou": soil_iou,
                "val_mdice": val_mdice,
                "config": vars(args)
            }, best_path)
            print(f"  [CHECKPOINT] New best model saved to {best_path} (mIoU: {val_miou*100:.2f}%)", flush=True)
            
    total_train_time = time.time() - start_train_time
    print("\n" + "=" * 70, flush=True)
    print(f"TRAINING COMPLETE in {total_train_time/60:.2f} minutes.", flush=True)
    print(f"Best Validation mIoU: {best_val_miou*100:.2f}%", flush=True)
    print("=" * 70, flush=True)
    
    # Save latest model checkpoint
    latest_path = os.path.join(ckpt_dir, f"{model_name}_latest.pth")
    torch.save(model.state_dict(), latest_path)
    
    # Save Training History CSV
    history_df = pd.DataFrame(history)
    history_csv = os.path.join(reports_dir, f"{model_name}_training_history.csv")
    history_df.to_csv(history_csv, index=False)
    print(f"Saved training history log: {history_csv}", flush=True)
    
    # Plot Training Graphs
    plot_training_curves(history_df, model_name, graphs_dir)
    return history_df

def plot_training_curves(df, model_name, graphs_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", color="royalblue", linewidth=2)
    axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", color="crimson", linewidth=2, linestyle="--")
    axes[0].set_title(f"{model_name.upper()} Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Combined Loss (CE + Dice)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # IoU plot
    axes[1].plot(df["epoch"], df["val_miou"] * 100, label="Mean IoU", color="purple", linewidth=2)
    axes[1].plot(df["epoch"], df["crop_iou"] * 100, label="Crop IoU", color="forestgreen", linewidth=1.5, linestyle=":")
    axes[1].plot(df["epoch"], df["weed_iou"] * 100, label="Weed IoU", color="darkorange", linewidth=1.5, linestyle="-.")
    axes[1].set_title(f"{model_name.upper()} Validation IoU Progression (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    graph_path = os.path.join(graphs_dir, f"{model_name}_training_curves.png")
    plt.savefig(graph_path, dpi=150)
    plt.close()
    print(f"Saved training curves visualization: {graph_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UAV Crop-Weed Segmentation Baseline")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "deeplab", "baseline_cnn"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ce_weight", type=float, default=0.5)
    parser.add_argument("--dice_weight", type=float, default=0.5)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    train_model(args)
