"""
Step 4.4: DataLoader Verification & Visual Inspection
Tests dataset loading, tensor dimensions, mask label integrity,
throughput, and generates an augmented batch verification figure.
"""

import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders

def test_pipeline():
    print("=" * 65)
    print("STEP 4.4: TESTING PREPROCESSING & DATALOADER PIPELINE")
    print("=" * 65)
    
    batch_size = 8
    image_size = (256, 256)
    splits_dir = os.path.join(PROJECT_ROOT, "data/splits")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        splits_dir=splits_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)} ({len(train_loader.dataset)} tiles)")
    print(f"Val batches:   {len(val_loader)} ({len(val_loader.dataset)} tiles)")
    print(f"Test batches:  {len(test_loader)} ({len(test_loader.dataset)} tiles)")
    
    # Fetch first training batch
    start_time = time.time()
    images, masks, meta = next(iter(train_loader))
    elapsed = time.time() - start_time
    
    print("\n--- Tensor Verification ---")
    print(f"Images Tensor: shape={images.shape}, dtype={images.dtype}, device={images.device}")
    print(f"Masks Tensor:  shape={masks.shape}, dtype={masks.dtype}, device={masks.device}")
    print(f"Batch fetch latency: {elapsed*1000:.2f} ms")
    
    # Assertions for research code integrity
    assert images.shape == (batch_size, 3, 256, 256), f"Unexpected image shape: {images.shape}"
    assert masks.shape == (batch_size, 256, 256), f"Unexpected mask shape: {masks.shape}"
    assert masks.dtype == torch.int64, f"Mask dtype must be int64, got {masks.dtype}"
    
    unique_classes = torch.unique(masks).tolist()
    print(f"Unique classes in batch: {unique_classes}")
    for c in unique_classes:
        assert c in [0, 1, 2], f"Illegal class ID detected in batch: {c}"
        
    print("[PASS] All tensor shape and class label assertions PASSED.")
    
    # Generate Visual Inspection Figure
    # Denormalize ImageNet for visualization
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    
    reports_dir = os.path.join(PROJECT_ROOT, "outputs/reports")
    os.makedirs(reports_dir, exist_ok=True)
    out_fig = os.path.join(reports_dir, "augmented_batch_sample.png")
    
    # Show first 4 samples of the batch
    num_samples = min(4, batch_size)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        # Denormalize image
        img_np = images[i].permute(1, 2, 0).cpu().numpy()
        img_np = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)
        
        mask_np = masks[i].cpu().numpy()
        
        # Color mask: 0=Black, 1=Green (Crop), 2=Red (Weed)
        color_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
        color_mask[mask_np == 1] = [0, 255, 0]
        color_mask[mask_np == 2] = [255, 0, 0]
        
        # Overlay
        overlay = img_np.copy()
        mask_any = mask_np > 0
        overlay[mask_any] = (overlay[mask_any] * 0.4 + color_mask[mask_any] * 0.6).astype(np.uint8)
        
        # Plot
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Augmented RGB [{meta['dataset_id'][i]}/{meta['frame_id'][i]}]")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(color_mask)
        crop_cnt = np.sum(mask_np == 1)
        weed_cnt = np.sum(mask_np == 2)
        axes[i, 1].set_title(f"Target Mask (Crop:{crop_cnt}px, Weed:{weed_cnt}px)")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay (Green=Crop, Red=Weed)")
        axes[i, 2].axis("off")
        
    plt.tight_layout()
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"\n[PASS] Saved batch visual verification figure to: {out_fig}")
    print("=" * 65)

if __name__ == "__main__":
    test_pipeline()
