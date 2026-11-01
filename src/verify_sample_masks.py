"""
Script to verify sample image-mask pairs and visualize class encoding.
Saves a visual verification figure to outputs/predictions/sample_verification.png.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def verify_sample():
    ds_path = "C:/Users/Admin/Downloads/Tiles/Tiles/RedEdge/000"
    rgb_path = os.path.join(ds_path, "tile/RGB/frame0008.png")
    imap_path = os.path.join(ds_path, "groundtruth/000_frame0008_GroundTruth_iMap.png")
    color_path = os.path.join(ds_path, "groundtruth/000_frame0008_GroundTruth_color.png")
    mask_path = os.path.join(ds_path, "mask/frame0008.png")
    
    # 1. Load RGB
    rgb = np.array(Image.open(rgb_path))
    
    # 2. Load iMap (16-bit grayscale)
    imap = np.array(Image.open(imap_path))
    
    # 3. Load color groundtruth (3-channel PNG)
    gt_color = np.array(Image.open(color_path))
    
    # 4. Load valid area mask
    valid_mask = np.array(Image.open(mask_path))
    
    # Clean iMap: map 10000 -> 1 (Crop)
    clean_mask = np.where(imap == 10000, 1, imap)
    clean_mask = np.where(clean_mask > 2, 0, clean_mask).astype(np.uint8)
    
    # Print statistics
    total_px = imap.size
    soil_px = np.sum(clean_mask == 0)
    crop_px = np.sum(clean_mask == 1)
    weed_px = np.sum(clean_mask == 2)
    valid_px = np.sum(valid_mask > 0)
    
    print("Sample: frame0008 in RedEdge/000")
    print(f"Total Pixels: {total_px} (Shape: {imap.shape})")
    print(f"Valid Field Pixels: {valid_px} ({valid_px/total_px*100:.1f}%)")
    print(f"Soil/Background Pixels: {soil_px} ({soil_px/total_px*100:.1f}%)")
    print(f"Crop Pixels: {crop_px} ({crop_px/total_px*100:.1f}%)")
    print(f"Weed Pixels: {weed_px} ({weed_px/total_px*100:.1f}%)")
    
    # Create side-by-side visualization
    os.makedirs("C:/Users/Admin/.gemini/antigravity/scratch/uav-crop-monitoring/outputs/reports", exist_ok=True)
    out_fig = "C:/Users/Admin/.gemini/antigravity/scratch/uav-crop-monitoring/outputs/reports/dataset_sample_verification.png"
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("UAV RGB Tile (480x360)")
    axes[0].axis("off")
    
    axes[1].imshow(gt_color)
    axes[1].set_title("GroundTruth Color (Black=Soil, Green=Crop, Red=Weed)")
    axes[1].axis("off")
    
    # Custom colored clean mask
    color_mapped = np.zeros((clean_mask.shape[0], clean_mask.shape[1], 3), dtype=np.uint8)
    color_mapped[clean_mask == 1] = [0, 255, 0] # Green for crop
    color_mapped[clean_mask == 2] = [255, 0, 0] # Red for weed
    axes[2].imshow(color_mapped)
    axes[2].set_title("Parsed 3-Class Label Map")
    axes[2].axis("off")
    
    axes[3].imshow(valid_mask, cmap='gray')
    axes[3].set_title("Boundary / Valid Area Mask")
    axes[3].axis("off")
    
    plt.tight_layout()
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"Saved verification visual to {out_fig}")

if __name__ == "__main__":
    verify_sample()

