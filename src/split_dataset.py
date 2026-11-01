"""
Step 4.1: Dataset Splitting & Class Distribution Analysis
Splits the WeedMap dataset by field/flight to prevent data leakage.
Filters out empty border tiles and calculates class imbalance statistics.
"""

import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import yaml

def create_splits(
    data_root="C:/Users/Admin/Downloads/Tiles/Tiles",
    output_dir="data/splits",
    config_path="config.yaml",
    min_valid_pixel_ratio=0.05
):
    print("=" * 65)
    print("STEP 4.1: CREATING LEAKAGE-FREE DATASET SPLITS")
    print("=" * 65)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define official field/flight assignments
    # 000, 002, 004 -> Train
    # 001           -> Val
    # 003           -> Test (Official held-out test flight)
    split_mapping = {
        "000": "train",
        "002": "train",
        "004": "train",
        "001": "val",
        "003": "test"
    }
    
    rededge_dir = os.path.join(data_root, "RedEdge")
    records = []
    
    class_stats = {
        "train": {"soil": 0, "crop": 0, "weed": 0, "valid_px": 0, "total_px": 0, "tiles": 0},
        "val":   {"soil": 0, "crop": 0, "weed": 0, "valid_px": 0, "total_px": 0, "tiles": 0},
        "test":  {"soil": 0, "crop": 0, "weed": 0, "valid_px": 0, "total_px": 0, "tiles": 0}
    }
    
    print("\nScanning RedEdge subsets and filtering boundary padding tiles...")
    
    for ds_id, split_name in split_mapping.items():
        ds_path = os.path.join(rededge_dir, ds_id)
        if not os.path.exists(ds_path):
            print(f"Warning: Subset {ds_id} not found at {ds_path}")
            continue
            
        rgb_dir = os.path.join(ds_path, "tile/RGB")
        gt_dir = os.path.join(ds_path, "groundtruth")
        mask_dir = os.path.join(ds_path, "mask")
        
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        
        subset_tiles_total = len(rgb_files)
        subset_effective_tiles = 0
        
        for rgb_path in rgb_files:
            frame_name = os.path.basename(rgb_path) # e.g. frame0008.png
            frame_id = frame_name.replace(".png", "") # e.g. frame0008
            
            imap_path = os.path.join(gt_dir, f"{ds_id}_{frame_id}_GroundTruth_iMap.png")
            color_path = os.path.join(gt_dir, f"{ds_id}_{frame_id}_GroundTruth_color.png")
            mask_path = os.path.join(mask_dir, f"{frame_id}.png")
            
            if not os.path.exists(imap_path) or not os.path.exists(mask_path):
                continue
                
            # Check validity ratio using the flight boundary mask
            valid_mask = np.array(Image.open(mask_path))
            valid_pixels = np.sum(valid_mask > 0)
            total_pixels = valid_mask.size
            valid_ratio = valid_pixels / total_pixels
            
            # Filter out black border tiles outside field boundary
            if valid_ratio < min_valid_pixel_ratio:
                continue
                
            # Read label to compute class distribution
            imap_arr = np.array(Image.open(imap_path))
            clean_mask = np.where(imap_arr == 10000, 1, imap_arr)
            clean_mask = np.where(clean_mask > 2, 0, clean_mask)
            
            soil_px = int(np.sum(clean_mask == 0))
            crop_px = int(np.sum(clean_mask == 1))
            weed_px = int(np.sum(clean_mask == 2))
            
            records.append({
                "dataset_id": ds_id,
                "frame_id": frame_id,
                "split": split_name,
                "rgb_path": rgb_path.replace("\\", "/"),
                "imap_path": imap_path.replace("\\", "/"),
                "color_path": color_path.replace("\\", "/"),
                "mask_path": mask_path.replace("\\", "/"),
                "valid_ratio": round(valid_ratio, 4),
                "soil_px": soil_px,
                "crop_px": crop_px,
                "weed_px": weed_px
            })
            
            # Aggregate stats
            class_stats[split_name]["soil"] += soil_px
            class_stats[split_name]["crop"] += crop_px
            class_stats[split_name]["weed"] += weed_px
            class_stats[split_name]["valid_px"] += int(valid_pixels)
            class_stats[split_name]["total_px"] += total_pixels
            class_stats[split_name]["tiles"] += 1
            subset_effective_tiles += 1
            
        print(f"  Subset {ds_id} ({split_name}): {subset_effective_tiles}/{subset_tiles_total} effective field tiles retained.")

    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Save manifest CSVs
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        out_csv = os.path.join(output_dir, f"{split_name}.csv")
        split_df.to_csv(out_csv, index=False)
        print(f"\nSaved {split_name.upper()} split manifest: {out_csv} ({len(split_df)} tiles)")
        
    all_csv = os.path.join(output_dir, "all_effective_tiles.csv")
    df.to_csv(all_csv, index=False)
    print(f"Saved ALL effective tiles manifest: {all_csv} ({len(df)} total tiles)")
    
    # Print Class Distribution & Imbalance
    print("\n" + "=" * 65)
    print("CLASS DISTRIBUTION & IMBALANCE ANALYSIS")
    print("=" * 65)
    
    total_soil = sum(class_stats[s]["soil"] for s in class_stats)
    total_crop = sum(class_stats[s]["crop"] for s in class_stats)
    total_weed = sum(class_stats[s]["weed"] for s in class_stats)
    total_all = total_soil + total_crop + total_weed
    
    print(f"Total Effective Labeled Pixels: {total_all:,}")
    print(f"  Class 0 (Soil/Background): {total_soil:,} pixels ({total_soil/total_all*100:.2f}%)")
    print(f"  Class 1 (Crop/Sugar Beet): {total_crop:,} pixels ({total_crop/total_all*100:.2f}%)")
    print(f"  Class 2 (Weed):            {total_weed:,} pixels ({total_weed/total_all*100:.2f}%)")
    
    # Compute inverse class frequencies for loss weighting
    freqs = np.array([total_soil, total_crop, total_weed], dtype=np.float64) / total_all
    median_freq = np.median(freqs)
    class_weights = median_freq / freqs
    norm_weights = class_weights / np.sum(class_weights) * 3.0 # normalized around 1.0
    
    print("\nRecommended Class Weights for Cross-Entropy / Focal Loss:")
    print(f"  Raw Inverse Median Weights: {class_weights.round(4).tolist()}")
    print(f"  Normalized Weights (Sum=3):  {norm_weights.round(4).tolist()}")
    
    # Save class weight metadata
    weights_path = os.path.join(output_dir, "class_weights.yaml")
    with open(weights_path, "w") as f:
        yaml.dump({
            "class_counts": {
                "soil": int(total_soil),
                "crop": int(total_crop),
                "weed": int(total_weed)
            },
            "class_percentages": {
                "soil": float(round(total_soil/total_all*100, 2)),
                "crop": float(round(total_crop/total_all*100, 2)),
                "weed": float(round(total_weed/total_all*100, 2))
            },
            "normalized_weights": [float(w) for w in norm_weights.round(4)]
        }, f)
    print(f"Saved class weights to {weights_path}")
    print("=" * 65)
    
    return df

if __name__ == "__main__":
    create_splits()

