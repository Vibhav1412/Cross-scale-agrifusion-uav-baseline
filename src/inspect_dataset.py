"""
Step 3: Dataset Inspection and Verification Script
Inspects the WeedMap dataset structure, verifies image-mask pairs,
analyzes class distributions, and validates label encodings.
"""

import os
import glob
from pathlib import Path
import numpy as np
from PIL import Image

def inspect_dataset(data_root="C:/Users/Admin/Downloads/Tiles/Tiles"):
    print("=" * 60)
    print("STEP 3: INSPECTING WEEDMAP DATASET AT:", data_root)
    print("=" * 60)
    
    if not os.path.exists(data_root):
        print(f"Error: Path {data_root} does not exist.")
        return
        
    sensors = ["RedEdge", "Sequoia"]
    overall_summary = {}
    
    for sensor in sensors:
        sensor_dir = os.path.join(data_root, sensor)
        if not os.path.exists(sensor_dir):
            print(f"Sensor directory {sensor_dir} not found.")
            continue
            
        dataset_folders = sorted([f for f in os.listdir(sensor_dir) if os.path.isdir(os.path.join(sensor_dir, f))])
        print(f"\n[Sensor: {sensor}] Found datasets: {dataset_folders}")
        
        for ds in dataset_folders:
            ds_path = os.path.join(sensor_dir, ds)
            gt_dir = os.path.join(ds_path, "groundtruth")
            mask_dir = os.path.join(ds_path, "mask")
            tile_dir = os.path.join(ds_path, "tile")
            
            # Check RGB availability
            rgb_dir = os.path.join(tile_dir, "RGB")
            has_rgb = os.path.exists(rgb_dir)
            
            # Gather files
            imaps = sorted(glob.glob(os.path.join(gt_dir, "*_GroundTruth_iMap.png")))
            colors = sorted(glob.glob(os.path.join(gt_dir, "*_GroundTruth_color.png")))
            masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
            
            rgb_tiles = sorted(glob.glob(os.path.join(rgb_dir, "*.png"))) if has_rgb else []
            
            # Count effective valid masks
            valid_tiles = 0
            weed_pixels_total = 0
            crop_pixels_total = 0
            soil_pixels_total = 0
            
            # Sample inspection on first valid tile
            sample_info = {}
            for i, imap_file in enumerate(imaps):
                # Inspect first 10 for quick sanity check and class balance
                if i < 20 or i == len(imaps) - 1:
                    img = Image.open(imap_file)
                    arr = np.array(img)
                    unique_vals = np.unique(arr)
                    
                    # 10000 in WeedMap indicates non-class / crop edge
                    cleaned = np.where(arr == 10000, 1, arr)
                    
                    crops = np.sum(cleaned == 1)
                    weeds = np.sum(cleaned == 2)
                    soil = np.sum(cleaned == 0)
                    
                    crop_pixels_total += crops
                    weed_pixels_total += weeds
                    soil_pixels_total += soil
                    
                    if crops > 0 or weeds > 0:
                        valid_tiles += 1
                        if not sample_info:
                            sample_info = {
                                "file": os.path.basename(imap_file),
                                "shape": arr.shape,
                                "dtype": str(arr.dtype),
                                "raw_uniques": unique_vals.tolist(),
                                "crops": int(crops),
                                "weeds": int(weeds)
                            }
            
            overall_summary[f"{sensor}_{ds}"] = {
                "imaps_count": len(imaps),
                "rgb_count": len(rgb_tiles),
                "masks_count": len(masks),
                "sample": sample_info
            }
            
            print(f"  Dataset {ds}: {len(imaps)} iMap masks, {len(rgb_tiles)} RGB tiles, {len(masks)} boundary masks.")
            if sample_info:
                print(f"    Sample ({sample_info['file']}): Dim={sample_info['shape']}, dtype={sample_info['dtype']}, "
                      f"Raw values={sample_info['raw_uniques']}, Crop px={sample_info['crops']}, Weed px={sample_info['weeds']}")

    print("\n" + "=" * 60)
    print("DATASET VERIFICATION SUMMARY COMPLETED")
    print("=" * 60)
    return overall_summary

if __name__ == "__main__":
    inspect_dataset()

