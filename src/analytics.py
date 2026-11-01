"""
Step 8: Agricultural Analytics & Decision Support
Implements:
- Feature 3: Crop and Weed Coverage Estimation (Pixel & Physical m^2).
- Feature 4: Spatial Infestation Heatmap Generation & Risk Categorization.
- Feature 11: Agricultural Decision Support & Field Scouting Summary.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

def compute_field_coverage(pred_mask: np.ndarray, valid_mask: np.ndarray = None, gsd_cm: float = 1.0):
    """
    Computes pixel and physical land-area coverage statistics.
    Args:
        pred_mask (np.ndarray): 2D integer array (0: Soil, 1: Crop, 2: Weed).
        valid_mask (np.ndarray, optional): 2D boolean or binary array (255: Valid, 0: Invalid padding).
        gsd_cm (float): Ground Sampling Distance in cm/pixel (default 1.0 cm/px for WeedMap at 10m altitude).
    Returns:
        dict: Detailed coverage metrics and physical area estimations.
    """
    if valid_mask is not None:
        valid_pixels_mask = (valid_mask > 0)
    else:
        valid_pixels_mask = np.ones_like(pred_mask, dtype=bool)

    total_image_pixels = pred_mask.size
    valid_field_pixels = int(np.sum(valid_pixels_mask))
    
    if valid_field_pixels == 0:
        valid_field_pixels = total_image_pixels
        valid_pixels_mask = np.ones_like(pred_mask, dtype=bool)

    # Pixel counts within valid field area
    soil_pixels = int(np.sum((pred_mask == 0) & valid_pixels_mask))
    crop_pixels = int(np.sum((pred_mask == 1) & valid_pixels_mask))
    weed_pixels = int(np.sum((pred_mask == 2) & valid_pixels_mask))
    total_vegetation_pixels = crop_pixels + weed_pixels

    # Coverage percentages
    weed_coverage_pct = (weed_pixels / valid_field_pixels) * 100.0
    crop_coverage_pct = (crop_pixels / valid_field_pixels) * 100.0
    vegetation_coverage_pct = (total_vegetation_pixels / valid_field_pixels) * 100.0
    
    # Weed-to-Crop ratio
    weed_to_crop_ratio = (weed_pixels / (crop_pixels + 1e-6))

    # Physical Area Estimation in square meters:
    # 1 pixel = (gsd_cm / 100)^2 m^2
    pixel_area_m2 = (gsd_cm / 100.0) ** 2
    valid_area_m2 = valid_field_pixels * pixel_area_m2
    crop_area_m2 = crop_pixels * pixel_area_m2
    weed_area_m2 = weed_pixels * pixel_area_m2

    return {
        "total_image_pixels": total_image_pixels,
        "valid_field_pixels": valid_field_pixels,
        "soil_pixels": soil_pixels,
        "crop_pixels": crop_pixels,
        "weed_pixels": weed_pixels,
        "weed_coverage_pct": round(weed_coverage_pct, 2),
        "crop_coverage_pct": round(crop_coverage_pct, 2),
        "vegetation_coverage_pct": round(vegetation_coverage_pct, 2),
        "weed_to_crop_ratio": round(weed_to_crop_ratio, 3),
        "gsd_cm": gsd_cm,
        "physical_valid_area_m2": round(valid_area_m2, 4),
        "physical_crop_area_m2": round(crop_area_m2, 4),
        "physical_weed_area_m2": round(weed_area_m2, 4)
    }

def generate_infestation_heatmap(
    pred_mask: np.ndarray,
    grid_size: int = 16,
    thresholds: dict = None,
    output_path: str = None
):
    """
    Divides image into spatial grid cells, computes localized weed density,
    and categorizes infestation severity into Low, Moderate, High risk.
    """
    thresholds = thresholds or {"low": 5.0, "moderate": 15.0} # Percentages
    
    h, w = pred_mask.shape
    grid_h = h // grid_size
    grid_w = w // grid_size
    
    density_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    risk_map = np.zeros((grid_size, grid_size), dtype=np.int32) # 0: Low, 1: Moderate, 2: High
    
    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * grid_h, (i + 1) * grid_h
            x1, x2 = j * grid_w, (j + 1) * grid_w
            
            cell = pred_mask[y1:y2, x1:x2]
            cell_weed_px = np.sum(cell == 2)
            cell_total_px = cell.size
            
            density = (cell_weed_px / cell_total_px) * 100.0
            density_map[i, j] = density
            
            if density < thresholds["low"]:
                risk_map[i, j] = 0 # Low
            elif density < thresholds["moderate"]:
                risk_map[i, j] = 1 # Moderate
            else:
                risk_map[i, j] = 2 # High

    # Upsample density map smoothly to original image size
    smooth_heatmap = cv2.resize(density_map, (w, h), interpolation=cv2.INTER_CUBIC)
    smooth_heatmap = np.clip(smooth_heatmap, 0, 100)

    # Risk categorization summary
    high_cells = int(np.sum(risk_map == 2))
    moderate_cells = int(np.sum(risk_map == 1))
    low_cells = int(np.sum(risk_map == 0))
    total_cells = grid_size * grid_size

    heatmap_summary = {
        "grid_resolution": f"{grid_size}x{grid_size} ({total_cells} zones)",
        "max_local_density_pct": round(float(np.max(density_map)), 2),
        "mean_local_density_pct": round(float(np.mean(density_map)), 2),
        "high_infestation_zones": high_cells,
        "high_infestation_pct": round(high_cells / total_cells * 100.0, 1),
        "moderate_infestation_zones": moderate_cells,
        "moderate_infestation_pct": round(moderate_cells / total_cells * 100.0, 1),
        "low_infestation_zones": low_cells,
        "low_infestation_pct": round(low_cells / total_cells * 100.0, 1),
        "thresholds_used": thresholds
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Continuous density heatmap
        im1 = axes[0].imshow(smooth_heatmap, cmap="jet", vmin=0, vmax=max(25, float(np.max(density_map))))
        axes[0].set_title("Local Weed Density Heatmap (%)", fontsize=12, fontweight="bold")
        axes[0].axis("off")
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Discrete Risk Classification Map
        try:
            import matplotlib as mpl
            cmap_risk = mpl.colormaps["YlOrRd"].resampled(3)
        except Exception:
            cmap_risk = plt.cm.get_cmap("YlOrRd", 3)
        im2 = axes[1].imshow(risk_map, cmap=cmap_risk, vmin=0, vmax=2)
        axes[1].set_title(f"Field Risk Zoning (High: {high_cells} zones)", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        cbar = plt.colorbar(im2, ax=axes[1], ticks=[0, 1, 2], fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(['Low (<5%)', 'Moderate (5-15%)', 'High (>15%)'])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    return smooth_heatmap, density_map, risk_map, heatmap_summary

def generate_decision_support_report(coverage: dict, heatmap: dict):
    """
    Generates a structured agronomic decision-support assessment.
    """
    weed_pct = coverage["weed_coverage_pct"]
    high_zones = heatmap["high_infestation_zones"]
    
    if weed_pct < 3.0 and high_zones == 0:
        action = "Clean / Low Risk: No immediate chemical herbicide intervention required. Routine visual scouting advised."
        priority = "LOW"
    elif weed_pct < 10.0 and high_zones < 5:
        action = f"Moderate Localized Infestation: Targeted spot-spraying recommended in {high_zones} identified hot zones to avoid wasteful broad-acre spraying."
        priority = "MODERATE"
    else:
        action = f"Critical / High Infestation: Weed density exceeds economic injury threshold across {high_zones} field zones. Immediate mechanical or targeted herbicide treatment required."
        priority = "HIGH"

    return {
        "assessment_priority": priority,
        "recommended_action": action,
        "weed_coverage": f"{weed_pct}% ({coverage['physical_weed_area_m2']} m²)",
        "crop_coverage": f"{coverage['crop_coverage_pct']}% ({coverage['physical_crop_area_m2']} m²)",
        "high_risk_zones": high_zones,
        "scouting_priority_locations": "Highlighted in discrete risk zoning map."
    }

