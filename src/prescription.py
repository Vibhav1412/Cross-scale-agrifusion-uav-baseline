"""
Step 8.2: Precision Agronomic Prescription & Variable Rate Application (VRA) Engine
Implements:
- 4-Tier Weed Infestation Prescription Zoning (0%, 30%, 70%, 100% dosage).
- Nozzle Pulse Width Modulation (PWM) and automatic nozzle shutoff calculation.
- Rigorous Agricultural Economics & Chemical Reduction Calculator (INR & USD).
- Environmental Soil & Groundwater Protection Metrics (Active Ingredient Runoff Spared).
- Commercial Autonomous Drone Mission Exporters (DJI Agras / XAG CSV & GeoJSON).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Standard dosage tiers based on European & ICAR Precision Agriculture Protocols
DOSAGE_TIERS = {
    0: {
        "name": "Zero-Spray / Organic Buffer",
        "min_density": 0.0,
        "max_density": 0.01,
        "multiplier": 0.0,
        "nozzle_state": "OFF",
        "pwm_pct": 0,
        "color_rgb": (34, 197, 94),     # Green #22c55e
        "color_hex": "#22c55e",
        "description": "Clean soil/crop canopy. Nozzle completely shut off to preserve beneficial organisms."
    },
    1: {
        "name": "Micro-Pulse Spot Spray",
        "min_density": 0.01,
        "max_density": 0.05,
        "multiplier": 0.30,
        "nozzle_state": "PWM_PULSE",
        "pwm_pct": 30,
        "color_rgb": (234, 179, 8),     # Yellow #eab308
        "color_hex": "#eab308",
        "description": "Incipient weed emergence. Micro-pulse localized spraying to halt seed dispersion."
    },
    2: {
        "name": "Targeted Standard Spray",
        "min_density": 0.05,
        "max_density": 0.15,
        "multiplier": 0.70,
        "nozzle_state": "PWM_MODERATE",
        "pwm_pct": 70,
        "color_rgb": (249, 115, 22),    # Orange #f97316
        "color_hex": "#f97316",
        "description": "Moderate weed colony. Targeted application with droplet size calibration."
    },
    3: {
        "name": "Intensive Curative Spray",
        "min_density": 0.15,
        "max_density": 1.0,
        "multiplier": 1.0,
        "nozzle_state": "FULL_ON",
        "pwm_pct": 100,
        "color_rgb": (239, 68, 68),     # Red #ef4444
        "color_hex": "#ef4444",
        "description": "Critical weed density. Full canopy penetration required to eliminate yield loss."
    }
}

def generate_prescription_grid(pred_mask: np.ndarray, grid_size: int = 16, gsd_cm: float = 1.0, base_rate_l_ha: float = 200.0):
    """
    Partitions the semantic prediction mask into an N x N spatial prescription grid.
    Computes weed density and assigns Variable Rate Application (VRA) tiers to each cell.
    """
    h, w = pred_mask.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    grid_cells = []
    tier_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    pixel_area_m2 = (gsd_cm / 100.0) ** 2
    cell_area_m2 = (cell_h * cell_w) * pixel_area_m2
    
    density_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    dosage_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    tier_map = np.zeros((grid_size, grid_size), dtype=np.int32)

    for r in range(grid_size):
        for c in range(grid_size):
            r_start = r * cell_h
            r_end = h if r == grid_size - 1 else (r + 1) * cell_h
            c_start = c * cell_w
            c_end = w if c == grid_size - 1 else (c + 1) * cell_w

            cell_patch = pred_mask[r_start:r_end, c_start:c_end]
            cell_pixels = cell_patch.size

            soil_count = int(np.sum(cell_patch == 0))
            crop_count = int(np.sum(cell_patch == 1))
            weed_count = int(np.sum(cell_patch == 2))

            weed_density = (weed_count / cell_pixels) if cell_pixels > 0 else 0.0
            density_map[r, c] = weed_density

            # Assign Tier
            if weed_density < 0.01:
                tier = 0
            elif weed_density < 0.05:
                tier = 1
            elif weed_density < 0.15:
                tier = 2
            else:
                tier = 3

            tier_counts[tier] += 1
            tier_map[r, c] = tier
            
            dosage_l_ha = base_rate_l_ha * DOSAGE_TIERS[tier]["multiplier"]
            dosage_map[r, c] = dosage_l_ha

            grid_cells.append({
                "grid_row": r,
                "grid_col": c,
                "row_start_px": r_start,
                "row_end_px": r_end,
                "col_start_px": c_start,
                "col_end_px": c_end,
                "total_pixels": cell_pixels,
                "weed_pixels": weed_count,
                "crop_pixels": crop_count,
                "soil_pixels": soil_count,
                "weed_density_pct": round(weed_density * 100, 2),
                "tier": tier,
                "tier_name": DOSAGE_TIERS[tier]["name"],
                "dosage_l_ha": round(dosage_l_ha, 2),
                "nozzle_state": DOSAGE_TIERS[tier]["nozzle_state"],
                "pwm_pct": DOSAGE_TIERS[tier]["pwm_pct"],
                "color_hex": DOSAGE_TIERS[tier]["color_hex"],
                "cell_area_m2": round(cell_area_m2, 4)
            })

    total_cells = grid_size * grid_size
    nozzle_shutoff_pct = round((tier_counts[0] / total_cells) * 100, 2)
    mean_density_pct = round(float(np.mean(density_map)) * 100, 2)
    mean_dosage_l_ha = round(float(np.mean(dosage_map)), 2)

    return {
        "grid_size": grid_size,
        "cell_h": cell_h,
        "cell_w": cell_w,
        "total_cells": total_cells,
        "tier_counts": tier_counts,
        "nozzle_shutoff_pct": nozzle_shutoff_pct,
        "mean_density_pct": mean_density_pct,
        "mean_dosage_l_ha": mean_dosage_l_ha,
        "density_map": density_map,
        "dosage_map": dosage_map,
        "tier_map": tier_map,
        "cells": grid_cells
    }

def compute_prescription_economics(
    prescription_data: dict,
    field_area_ha: float = 5.0,
    base_rate_l_ha: float = 200.0,
    chemical_cost_per_l_inr: float = 650.0,
    active_ingredient_kg_l: float = 0.48, # e.g., Ethofumesate / Phenmedipham sugar beet herbicide
    usd_inr_exchange_rate: float = 84.5
):
    """
    Computes rigorous agricultural economics, cost comparisons, and environmental savings.
    Compares Conventional Uniform Broadcast Spraying vs. AI Precision Spot-Spraying.
    """
    mean_dosage = prescription_data["mean_dosage_l_ha"]
    nozzle_shutoff_pct = prescription_data["nozzle_shutoff_pct"]
    
    # 1. Conventional Uniform Broadcast Spraying Baseline
    conv_total_vol_l = round(field_area_ha * base_rate_l_ha, 2)
    conv_cost_inr = round(conv_total_vol_l * chemical_cost_per_l_inr, 2)
    conv_cost_usd = round(conv_cost_inr / usd_inr_exchange_rate, 2)
    conv_active_chem_kg = round(conv_total_vol_l * active_ingredient_kg_l, 2)

    # 2. AI Precision Spot-Spraying (Variable Rate Application)
    ai_total_vol_l = round(field_area_ha * mean_dosage, 2)
    ai_cost_inr = round(ai_total_vol_l * chemical_cost_per_l_inr, 2)
    ai_cost_usd = round(ai_cost_inr / usd_inr_exchange_rate, 2)
    ai_active_chem_kg = round(ai_total_vol_l * active_ingredient_kg_l, 2)

    # 3. Savings & Reductions
    vol_saved_l = round(conv_total_vol_l - ai_total_vol_l, 2)
    vol_reduction_pct = round((vol_saved_l / (conv_total_vol_l + 1e-6)) * 100, 2)
    cost_saved_inr = round(conv_cost_inr - ai_cost_inr, 2)
    cost_saved_usd = round(conv_cost_usd - ai_cost_usd, 2)
    active_chem_spared_kg = round(conv_active_chem_kg - ai_active_chem_kg, 2)

    # 4. Carbon & Drone Operational Efficiency
    # Broadcast requires ~4-5 drone battery payload refills per 5 ha. Precision saves refills.
    drone_payload_capacity_l = 16.0 # Standard DJI Agras T30/T40 tank
    conv_drone_flights = int(np.ceil(conv_total_vol_l / drone_payload_capacity_l))
    ai_drone_flights = int(np.ceil(ai_total_vol_l / drone_payload_capacity_l))
    drone_flights_saved = max(0, conv_drone_flights - ai_drone_flights)
    co2_offset_kg = round(drone_flights_saved * 1.85, 2) # Saved generator battery charges

    return {
        "field_area_ha": field_area_ha,
        "base_rate_l_ha": base_rate_l_ha,
        "chemical_cost_per_l_inr": chemical_cost_per_l_inr,
        "chemical_cost_per_l_usd": round(chemical_cost_per_l_inr / usd_inr_exchange_rate, 2),
        "conventional": {
            "total_volume_liters": conv_total_vol_l,
            "total_cost_inr": conv_cost_inr,
            "total_cost_usd": conv_cost_usd,
            "active_ingredient_kg": conv_active_chem_kg,
            "drone_payload_flights": conv_drone_flights
        },
        "ai_precision": {
            "total_volume_liters": ai_total_vol_l,
            "total_cost_inr": ai_cost_inr,
            "total_cost_usd": ai_cost_usd,
            "active_ingredient_kg": ai_active_chem_kg,
            "drone_payload_flights": ai_drone_flights
        },
        "savings": {
            "volume_saved_liters": vol_saved_l,
            "volume_reduction_pct": vol_reduction_pct,
            "financial_saved_inr": cost_saved_inr,
            "financial_saved_usd": cost_saved_usd,
            "active_chem_spared_kg": active_chem_spared_kg,
            "nozzle_shutoff_pct": nozzle_shutoff_pct,
            "drone_flights_saved": drone_flights_saved,
            "co2_offset_kg": co2_offset_kg
        }
    }

def generate_prescription_visualization(
    rgb_image_np: np.ndarray,
    prescription_data: dict,
    economics: dict,
    output_path: str = None
):
    """
    Renders publication-grade 4-panel Variable Rate Application (VRA) prescription diagram.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200)
    
    # Panel 1: Original UAV Field Image
    axes[0].imshow(rgb_image_np)
    axes[0].set_title("1. High-Resolution UAV Aerial Input (1.0 cm GSD)", fontsize=11, fontweight="bold", pad=8)
    axes[0].axis("off")

    # Panel 2: 2D Spatial VRA Prescription Grid
    grid_size = prescription_data["grid_size"]
    tier_map = prescription_data["tier_map"]
    
    # Create colorized RGB grid overlay
    color_grid = np.zeros((*tier_map.shape, 3), dtype=np.uint8)
    for t_id, t_info in DOSAGE_TIERS.items():
        color_grid[tier_map == t_id] = t_info["color_rgb"]

    im2 = axes[1].imshow(color_grid, interpolation="nearest")
    axes[1].set_title(f"2. Variable Rate Application (VRA) Map ({grid_size}×{grid_size} Zones)", fontsize=11, fontweight="bold", pad=8)
    axes[1].set_xticks(range(0, grid_size, 4))
    axes[1].set_yticks(range(0, grid_size, 4))
    axes[1].set_xlabel("Field Coordinate X (Grid Cells)")
    axes[1].set_ylabel("Field Coordinate Y (Grid Cells)")
    
    # Grid lines
    for i in range(grid_size + 1):
        axes[1].axhline(i - 0.5, color="#ffffff", lw=0.6, alpha=0.5)
        axes[1].axvline(i - 0.5, color="#ffffff", lw=0.6, alpha=0.5)

    # Custom legend for tiers
    legend_elements = [
        patches.Patch(facecolor=DOSAGE_TIERS[0]["color_hex"], edgecolor="#000", label=f"Tier 0: No Spray (0 L/ha) - {prescription_data['tier_counts'][0]} zones"),
        patches.Patch(facecolor=DOSAGE_TIERS[1]["color_hex"], edgecolor="#000", label=f"Tier 1: Pulse 30% ({round(0.3*economics['base_rate_l_ha'])} L/ha) - {prescription_data['tier_counts'][1]} zones"),
        patches.Patch(facecolor=DOSAGE_TIERS[2]["color_hex"], edgecolor="#000", label=f"Tier 2: Targeted 70% ({round(0.7*economics['base_rate_l_ha'])} L/ha) - {prescription_data['tier_counts'][2]} zones"),
        patches.Patch(facecolor=DOSAGE_TIERS[3]["color_hex"], edgecolor="#000", label=f"Tier 3: Full 100% ({round(1.0*economics['base_rate_l_ha'])} L/ha) - {prescription_data['tier_counts'][3]} zones"),
    ]
    axes[1].legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.32), fontsize=8.5, frameon=True)

    # Panel 3: Economics & Chemical Reduction Bar Chart
    sav = economics["savings"]
    conv = economics["conventional"]
    ai = economics["ai_precision"]

    categories = ["Herbicide (L)", "Cost (₹ in 1000s)", "Pesticide Runoff (kg)"]
    conv_vals = [conv["total_volume_liters"], conv["total_cost_inr"] / 1000.0, conv["active_ingredient_kg"]]
    ai_vals = [ai["total_volume_liters"], ai["total_cost_inr"] / 1000.0, ai["active_ingredient_kg"]]

    x = np.arange(len(categories))
    w = 0.35

    bars1 = axes[2].bar(x - w/2, conv_vals, w, label="Conventional Broadcast", color="#ef4444", alpha=0.85, edgecolor="#b91c1c")
    bars2 = axes[2].bar(x + w/2, ai_vals, w, label="AI Precision Spot-Spray", color="#10b981", alpha=0.85, edgecolor="#047857")

    axes[2].set_title(f"3. Economic & Environmental Savings ({sav['volume_reduction_pct']}% Slashed)", fontsize=11, fontweight="bold", pad=8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(categories, fontsize=9.5, fontweight="bold")
    axes[2].legend(loc="upper right", fontsize=8.5)
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)

    # Value labels on bars
    for b in bars1:
        val = b.get_height()
        axes[2].text(b.get_x() + b.get_width()/2, val + 1, f"{val:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for b in bars2:
        val = b.get_height()
        axes[2].text(b.get_x() + b.get_width()/2, val + 1, f"{val:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#047857")

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        return output_path
        
    return fig

def export_dji_agras_mission(prescription_data: dict, base_lat: float = 50.627500, base_lon: float = 6.945000, altitude_m: float = 3.5, flight_speed_ms: float = 4.0):
    """
    Generates standard DJI Agras / XAG Drone Spraying Mission CSV.
    Each waypoint represents an autonomous GPS coordinate with assigned nozzle PWM flow rates.
    """
    grid_size = prescription_data["grid_size"]
    # 0.0001 deg lat ~= 11.1 meters (approx field cell dimension)
    lat_step = 0.00005
    lon_step = 0.00008

    mission_records = []
    waypoint_id = 1

    for cell in prescription_data["cells"]:
        r = cell["grid_row"]
        c = cell["grid_col"]
        cell_lat = round(base_lat + (r * lat_step), 7)
        cell_lon = round(base_lon + (c * lon_step), 7)

        mission_records.append({
            "Waypoint_ID": waypoint_id,
            "Latitude": cell_lat,
            "Longitude": cell_lon,
            "Altitude_m": altitude_m,
            "Flight_Speed_ms": flight_speed_ms,
            "Weed_Density_Pct": cell["weed_density_pct"],
            "Prescription_Tier": cell["tier"],
            "Dosage_L_ha": cell["dosage_l_ha"],
            "PWM_Duty_Cycle_Pct": cell["pwm_pct"],
            "Nozzle_Action": cell["nozzle_state"]
        })
        waypoint_id += 1

    df_mission = pd.DataFrame(mission_records)
    return df_mission

def export_geojson_prescription(prescription_data: dict, base_lat: float = 50.627500, base_lon: float = 6.945000):
    """
    Produces standard GeoJSON FeatureCollection for import into GIS software (QGIS, ArcGIS).
    """
    grid_size = prescription_data["grid_size"]
    lat_step = 0.00005
    lon_step = 0.00008

    features = []
    for cell in prescription_data["cells"]:
        r = cell["grid_row"]
        c = cell["grid_col"]

        min_lat = base_lat + (r * lat_step)
        max_lat = min_lat + lat_step
        min_lon = base_lon + (c * lon_step)
        max_lon = min_lon + lon_step

        polygon = [
            [round(min_lon, 7), round(min_lat, 7)],
            [round(max_lon, 7), round(min_lat, 7)],
            [round(max_lon, 7), round(max_lat, 7)],
            [round(min_lon, 7), round(max_lat, 7)],
            [round(min_lon, 7), round(min_lat, 7)]
        ]

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon]
            },
            "properties": {
                "grid_row": r,
                "grid_col": c,
                "weed_density_pct": cell["weed_density_pct"],
                "prescription_tier": cell["tier"],
                "dosage_l_ha": cell["dosage_l_ha"],
                "nozzle_state": cell["nozzle_state"],
                "pwm_pct": cell["pwm_pct"],
                "color": cell["color_hex"]
            }
        }
        features.append(feature)

    geojson_data = {
        "type": "FeatureCollection",
        "metadata": {
            "project": "Cross-Scale AgriFusion",
            "institution": "PES University (PW26_YBS_02)",
            "mean_dosage_l_ha": prescription_data["mean_dosage_l_ha"],
            "nozzle_shutoff_pct": prescription_data["nozzle_shutoff_pct"]
        },
        "features": features
    }
    return geojson_data

