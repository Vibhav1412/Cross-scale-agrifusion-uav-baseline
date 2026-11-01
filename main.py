"""
Unified Command-Line Interface (CLI) for UAV Crop-Weed Baseline
Cross-Scale UAV-Satellite Data Fusion (PES University Capstone PW26_YBS_02)
"""

import os
import sys
import time
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description="Cross-Scale AgriFusion: UAV Crop-Weed Baseline CLI")
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=[
            "inspect",
            "split",
            "verify_data",
            "verify_models",
            "train",
            "evaluate",
            "predict",
            "compare",
            "app"
        ],
        help="Action to execute."
    )
    # Training & Model arguments
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "deeplab", "baseline_cnn"])
    parser.add_argument("--epochs", type=int, default=10)
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
    
    # Inference arguments
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.action == "inspect":
        from src.inspect_dataset import inspect_dataset
        inspect_dataset()
        
    elif args.action == "split":
        from src.split_dataset import create_splits
        create_splits()
        
    elif args.action == "verify_data":
        from src.test_dataloader import test_pipeline
        test_pipeline()
        
    elif args.action == "verify_models":
        from models.verify_models import verify_all_models
        verify_all_models()
        
    elif args.action == "train":
        from src.train import train_model
        train_model(args)
        
    elif args.action == "evaluate":
        from src.evaluate import evaluate_model
        evaluate_model(args)
        
    elif args.action == "compare":
        from src.compare_models import compare_all_models
        compare_all_models()
        
    elif args.action == "predict":
        from PIL import Image
        import numpy as np
        import matplotlib.pyplot as plt
        from src.predict import load_inference_model, predict_large_image, create_prediction_overlay
        from src.analytics import compute_field_coverage, generate_infestation_heatmap, generate_decision_support_report
        from src.database import log_prediction
        
        img_path = args.image or "C:/Users/Admin/Downloads/Tiles/Tiles/RedEdge/003/tile/RGB/frame0008.png"
        print(f"Running inference on: {img_path}")
        image_pil = Image.open(img_path).convert("RGB")
        model, device = load_inference_model(args.checkpoint)
        
        t0 = time.time()
        pred_mask, _ = predict_large_image(model, image_pil, device)
        inf_latency_ms = (time.time() - t0) * 1000
        
        overlay, color_mask = create_prediction_overlay(np.array(image_pil), pred_mask)
        coverage = compute_field_coverage(pred_mask)
        _, _, _, heatmap_summary = generate_infestation_heatmap(pred_mask, grid_size=16)
        decision_report = generate_decision_support_report(coverage, heatmap_summary)
        
        # Log to SQLite Database
        record_id = log_prediction(
            image_name=img_path,
            coverage=coverage,
            heatmap_summary=heatmap_summary,
            decision_report=decision_report,
            inference_time_ms=inf_latency_ms
        )
        print(f"[DATABASE] Logged prediction to outputs/agrifusion.db (Record #{record_id})")
        
        print("\n--- Coverage Analytics ---")
        print(f"Weed Coverage: {coverage['weed_coverage_pct']}% ({coverage['physical_weed_area_m2']} m²)")
        print(f"Crop Coverage: {coverage['crop_coverage_pct']}% ({coverage['physical_crop_area_m2']} m²)")
        print(f"Weed-to-Crop Ratio: {coverage['weed_to_crop_ratio']}")
        print(f"Agronomic Priority: {decision_report['assessment_priority']}")
        print(f"Recommended Action: {decision_report['recommended_action']}")
        
        out_path = os.path.join(PROJECT_ROOT, "outputs/predictions/cli_prediction_result.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image_pil)
        axes[0].set_title("Input UAV Image")
        axes[0].axis("off")
        
        axes[1].imshow(color_mask)
        axes[1].set_title("Predicted Mask (Green:Crop, Red:Weed)")
        axes[1].axis("off")
        
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay Prediction")
        axes[2].axis("off")
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved prediction visual: {out_path}")
        
    elif args.action == "app":
        print("Launching Streamlit Web Dashboard...")
        app_file = os.path.join(PROJECT_ROOT, "app/app.py")
        subprocess.run(["streamlit", "run", app_file])

if __name__ == "__main__":
    main()
