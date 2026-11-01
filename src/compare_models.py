"""
Feature 7: Model Comparison & Benchmarking Suite
Compares Primary U-Net vs DeepLabV3+ vs Basic CNN across:
- Parameter Count & Model Size
- Inference Latency (ms)
- Precision, Recall, Dice, and IoU (Crop, Weed, Macro)
Generates publication-ready comparative bar charts and summary reports.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def compare_all_models():
    print("=" * 75)
    print("FEATURE 7: MULTI-MODEL BENCHMARK & COMPARISON SUITE")
    print("=" * 75)
    
    reports_dir = os.path.join(PROJECT_ROOT, "outputs/reports")
    graphs_dir = os.path.join(PROJECT_ROOT, "outputs/graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    
    arch_summary_path = os.path.join(reports_dir, "model_architectures_summary.csv")
    if not os.path.exists(arch_summary_path):
        print("Architecture summary not found. Run verify_models first.")
        return
        
    arch_df = pd.read_csv(arch_summary_path)
    print("\nArchitectural Comparison:")
    print(arch_df.to_string(index=False))
    
    # Check evaluated checkpoints
    models = ["unet", "deeplab", "baseline_cnn"]
    model_eval_records = []
    
    for m in models:
        eval_csv = os.path.join(reports_dir, f"{m}_test_evaluation.csv")
        ckpt_path = os.path.join(PROJECT_ROOT, f"outputs/checkpoints/{m}_best.pth")
        
        if os.path.exists(eval_csv):
            df_eval = pd.read_csv(eval_csv).set_index("Metric")
            miou = float(df_eval.loc["mIoU", "Value"]) * 100
            mdice = float(df_eval.loc["Mean Dice (mDice)", "Value"]) * 100
            crop_iou = float(df_eval.loc["Crop IoU", "Value"]) * 100
            weed_iou = float(df_eval.loc["Weed IoU", "Value"]) * 100
            recall = float(df_eval.loc["Macro Recall", "Value"]) * 100
            prec = float(df_eval.loc["Macro Precision", "Value"]) * 100
            status = "Trained & Evaluated"
        else:
            miou, mdice, crop_iou, weed_iou, recall, prec = None, None, None, None, None, None
            status = "Trained Checkpoint Available" if os.path.exists(ckpt_path) else "Pending Training"
            
        model_eval_records.append({
            "Model": m.upper(),
            "Status": status,
            "Test mIoU (%)": miou,
            "Test mDice (%)": mdice,
            "Crop IoU (%)": crop_iou,
            "Weed IoU (%)": weed_iou,
            "Macro Recall (%)": recall,
            "Macro Prec (%)": prec
        })
        
    comp_df = pd.DataFrame(model_eval_records)
    print("\nTest Set Performance Comparison (Held-Out Flight 003):")
    print(comp_df.to_string(index=False))
    
    # Plot comparison chart if at least one model has metrics
    valid_plot_data = comp_df.dropna(subset=["Test mIoU (%)"])
    if len(valid_plot_data) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(valid_plot_data))
        width = 0.2
        
        ax.bar(x - 1.5*width, valid_plot_data["Test mIoU (%)"], width, label="Mean IoU", color="#4f46e5")
        ax.bar(x - 0.5*width, valid_plot_data["Test mDice (%)"], width, label="Mean Dice (F1)", color="#06b6d4")
        ax.bar(x + 0.5*width, valid_plot_data["Crop IoU (%)"], width, label="Crop IoU", color="#10b981")
        ax.bar(x + 1.5*width, valid_plot_data["Weed IoU (%)"], width, label="Weed IoU", color="#f59e0b")
        
        ax.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold")
        ax.set_title("Held-Out Test Generalization Performance by Architecture", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(valid_plot_data["Model"], fontsize=11, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        chart_path = os.path.join(graphs_dir, "models_comparison_barchart.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"\nSaved model comparison bar chart: {chart_path}")
        
    # Save combined report
    combined_report_path = os.path.join(reports_dir, "models_benchmark_comparison.csv")
    comp_df.to_csv(combined_report_path, index=False)
    print(f"Saved benchmark comparison CSV: {combined_report_path}")
    print("=" * 75)

if __name__ == "__main__":
    compare_all_models()

