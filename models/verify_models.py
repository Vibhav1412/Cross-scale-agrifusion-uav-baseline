"""
Step 5: Architectural Verification Suite for Segmentation Models
Validates forward passes, output dimensions, parameter counts,
memory footprints, and inference latency for U-Net, DeepLabV3+, and Basic CNN.
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import pandas as pd

from models.unet import build_unet
from models.baseline_cnn import build_baseline_cnn
from models.deeplab import build_deeplabv3plus

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    return total, trainable, size_mb

def verify_all_models():
    print("=" * 70)
    print("STEP 5: MODEL ARCHITECTURE IMPLEMENTATION & VERIFICATION")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verification Device: {device}")
    
    # Dummy input: Batch=4, RGB=3, H=256, W=256
    dummy_input = torch.randn(4, 3, 256, 256, device=device)
    dummy_aux = torch.randn(4, 16, 128, 128, device=device) # For Phase-II fusion hook
    
    models = {
        "U-Net (ResNet18) [Proposed Baseline]": build_unet(num_classes=3, pretrained=False),
        "DeepLabV3+ (ResNet18)": build_deeplabv3plus(num_classes=3, pretrained=False),
        "Basic CNN Segmenter": build_baseline_cnn(num_classes=3)
    }
    
    summary_records = []
    
    for name, model in models.items():
        model = model.to(device)
        model.eval()
        
        total_params, trainable_params, size_mb = count_parameters(model)
        
        # Warmup and Latency benchmark
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)
                
            start_time = time.time()
            iters = 10
            for _ in range(iters):
                output = model(dummy_input)
            latency_ms = ((time.time() - start_time) / iters) * 1000
            
        # Assertions
        assert output.shape == (4, 3, 256, 256), f"Wrong output shape: {output.shape}"
        assert not torch.isnan(output).any(), "NaN values detected in output logits"
        assert not torch.isinf(output).any(), "Inf values detected in output logits"
        
        print(f"\n[MODEL]: {name}")
        print(f"  Input Shape:     {tuple(dummy_input.shape)}")
        print(f"  Output Shape:    {tuple(output.shape)}")
        print(f"  Total Params:    {total_params:,} ({total_params/1e6:.2f} M)")
        print(f"  Trainable:       {trainable_params:,}")
        print(f"  Model Size:      {size_mb:.2f} MB")
        print(f"  Inference (B=4): {latency_ms:.2f} ms ({latency_ms/4:.2f} ms/image)")
        print(f"  [PASS] Forward pass assertion verified.")
        
        # Test Cross-Scale satellite fusion hook on U-Net
        if "U-Net" in name:
            with torch.no_grad():
                fused_out = model(dummy_input, auxiliary_satellite_feat=dummy_aux)
                assert fused_out.shape == (4, 3, 256, 256)
            print("  [PASS] Phase-II Cross-Scale Fusion Hook verified.")
            
        summary_records.append({
            "Architecture": name,
            "Parameters (M)": round(total_params / 1e6, 2),
            "Size (MB)": round(size_mb, 2),
            "Latency B=4 (ms)": round(latency_ms, 2),
            "Per-Image Latency (ms)": round(latency_ms / 4, 2),
            "Output Resolution": "256x256x3"
        })

    # Save verification report table
    df = pd.DataFrame(summary_records)
    reports_dir = os.path.join(PROJECT_ROOT, "outputs/reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_csv = os.path.join(reports_dir, "model_architectures_summary.csv")
    df.to_csv(report_csv, index=False)
    
    print("\n" + "=" * 70)
    print("ARCHITECTURAL VERIFICATION BENCHMARK SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nSaved architecture summary to: {report_csv}")
    print("=" * 70)

if __name__ == "__main__":
    verify_all_models()

