"""
commit_resnet50_updates.py
Commits each updated and new file individually for the ResNet50 baseline integration.
"""

import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

commits = [
    ("models/unet.py", "feat(model): integrate ResNet50 backbone (43.88M params) into U-Net architecture"),
    ("src/train.py", "feat(training): add ResNet50 training pipeline with ImageNet pre-trained weights"),
    ("src/evaluate.py", "feat(evaluation): add ResNet50 test evaluation on held-out flight 003"),
    ("src/predict.py", "feat(inference): support ResNet50 checkpoint inference and sliding-window reconstruction"),
    ("src/compare_models.py", "feat(benchmarks): add ResNet50 heavyweight baseline to 5-model comparative benchmark"),
    ("models/verify_models.py", "test(models): add ResNet50 parameter and latency benchmark verification"),
    ("main.py", "feat(cli): add resnet50 choice to CLI arguments"),
    ("app/app.py", "feat(ui): add ResNet50 option to sidebar model selector and benchmark tab"),
    ("outputs/reports/resnet50_training_history.csv", "results(data): add ResNet50 10-epoch training and validation loss history"),
    ("outputs/reports/resnet50_test_evaluation.csv", "results(data): add ResNet50 test evaluation metrics (51.10% mIoU on flight 003)"),
    ("outputs/graphs/resnet50_training_curves.png", "results(graphs): add ResNet50 training loss convergence and validation mIoU curves"),
    ("outputs/predictions/resnet50_test_predictions_gallery.png", "results(predictions): add ResNet50 test flight 003 prediction gallery"),
    ("outputs/reports/model_architectures_summary.csv", "results(benchmarks): update model architectures summary table with 5 models"),
    ("outputs/reports/models_benchmark_comparison.csv", "results(benchmarks): update multi-model benchmark comparison table with ResNet50"),
    ("outputs/graphs/models_comparison_barchart.png", "results(graphs): update 5-model comparative bar chart with ResNet50"),
    ("scripts/render_all_segformer_slides.py", "feat(scripts): update slide generator with ResNet50 David vs Goliath metrics"),
    ("outputs/reports/segformer_architecture_detail_slide.png", "docs(presentation): update SegFormer detail slide with ResNet50 comparison (Light Mode)"),
    ("outputs/reports/segformer_architecture_detail_slide_dark.png", "docs(presentation): update SegFormer detail slide with ResNet50 comparison (Dark Mode)"),
    ("outputs/reports/architecture_highlights_slide.png", "docs(presentation): update architecture highlights slide with ResNet50 comparison (Light Mode)"),
    ("outputs/reports/architecture_highlights_slide_dark.png", "docs(presentation): update architecture highlights slide with ResNet50 comparison (Dark Mode)"),
    ("outputs/reports/further_implementation_roadmap_slide.png", "docs(presentation): update roadmap slide with 5-model benchmark (Light Mode)"),
    ("outputs/reports/further_implementation_roadmap_slide_dark.png", "docs(presentation): update roadmap slide with 5-model benchmark (Dark Mode)"),
]

for file_path, msg in commits:
    full_p = os.path.join(PROJECT_ROOT, file_path)
    if os.path.exists(full_p):
        subprocess.run(["git", "add", file_path], cwd=PROJECT_ROOT, check=True)
        diff_res = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=PROJECT_ROOT)
        if diff_res.returncode != 0:
            subprocess.run(["git", "commit", "-m", msg], cwd=PROJECT_ROOT, check=True)
            print(f"[COMMITTED] {file_path} -> '{msg}'")
        else:
            print(f"[SKIP] No changes staged for {file_path}")
    else:
        print(f"[WARN] File not found: {file_path}")
