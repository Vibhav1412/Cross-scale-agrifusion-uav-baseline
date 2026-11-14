"""
commit_segformer_updates.py
Commits each updated and new file individually to maximize GitHub contribution score.
"""

import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

commits = [
    (".gitignore", "chore: update gitignore rules for checkpoints and cache"),
    ("models/segformer.py", "feat(model): implement SegFormer (MiT-B0) Vision Transformer with cross-scale satellite fusion adapter"),
    ("src/explainability.py", "feat(xai): upgrade to class-masked Grad-CAM and universal multi-architecture layer resolution"),
    ("src/predict.py", "feat(inference): support dynamic model loading and SegFormer patch reconstruction"),
    ("src/train.py", "feat(training): integrate SegFormer training pipeline with pretrained Mix-Transformer weights"),
    ("src/evaluate.py", "feat(evaluation): add SegFormer checkpoint evaluation on held-out flight 003"),
    ("src/compare_models.py", "feat(benchmarks): add SegFormer to comparative benchmarking suite"),
    ("models/verify_models.py", "test(models): add SegFormer parameter and CPU inference latency benchmarks"),
    ("main.py", "feat(cli): add segformer choices to CLI actions"),
    ("outputs/reports/segformer_training_history.csv", "results(data): add SegFormer 10-epoch training loss and validation mIoU metrics"),
    ("outputs/reports/segformer_test_evaluation.csv", "results(data): add SegFormer held-out test evaluation metrics (54.33% mIoU)"),
    ("outputs/graphs/segformer_training_curves.png", "results(graphs): add SegFormer training loss convergence and validation mIoU curves"),
    ("outputs/predictions/segformer_test_predictions_gallery.png", "results(predictions): add SegFormer test flight 003 prediction gallery"),
    ("outputs/reports/model_architectures_summary.csv", "results(benchmarks): update model architectures summary table with SegFormer"),
    ("outputs/reports/models_benchmark_comparison.csv", "results(benchmarks): update multi-model benchmark comparison table with SegFormer #1 ranking"),
    ("outputs/graphs/models_comparison_barchart.png", "results(graphs): update 4-model comparative bar chart with SegFormer"),
    ("scripts/render_all_segformer_slides.py", "feat(scripts): add presentation slides rendering generator for SegFormer"),
    ("outputs/reports/segformer_architecture_detail_slide.png", "docs(presentation): add SegFormer MiT-B0 architecture detail slide (Light Mode)"),
    ("outputs/reports/segformer_architecture_detail_slide_dark.png", "docs(presentation): add SegFormer MiT-B0 architecture detail slide (Dark Mode)"),
    ("outputs/reports/system_architecture_block_diagram_slide.png", "docs(presentation): update 5-stage system pipeline slide with SegFormer (Light Mode)"),
    ("outputs/reports/system_architecture_block_diagram_slide_dark.png", "docs(presentation): update 5-stage system pipeline slide with SegFormer (Dark Mode)"),
    ("outputs/reports/architecture_highlights_slide.png", "docs(presentation): update architecture highlights slide with SegFormer (Light Mode)"),
    ("outputs/reports/architecture_highlights_slide_dark.png", "docs(presentation): update architecture highlights slide with SegFormer (Dark Mode)"),
    ("outputs/reports/further_implementation_roadmap_slide.png", "docs(presentation): update further implementation roadmap slide with SegFormer (Light Mode)"),
    ("outputs/reports/further_implementation_roadmap_slide_dark.png", "docs(presentation): update further implementation roadmap slide with SegFormer (Dark Mode)"),
    ("app/app.py", "feat(ui): add sidebar model selector with SegFormer default and updated benchmark dashboard"),
]

for file_path, msg in commits:
    full_p = os.path.join(PROJECT_ROOT, file_path)
    if os.path.exists(full_p):
        subprocess.run(["git", "add", file_path], cwd=PROJECT_ROOT, check=True)
        # Check if there are changes staged
        diff_res = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=PROJECT_ROOT)
        if diff_res.returncode != 0:
            subprocess.run(["git", "commit", "-m", msg], cwd=PROJECT_ROOT, check=True)
            print(f"[COMMITTED] {file_path} -> '{msg}'")
        else:
            print(f"[SKIP] No changes staged for {file_path}")
    else:
        print(f"[WARN] File not found: {file_path}")
