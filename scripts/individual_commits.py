"""
Script to create individual, granular, high-quality git commits for every single file.
"""

import os
import subprocess
import sys

# Define explicit ordered list of (file_path, commit_message)
commits = [
    (".gitignore", "chore: initialize repository ignore rules"),
    ("requirements.txt", "build: define python dependencies and torch environment"),
    ("config.yaml", "config: establish global hyperparameters and training settings"),

    # Documentation
    ("docs/PHASE_1_DATASET_RESEARCH.md", "docs(research): add comparative UAV dataset analysis across 13 criteria"),
    ("docs/DUAL_DATASET_STRATEGY.md", "docs(strategy): establish dual-dataset baseline and expansion strategy"),
    ("docs/PHASE_3_MODEL_ARCHITECTURES.md", "docs(models): document segmentation model design specifications"),
    ("docs/PHASE_4_PREPROCESSING_REPORT.md", "docs(data): document preprocessing, boundary filtering, and splitting"),
    ("docs/FINAL_RESEARCH_EXPERIMENT_REPORT.md", "docs(results): author comprehensive research experiment and benchmark report"),

    # Data Splits & Weights
    ("data/splits/class_weights.yaml", "feat(data): add normalized median frequency class weights"),
    ("data/splits/train.csv", "feat(data): add leakage-free training split (flights 000, 002, 004)"),
    ("data/splits/val.csv", "feat(data): add validation split metadata (flight 001)"),
    ("data/splits/test.csv", "feat(data): add held-out test split metadata (flight 003)"),
    ("data/splits/all_effective_tiles.csv", "feat(data): add filtered non-boundary tile master registry"),

    # Source code
    ("src/inspect_dataset.py", "feat(dataset): implement dataset verification and structure inspector"),
    ("src/verify_sample_masks.py", "test(masks): add pixel-level mask verification utility"),
    ("src/split_dataset.py", "feat(splits): implement flight-level spatial splitting generator"),
    ("src/augmentation.py", "feat(transforms): add synchronized geometric and photometric augmentations"),
    ("src/dataset.py", "feat(dataloader): implement custom PyTorch dataset and sliding-window loaders"),
    ("src/loss.py", "feat(loss): implement combined weighted cross-entropy and soft dice loss"),
    ("src/metrics.py", "feat(metrics): implement confusion matrix, IoU, Dice, Precision, and Recall calculators"),
    ("src/train.py", "feat(training): implement multi-epoch training pipeline with cosine annealing"),
    ("src/evaluate.py", "feat(evaluation): implement held-out test evaluation suite"),
    ("src/predict.py", "feat(inference): implement large-scale patch inference and reconstruction engine"),
    ("src/analytics.py", "feat(analytics): implement physical area estimation and spatial heatmap generator"),
    ("src/explainability.py", "feat(xai): implement class-specific Grad-CAM visual attribution module"),
    ("src/uncertainty.py", "feat(uncertainty): implement Monte Carlo Dropout predictive entropy estimation"),
    ("src/database.py", "feat(database): implement SQLite relational schema and audit logger"),
    ("src/compare_models.py", "feat(benchmarks): implement multi-model comparative benchmarking suite"),
    ("src/test_dataloader.py", "test(pipeline): add dataloader throughput and tensor verification test"),
    ("src/test_xai_uncertainty.py", "test(xai): add explainability and uncertainty validation test"),

    # Models
    ("models/baseline_cnn.py", "feat(model): implement basic convolutional encoder-decoder ablation baseline"),
    ("models/deeplab.py", "feat(model): implement DeepLabV3+ architecture with ASPP module"),
    ("models/unet.py", "feat(model): implement primary ResNet18 U-Net with satellite fusion adapter"),
    ("models/verify_models.py", "test(models): add parameter and inference latency benchmark verification"),

    # Outputs & DB
    ("outputs/checkpoints/.gitkeep", "chore(checkpoints): initialize checkpoints directory structure"),
    ("outputs/agrifusion.db", "feat(database): initialize SQLite relational audit store"),

    # Graphs
    ("outputs/graphs/unet_training_curves.png", "results(graphs): add ResNet18 U-Net training loss and mIoU curves"),
    ("outputs/graphs/deeplab_training_curves.png", "results(graphs): add DeepLabV3+ training loss and mIoU curves"),
    ("outputs/graphs/baseline_cnn_training_curves.png", "results(graphs): add Basic CNN training loss and mIoU curves"),
    ("outputs/graphs/models_comparison_barchart.png", "results(graphs): add multi-model generalization benchmark bar chart"),

    # Predictions
    ("outputs/predictions/unet_test_predictions_gallery.png", "results(predictions): add U-Net test segmentation comparison gallery"),
    ("outputs/predictions/deeplab_test_predictions_gallery.png", "results(predictions): add DeepLabV3+ test prediction gallery"),
    ("outputs/predictions/baseline_cnn_test_predictions_gallery.png", "results(predictions): add Basic CNN test prediction gallery"),
    ("outputs/predictions/cli_prediction_result.png", "results(predictions): add sample CLI patch inference result overlay"),

    # Reports
    ("outputs/reports/dataset_sample_verification.png", "results(reports): add dataset visual verification sample"),
    ("outputs/reports/augmented_batch_sample.png", "results(reports): add augmented batch sample verification figure"),
    ("outputs/reports/gradcam_sample_explanation.png", "results(reports): add Grad-CAM class attribution visual explanation"),
    ("outputs/reports/mc_dropout_uncertainty_sample.png", "results(reports): add Monte Carlo Dropout predictive entropy heatmap"),
    ("outputs/reports/model_architectures_summary.csv", "results(reports): add architectural specifications and latency benchmark CSV"),
    ("outputs/reports/unet_training_history.csv", "results(reports): add U-Net 10-epoch training history log"),
    ("outputs/reports/deeplab_training_history.csv", "results(reports): add DeepLabV3+ 10-epoch training history log"),
    ("outputs/reports/baseline_cnn_training_history.csv", "results(reports): add Basic CNN 10-epoch training history log"),
    ("outputs/reports/unet_test_evaluation.csv", "results(reports): add U-Net test evaluation metric table"),
    ("outputs/reports/deeplab_test_evaluation.csv", "results(reports): add DeepLabV3+ test evaluation metric table"),
    ("outputs/reports/baseline_cnn_test_evaluation.csv", "results(reports): add Basic CNN test evaluation metric table"),
    ("outputs/reports/models_benchmark_comparison.csv", "results(reports): add multi-model comparative benchmark table"),
    ("outputs/reports/test_heat.png", "results(reports): add sample risk heatmap test artifact"),

    # Architecture Diagrams
    ("outputs/reports/system_architecture_block_diagram.png", "docs(diagrams): add high-resolution system architecture block diagram"),
    ("outputs/reports/proposed_unet_architecture_detail.png", "docs(diagrams): add detailed ResNet18 U-Net neural architecture diagram"),
    ("outputs/reports/system_architecture_block_diagram_slide.png", "docs(diagrams): add presentation slide system block diagram (light mode)"),
    ("outputs/reports/system_architecture_block_diagram_slide_dark.png", "docs(diagrams): add presentation slide system block diagram (dark mode)"),
    ("outputs/reports/further_implementation_roadmap_slide.png", "docs(diagrams): add further implementation roadmap diagram (light mode)"),
    ("outputs/reports/further_implementation_roadmap_slide_dark.png", "docs(diagrams): add further implementation roadmap diagram (dark mode)"),

    # Demo Samples
    ("demo_samples/frame0008.png", "data(demo): add high weed infestation UAV test sample"),
    ("demo_samples/frame0076.png", "data(demo): add clean sugar beet crop row UAV test sample"),
    ("demo_samples/frame0103.png", "data(demo): add weed cluster UAV test sample"),
    ("demo_samples/frame0110.png", "data(demo): add mixed canopy UAV test sample"),
    ("demo_samples/frame0144.png", "data(demo): add moderate field UAV test sample"),

    # Scripts
    ("scripts/redEdge_from_ortho_to_tiles.m", "scripts(ortho): add MATLAB tiling utility script"),
    ("scripts/redEdge_from_tile_to_ortho_ground_truth.m", "scripts(ortho): add MATLAB ground truth reconstruction script"),
    ("scripts/generate_architecture_diagrams.py", "scripts(diagrams): add script for system and neural architecture diagram generation"),
    ("scripts/render_slide_block_diagram.py", "scripts(diagrams): add slide block diagram generator"),
    ("scripts/render_further_implementation_diagram.py", "scripts(diagrams): add further implementation roadmap generator"),

    # Main App & CLI
    ("main.py", "feat(cli): implement unified command-line interface with subcommands"),
    ("app/app.py", "feat(ui): implement interactive 7-tab Streamlit dashboard application"),

    # Final Master README
    ("README.md", "docs: publish comprehensive research documentation, benchmarks, and guides")
]

count = 0
for fpath, msg in commits:
    norm_path = fpath.replace("/", os.sep)
    if os.path.exists(norm_path):
        subprocess.run(["git", "add", norm_path], check=True)
        # Commit if changes staged
        res = subprocess.run(["git", "commit", "-m", msg], capture_output=True, text=True)
        if res.returncode == 0:
            count += 1
            print(f"[{count:02d}/{len(commits)}] Committed: {fpath} -> {msg}")
        else:
            print(f"Skipped/Empty: {fpath} ({res.stderr.strip() or res.stdout.strip()})")
    else:
        print(f"File not found: {fpath}")

print(f"\n[DONE] Successfully created {count} granular git commits!")
