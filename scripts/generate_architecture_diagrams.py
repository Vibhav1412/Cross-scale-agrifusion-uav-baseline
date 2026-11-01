"""
Script: generate_architecture_diagrams.py
Generates high-resolution publication-quality architecture and block diagrams
for the Capstone Presentation (PW26_YBS_02, PES University).
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs/reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_system_block_diagram():
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Background canvas
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    # Title Banner
    ax.text(7.5, 8.5, "PROPOSED SYSTEM ARCHITECTURE & BLOCK DIAGRAM", 
            ha="center", va="center", fontsize=16, fontweight="bold", color="#0f172a", fontfamily="sans-serif")
    ax.text(7.5, 8.15, "Cross-Scale UAV-Satellite Data Fusion: AI-Based Precision Agriculture (PES University | PW26_YBS_02)", 
            ha="center", va="center", fontsize=10.5, color="#475569", fontfamily="sans-serif")

    # Helper function to draw rounded boxes
    def draw_box(x, y, w, h, title, subtitle, items, bg_color, border_color, title_color="#0f172a"):
        # Shadow
        shadow = patches.FancyBboxPatch((x + 0.05, y - 0.05), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                        facecolor="#cbd5e1", edgecolor="none", alpha=0.5, zorder=1)
        ax.add_patch(shadow)
        
        # Main Box
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                     facecolor=bg_color, edgecolor=border_color, linewidth=2, zorder=2)
        ax.add_patch(box)
        
        # Header text
        ax.text(x + w/2, y + h - 0.35, title, ha="center", va="center", fontsize=11, fontweight="bold", color=title_color, zorder=3)
        if subtitle:
            ax.text(x + w/2, y + h - 0.65, subtitle, ha="center", va="center", fontsize=8.5, fontstyle="italic", color="#475569", zorder=3)
        
        # Bullet items
        start_y = y + h - 1.05
        for i, item in enumerate(items):
            ax.text(x + 0.2, start_y - (i * 0.42), f"• {item}", ha="left", va="center", fontsize=8.2, color="#1e293b", zorder=3)

    # 1. Ingestion Layer
    draw_box(0.6, 3.8, 3.0, 3.8,
             "1. DATA INGESTION & PREP", "Aerial Inputs & Balancing",
             ["WeedMap UAV Flights", "RGB Imagery (GSD: 1.0 cm/px)", "Boundary Filtering (639 Tiles)", "Sliding Window Patches (256x256)", "Spatial Geometric Augmentations", "Inverse Class Weights", "Strict Flight-Level Split (003 Test)"],
             "#e0f2fe", "#0284c7", "#0369a1")

    # 2. Primary Deep Learning Engine
    draw_box(4.1, 3.8, 4.4, 3.8,
             "2. AI SEGMENTATION ENGINE", "Proposed Primary ResNet18 U-Net",
             ["Pretrained ResNet18 Encoder", "Hierarchical Residual Stages (L1-L4)", "Multi-Scale Skip Connections", "Phase-II Satellite Fusion Adapter", "1x1 Conv Head -> 3 Classes", "Soft Dice + Weighted CE Loss", "Cosine Annealing Learning Scheduler"],
             "#f0fdf4", "#16a34a", "#15803d")

    # 3. Trustworthy AI Module
    draw_box(8.9, 5.0, 5.5, 2.6,
             "3. TRUSTWORTHY AI & EXPLAINABILITY", "Interpretability & Boundary Safety",
             ["Class-Specific Grad-CAM (Target: model.layer4 for Crop vs Weed)", "Monte Carlo Dropout (T=8 Stochastic Forward Passes)", "Pixel-Wise Predictive Entropy: H(x) = -sum(p log p)", "High-Uncertainty Boundary Localization"],
             "#fef3c7", "#d97706", "#b45309")

    # 4. Spatial Analytics & Decision Support
    draw_box(8.9, 1.8, 5.5, 2.8,
             "4. SPATIAL ANALYTICS & DECISION SUPPORT", "Agronomic Action Intelligence",
             ["GSD Physical Area Translation (m²)", "Weed & Crop Coverage Percentage", "NxN Grid Spatial Weed Infestation Heatmaps", "3-Tier Risk Classification (Low, Moderate, High)", "Targeted Prescription Spot-Spraying Guidelines"],
             "#f5f3ff", "#7c3aed", "#6d28d9")

    # 5. Deployment & Relational Storage (Bottom Wide Box)
    draw_box(0.6, 0.4, 7.9, 2.8,
             "5. DEPLOYMENT & RELATIONAL AUDIT SYSTEM", "Streamlit UI & SQLite Database",
             ["Interactive 7-Tab Streamlit Web Application (Overview, Inference, Heatmaps, XAI, Uncertainty, Benchmarks, History)", "SQLite Relational Database (outputs/agrifusion.db) persisting full audit logs", "Automated CSV Audit Report Generator for agricultural compliance", "Live Side-by-Side Model Prediction Galleries & Confusion Matrices"],
             "#ede9fe", "#6366f1", "#4338ca")

    # Helper function for directional arrows
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor="#334155", edgecolor="#334155", width=2.0, headwidth=7, headlength=7, shrink=0.05),
                    zorder=4)
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 0.15, label, ha="center", va="center", fontsize=7.5, fontweight="bold", color="#334155",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.9), zorder=5)

    # Connecting Arrows
    draw_arrow(3.6, 5.7, 4.1, 5.7, "Normalized Patches")
    draw_arrow(8.5, 6.3, 8.9, 6.3, "Logits / Feature Maps")
    draw_arrow(8.5, 5.1, 8.9, 3.2, "Predicted Mask (0,1,2)")
    draw_arrow(8.9, 2.2, 8.5, 2.2, "Coverage & Heatmap")
    draw_arrow(6.3, 3.8, 6.3, 3.2, "Model Inference Pipeline")

    plt.tight_layout()
    out_path = os.path.join(REPORTS_DIR, "system_architecture_block_diagram.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[SUCCESS] Saved system architecture block diagram to: {out_path}")

def generate_unet_detailed_diagram():
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.axis("off")

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.text(7.5, 8.5, "PRIMARY PROPOSED MODEL: RESNET18 U-NET ARCHITECTURE", 
            ha="center", va="center", fontsize=15, fontweight="bold", color="#0f172a")
    ax.text(7.5, 8.15, "Hierarchical Encoder-Decoder with Skip Connections & Modular Phase-II Satellite Fusion Adapter", 
            ha="center", va="center", fontsize=10, color="#64748b")

    # Encoder Blocks (Left side, descending)
    encoder_blocks = [
        ("Input UAV Patch\n3 x 256 x 256", 1.0, 6.5, 2.0, 1.1, "#f1f5f9", "#94a3b8"),
        ("ResNet Initial Conv\n64 x 128 x 128 (MaxPool)", 1.0, 5.0, 2.0, 1.1, "#e0f2fe", "#0284c7"),
        ("ResNet Layer 1\n64 x 64 x 64", 1.0, 3.6, 2.0, 1.1, "#bae6fd", "#0284c7"),
        ("ResNet Layer 2\n128 x 32 x 32", 1.0, 2.2, 2.0, 1.1, "#7dd3fc", "#0284c7"),
        ("ResNet Layer 3\n256 x 16 x 16", 1.0, 0.8, 2.0, 1.1, "#38bdf8", "#0369a1"),
    ]

    # Bottleneck (Bottom Center)
    bottleneck = ("Bottleneck (Layer 4)\n512 x 8 x 8\n+ Monte Carlo Dropout (p=0.2)", 5.5, 0.6, 4.0, 1.3, "#fef3c7", "#d97706")

    # Decoder Blocks (Right side, ascending)
    decoder_blocks = [
        ("Decoder Block 4\nConvTrans + Skip(L3) -> 256 x 16 x 16", 11.5, 1.0, 2.8, 1.1, "#dcfce7", "#16a34a"),
        ("Decoder Block 3\nConvTrans + Skip(L2) -> 128 x 32 x 32", 11.5, 2.4, 2.8, 1.1, "#bbf7d0", "#16a34a"),
        ("Decoder Block 2\nConvTrans + Skip(L1) -> 64 x 64 x 64", 11.5, 3.8, 2.8, 1.1, "#86efac", "#16a34a"),
        ("Decoder Block 1\nConvTrans + Skip(Init) -> 32 x 128 x 128", 11.5, 5.2, 2.8, 1.1, "#4ade80", "#15803d"),
        ("Final Upsampling & 1x1 Conv\n3 x 256 x 256 (Softmax Logits)", 11.5, 6.6, 2.8, 1.1, "#22c55e", "#15803d"),
    ]

    # Draw Encoder
    for text, x, y, w, h, bg, border in encoder_blocks:
        b = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.1", facecolor=bg, edgecolor=border, linewidth=1.8)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8, fontweight="bold", color="#0f172a")

    # Draw Bottleneck
    text, x, y, w, h, bg, border = bottleneck
    b = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.1", facecolor=bg, edgecolor=border, linewidth=2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#92400e")

    # Satellite Fusion Adapter (Center Special Block)
    sat_b = patches.FancyBboxPatch((5.5, 3.2), 4.0, 1.8, boxstyle="round,pad=0.08,rounding_size=0.1", facecolor="#f3e8ff", edgecolor="#9333ea", linewidth=2, linestyle="--")
    ax.add_patch(sat_b)
    ax.text(7.5, 4.4, "PHASE-II CROSS-SCALE FUSION ADAPTER", ha="center", va="center", fontsize=9, fontweight="bold", color="#6b21a8")
    ax.text(7.5, 3.9, "Sentinel-2 Multi-Spectral Tensor (16 Ch.)\n1x1 Conv Projection + Residual Elementwise Addition\nFused = UAV_feat + Proj(Satellite_feat)", ha="center", va="center", fontsize=7.8, color="#581c87")

    # Draw Decoder
    for text, x, y, w, h, bg, border in decoder_blocks:
        b = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.1", facecolor=bg, edgecolor=border, linewidth=1.8)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8, fontweight="bold", color="#0f172a")

    # Downsampling Arrows (Encoder)
    for i in range(len(encoder_blocks)-1):
        x = 2.0
        y1 = encoder_blocks[i][2]
        y2 = encoder_blocks[i+1][2] + encoder_blocks[i+1][4]
        ax.annotate("", xy=(x, y2), xytext=(x, y1), arrowprops=dict(facecolor="#0284c7", edgecolor="#0284c7", width=1.5, headwidth=5))

    # Encoder to Bottleneck
    ax.annotate("", xy=(5.5, 1.25), xytext=(3.0, 1.35), arrowprops=dict(facecolor="#0284c7", edgecolor="#0284c7", width=1.5, headwidth=5))
    # Bottleneck to Decoder
    ax.annotate("", xy=(11.5, 1.55), xytext=(9.5, 1.25), arrowprops=dict(facecolor="#16a34a", edgecolor="#16a34a", width=1.5, headwidth=5))

    # Upsampling Arrows (Decoder)
    for i in range(len(decoder_blocks)-1):
        x = 12.9
        y1 = decoder_blocks[i][2] + decoder_blocks[i][4]
        y2 = decoder_blocks[i+1][2]
        ax.annotate("", xy=(x, y2), xytext=(x, y1), arrowprops=dict(facecolor="#16a34a", edgecolor="#16a34a", width=1.5, headwidth=5))

    # Skip Connections (Horizontal dashed arrows from encoder to decoder)
    skips = [
        (3.0, 5.55, 11.5, 5.75, "Skip 1 (64ch)"),
        (3.0, 4.15, 11.5, 4.35, "Skip 2 (64ch)"),
        (3.0, 2.75, 11.5, 2.95, "Skip 3 (128ch)"),
        (3.0, 1.35, 11.5, 1.55, "Skip 4 (256ch)"),
    ]
    for x1, y1, x2, y2, lbl in skips:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#e11d48", lw=1.8, linestyle="--"))
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.18, lbl, ha="center", va="center", fontsize=7.5, color="#be123c", fontweight="bold")

    # Fusion Adapter Connection Arrow
    ax.annotate("", xy=(11.5, 4.0), xytext=(9.5, 4.0),
                arrowprops=dict(facecolor="#9333ea", edgecolor="#9333ea", width=1.5, headwidth=5, linestyle=":"))

    plt.tight_layout()
    out_path = os.path.join(REPORTS_DIR, "proposed_unet_architecture_detail.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[SUCCESS] Saved detailed U-Net architecture diagram to: {out_path}")

if __name__ == "__main__":
    generate_system_block_diagram()
    generate_unet_detailed_diagram()

