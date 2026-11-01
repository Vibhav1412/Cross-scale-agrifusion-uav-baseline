"""
Script to generate the 'Further Implementation & Project Roadmap' slide diagram
(Both Light and Dark mode) saved directly to Downloads and reports directory.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs/reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def render_further_implementation_diagram(dark_mode=False):
    fig, ax = plt.subplots(figsize=(19, 8.5), dpi=300)
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    if dark_mode:
        bg_canvas = "#0f172a"
        card_bg_phase1 = "#1e293b"
        card_border_phase1 = "#059669"
        card_bg_phase2 = "#1e1b4b"
        card_border_phase2 = "#6366f1"
        header_p1 = "#34d399"
        header_p2 = "#818cf8"
        text_primary = "#f8fafc"
        text_secondary = "#94a3b8"
        node_bg = "#0f172a"
        node_border = "#334155"
        arrow_color = "#38bdf8"
    else:
        bg_canvas = "#ffffff"
        card_bg_phase1 = "#f0fdf4"
        card_border_phase1 = "#16a34a"
        card_bg_phase2 = "#eef2ff"
        card_border_phase2 = "#4f46e5"
        header_p1 = "#15803d"
        header_p2 = "#4338ca"
        text_primary = "#0f172a"
        text_secondary = "#475569"
        node_bg = "#ffffff"
        node_border = "#cbd5e1"
        arrow_color = "#2563eb"

    fig.patch.set_facecolor(bg_canvas)
    ax.set_facecolor(bg_canvas)

    # Main Slide Title
    ax.text(9.5, 7.9, "CURRENT MILESTONE DEMO & FURTHER IMPLEMENTATION ROADMAP", 
            ha="center", va="center", fontsize=15, fontweight="bold", color=text_primary, fontfamily="sans-serif")
    ax.text(9.5, 7.5, "PES University Capstone PW26_YBS_02 | Guide: Prof. Yoga B S", 
            ha="center", va="center", fontsize=10, color=text_secondary, fontfamily="sans-serif")

    # Helper: Draw Phase Container
    def draw_phase_card(x, y, w, h, title, status_badge, badge_bg, badge_fg, bg_col, border_col, title_col):
        # Container
        card = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.18",
                                      facecolor=bg_col, edgecolor=border_col, linewidth=2, zorder=1)
        ax.add_patch(card)
        # Title
        ax.text(x + 0.35, y + h - 0.45, title, ha="left", va="center", fontsize=12, fontweight="bold", color=title_col, zorder=2)
        # Badge
        badge = patches.FancyBboxPatch((x + w - 2.8, y + h - 0.65), 2.5, 0.45, boxstyle="round,pad=0.04,rounding_size=0.1",
                                      facecolor=badge_bg, edgecolor="none", zorder=2)
        ax.add_patch(badge)
        ax.text(x + w - 1.55, y + h - 0.42, status_badge, ha="center", va="center", fontsize=8, fontweight="bold", color=badge_fg, zorder=3)

    # Helper: Draw Feature Sub-card
    def draw_feature_box(x, y, w, h, title, points, tag=None):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                     facecolor=node_bg, edgecolor=node_border, linewidth=1.2, zorder=2)
        ax.add_patch(box)
        
        ax.text(x + 0.25, y + h - 0.32, title, ha="left", va="center", fontsize=9.2, fontweight="bold", color=text_primary, zorder=3)
        if tag:
            ax.text(x + w - 0.25, y + h - 0.32, tag, ha="right", va="center", fontsize=7.2, fontweight="bold", color=arrow_color, zorder=3)
            
        start_y = y + h - 0.68
        for i, pt in enumerate(points):
            ax.text(x + 0.25, start_y - (i * 0.36), f"• {pt}", ha="left", va="center", fontsize=7.8, color=text_secondary, zorder=3)

    # ================= LEFT PANEL: PHASE 1 (DEMO TODAY) =================
    draw_phase_card(0.6, 0.6, 8.2, 6.5,
                    "PHASE-I: UAV BASELINE SYSTEM", "[ COMPLETED & LIVE DEMO ]",
                    "#dcfce7" if not dark_mode else "#064e3b", "#15803d" if not dark_mode else "#6ee7b7",
                    card_bg_phase1, card_border_phase1, header_p1)

    # Feature 1.1: Multi-Model Segmentation
    draw_feature_box(0.9, 4.7, 7.6, 1.8,
                     "1. Multi-Model Semantic Segmentation",
                     ["ResNet18 U-Net (50.96% mIoU, 89.00% Recall on unseen Flight 003)",
                      "Benchmarked against DeepLabV3+ (51.70% mIoU) & Basic CNN (42.27%)",
                      "Inverse frequency weighting handling 98.8% soil class imbalance"],
                     tag="EVALUATED")

    # Feature 1.2: Trustworthy AI
    draw_feature_box(0.9, 2.7, 7.6, 1.8,
                     "2. Trustworthy AI (XAI & Uncertainty)",
                     ["Class-Specific Grad-CAM attributions for Crop vs Weed",
                      "Monte Carlo Dropout (T=8) pixel-wise predictive entropy H(x)",
                      "High-uncertainty boundary mapping to avoid herbicide misapplication"],
                     tag="VERIFIED")

    # Feature 1.3: Spatial Analytics & UI
    draw_feature_box(0.9, 0.85, 7.6, 1.65,
                     "3. Spatial Analytics & Relational Database",
                     ["GSD (1 cm/px) physical area mapping & NxN weed infestation heatmaps",
                      "Full 7-tab Streamlit Dashboard with real-time sliding-window inference",
                      "Persistent SQLite store (outputs/agrifusion.db) with CSV audit export"],
                     tag="DEPLOYED")

    # ================= TRANSITION ARROW (CENTER) =================
    ax.annotate("", xy=(9.7, 3.85), xytext=(8.9, 3.85),
                arrowprops=dict(facecolor=arrow_color, edgecolor=arrow_color, width=3.0, headwidth=10, headlength=10, shrink=0.05),
                zorder=4)
    ax.text(9.3, 4.3, "SCALE-GAP\nBRIDGE", ha="center", va="center", fontsize=8, fontweight="bold", color=arrow_color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_canvas, edgecolor=arrow_color, alpha=0.9), zorder=5)

    # ================= RIGHT PANEL: PHASE 2 (FURTHER IMPLEMENTATION) =================
    draw_phase_card(9.9, 0.6, 8.5, 6.5,
                    "PHASE-II: CROSS-SCALE FUSION & SCALING", "[ FURTHER IMPLEMENTATION ]",
                    "#ede9fe" if not dark_mode else "#312e81", "#4338ca" if not dark_mode else "#c7d2fe",
                    card_bg_phase2, card_border_phase2, header_p2)

    # Feature 2.1: Sentinel-2 Integration
    draw_feature_box(10.2, 5.0, 7.9, 1.5,
                     "1. Sentinel-2 Satellite Multi-Spectral Co-Registration",
                     ["Query Sentinel-2 L2A surface reflectance (B2, B3, B4, B8, B8A at 10m GSD)",
                      "UTM geospatial co-registration with Rheinbach & Eschikon UAV field footprints"],
                     tag="ROADMAP 1")

    # Feature 2.2: Cross-Scale Fusion Adapter
    draw_feature_box(10.2, 3.35, 7.9, 1.5,
                     "2. Cross-Scale Feature Fusion Engine",
                     ["Activate built-in CrossScaleFusionAdapter in U-Net decoder",
                      "1x1 Conv spectral projection + residual fusion: Fused = UAV + Proj(Sat)",
                      "Evaluate cross-scale context for enhanced vegetative boundary detection"],
                     tag="ROADMAP 2")

    # Feature 2.3: PhenoBench Transfer & Temporal Scaling
    draw_feature_box(10.2, 1.7, 7.9, 1.5,
                     "3. Multi-Dataset Cross-Domain Transfer (PhenoBench)",
                     ["Zero-shot & fine-tuned transfer onto Univ. of Bonn PhenoBench (1024x1024)",
                      "Domain shift assessment across diverse agricultural farm ecologies"],
                     tag="ROADMAP 3")

    # Feature 2.4: Edge Deployment
    draw_feature_box(10.2, 0.85, 7.9, 0.75,
                     "4. Edge Drone Companion Deployment (TensorRT / Jetson)",
                     ["ONNX / TensorRT FP16 quantization for real-time onboard UAV weeding robots."],
                     tag="ROADMAP 4")

    plt.tight_layout()
    filename = "further_implementation_roadmap_slide_dark.png" if dark_mode else "further_implementation_roadmap_slide.png"
    out_path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved {filename} to {out_path}")
    return out_path

if __name__ == "__main__":
    render_further_implementation_diagram(dark_mode=False)
    render_further_implementation_diagram(dark_mode=True)

