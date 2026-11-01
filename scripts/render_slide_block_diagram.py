"""
Script to render the exact Mermaid flowchart as a publication-grade, slide-ready
high-resolution PNG image (both Light Mode and Dark Mode) for easy download and insertion.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs/reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def render_exact_flowchart(dark_mode=False):
    # Proportions: Wide banner format (ideal for slides)
    fig, ax = plt.subplots(figsize=(20, 7.5), dpi=300)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    if dark_mode:
        bg_canvas = "#0f172a"
        subgraph_bg = "#1e293b"
        subgraph_border = "#334155"
        subgraph_title_col = "#94a3b8"
        node_bg = "#0f172a"
        node_border = "#475569"
        text_col = "#f8fafc"
        arrow_col = "#38bdf8"
        skip_col = "#f43f5e"
    else:
        bg_canvas = "#ffffff"
        subgraph_bg = "#f8fafc"
        subgraph_border = "#cbd5e1"
        subgraph_title_col = "#475569"
        node_bg = "#ffffff"
        node_border = "#94a3b8"
        text_col = "#0f172a"
        arrow_col = "#2563eb"
        skip_col = "#e11d48"

    fig.patch.set_facecolor(bg_canvas)
    ax.set_facecolor(bg_canvas)

    # Helper: draw subgraph container
    def draw_subgraph(x, y, w, h, title):
        sg = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                                    facecolor=subgraph_bg, edgecolor=subgraph_border, linewidth=1.5, linestyle="--", zorder=1)
        ax.add_patch(sg)
        ax.text(x + 0.25, y + h - 0.25, title.upper(), ha="left", va="center",
                fontsize=8.5, fontweight="bold", color=subgraph_title_col, fontfamily="sans-serif", zorder=2)

    # Helper: draw inner node
    def draw_node(x, y, w, h, title, subtitle=None, highlight=False, border_c=None):
        b_col = border_c if border_c else (arrow_col if highlight else node_border)
        nb = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.1",
                                    facecolor=node_bg, edgecolor=b_col, linewidth=1.6 if highlight else 1.2, zorder=3)
        ax.add_patch(nb)
        
        if subtitle:
            ax.text(x + w/2, y + h/2 + 0.16, title, ha="center", va="center",
                    fontsize=8.2, fontweight="bold", color=text_col, fontfamily="sans-serif", zorder=4)
            ax.text(x + w/2, y + h/2 - 0.18, subtitle, ha="center", va="center",
                    fontsize=7.2, color="#64748b" if not dark_mode else "#94a3b8", fontfamily="sans-serif", zorder=4)
        else:
            ax.text(x + w/2, y + h/2, title, ha="center", va="center",
                    fontsize=8.2, fontweight="bold", color=text_col, fontfamily="sans-serif", zorder=4)

    # Helper: directional arrow
    def connect_arrow(x1, y1, x2, y2, bidirectional=False, col=None):
        color = col if col else arrow_col
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor=color, edgecolor=color, width=1.5, headwidth=6, headlength=6, shrink=0.05),
                    zorder=5)
        if bidirectional:
            ax.annotate("", xy=(x1, y1), xytext=(x2, y2),
                        arrowprops=dict(facecolor=color, edgecolor=color, width=1.5, headwidth=6, headlength=6, shrink=0.05),
                        zorder=5)

    # ================= SUBGRAPHS & NODES =================
    
    # 1. Ingestion
    draw_subgraph(0.4, 2.2, 4.3, 3.2, "1. Data Ingestion & Preprocessing")
    draw_node(0.6, 2.6, 1.1, 2.2, "UAV Aerial\nFlights", "WeedMap RGB\nTiles")
    draw_node(1.9, 2.6, 1.25, 2.2, "Boundary\nFiltering", "Patch Extractor\n(256x256)")
    draw_node(3.35, 2.6, 1.2, 2.2, "Augmentation\n& Weights", "[0.009, 1.90, 1.09]")
    
    connect_arrow(1.7, 3.7, 1.9, 3.7)
    connect_arrow(3.15, 3.7, 3.35, 3.7)

    # 2. Proposed AI Segmentation Engine
    draw_subgraph(5.0, 2.2, 5.8, 3.2, "2. Proposed AI Segmentation Engine")
    draw_node(5.2, 2.6, 1.25, 2.2, "ResNet18\nEncoder", "Layers 1 to 4\n(Pretrained)", highlight=True)
    draw_node(6.65, 2.6, 1.25, 2.2, "Bottleneck", "512 Channels\n+ MC Dropout")
    draw_node(8.1, 2.6, 1.25, 2.2, "Decoder with\nSkip Conns", "Cross-Scale\nFusion Adapter", highlight=True)
    draw_node(9.55, 2.6, 1.1, 2.2, "Softmax\nClassifier", "Soil, Crop,\nWeed (1x1 Conv)")

    connect_arrow(4.55, 3.7, 5.2, 3.7)
    connect_arrow(6.45, 3.7, 6.65, 3.7)
    connect_arrow(7.9, 3.7, 8.1, 3.7)
    connect_arrow(9.35, 3.7, 9.55, 3.7)

    # 3. Trustworthy AI Module (Top Right Branch)
    draw_subgraph(11.2, 4.0, 3.8, 3.0, "3. Trustworthy AI Module")
    draw_node(11.4, 5.5, 3.4, 1.1, "Grad-CAM Saliency", "Crop vs Weed Attributions (Layer 4)")
    draw_node(11.4, 4.3, 3.4, 1.1, "Monte Carlo Dropout (T=8)", "Pixel-Wise Predictive Entropy H(x)")

    # 4. Spatial Analytics & Decision Support (Bottom Right Branch)
    draw_subgraph(11.2, 0.4, 4.7, 3.2, "4. Spatial Analytics & Decision Support")
    draw_node(11.4, 0.8, 1.35, 2.2, "GSD Physical\nArea (m²)", "1.0 cm/px\nCalibration")
    draw_node(12.95, 0.8, 1.35, 2.2, "Infestation\nHeatmaps", "Low, Mod,\nHigh Risk Zones")
    draw_node(14.5, 0.8, 1.25, 2.2, "Prescription\nGuidance", "Targeted Spot\nSpraying Rules")

    connect_arrow(12.75, 1.9, 12.95, 1.9)
    connect_arrow(14.3, 1.9, 14.5, 1.9)

    # Branching from Classifier Head
    # Arrow to XAI (Top Branch)
    ax.annotate("", xy=(11.4, 6.05), xytext=(10.65, 3.9),
                arrowprops=dict(facecolor=arrow_col, edgecolor=arrow_col, width=1.5, headwidth=6, headlength=6, shrink=0.05),
                zorder=5)
    ax.annotate("", xy=(11.4, 4.85), xytext=(10.65, 3.7),
                arrowprops=dict(facecolor=arrow_col, edgecolor=arrow_col, width=1.5, headwidth=6, headlength=6, shrink=0.05),
                zorder=5)
    # Arrow to Spatial Analytics (Bottom Branch)
    ax.annotate("", xy=(11.4, 1.9), xytext=(10.65, 3.5),
                arrowprops=dict(facecolor=arrow_col, edgecolor=arrow_col, width=1.5, headwidth=6, headlength=6, shrink=0.05),
                zorder=5)

    # 5. Dashboard & Relational Auditing (Far Right)
    draw_subgraph(16.3, 0.4, 3.3, 3.2, "5. Dashboard & Relational Auditing")
    draw_node(16.5, 0.8, 1.35, 2.2, "Streamlit\nWeb UI", "7 Interactive\nTabs")
    draw_node(18.05, 0.8, 1.35, 2.2, "SQLite DB", "outputs/\nagrifusion.db")

    connect_arrow(15.75, 1.9, 16.5, 1.9)
    connect_arrow(17.85, 1.9, 18.05, 1.9, bidirectional=True)

    plt.tight_layout()
    filename = "system_architecture_block_diagram_slide_dark.png" if dark_mode else "system_architecture_block_diagram_slide.png"
    out_path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved {filename} to {out_path}")
    return out_path

if __name__ == "__main__":
    render_exact_flowchart(dark_mode=False)
    render_exact_flowchart(dark_mode=True)

