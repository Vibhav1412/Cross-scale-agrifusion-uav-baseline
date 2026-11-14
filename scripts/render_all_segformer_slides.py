"""
render_all_segformer_slides.py
Renders publication-grade, slide-ready 16:9 presentation slides in both Light and Dark modes
for the PES University Capstone Presentation (PW26_YBS_02).

Outputs generated:
1. Slide 12: System Pipeline Block Diagram (5 Stages with SegFormer)
2. Slide 13: Primary Model Architecture Detail (SegFormer MiT-B0 vs U-Net)
3. Slide 14: Architecture Highlights (SegFormer, ViT Attention, Cross-Scale Fusion)
4. Slide 15: Further Implementation Roadmap & Live Demo (Multi-Model Benchmark)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs/reports")
BRAIN_DIR = r"C:\Users\Admin\.gemini\antigravity\brain\b2741856-4f5d-40f0-858d-90c0bffb4502"
os.makedirs(REPORTS_DIR, exist_ok=True)

def draw_pes_header(ax, slide_no, title, subtitle, dark=False):
    # Slide Number & Tag
    ax.text(0.7, 9.45, f"PHASE-II REVIEW • {slide_no}", fontsize=11, fontweight="bold",
            color="#38bdf8" if dark else "#0284c7", fontfamily="sans-serif")
    
    # Main Header
    ax.text(0.7, 8.85, title, fontsize=24, fontweight="bold",
            color="#f8fafc" if dark else "#0f2942", fontfamily="serif")
    
    # Subtitle / Model Tag
    ax.text(0.7, 8.25, subtitle, fontsize=11, fontweight="bold",
            color="#10b981" if dark else "#059669", fontfamily="sans-serif")
    
    # Thin divider line
    ax.plot([0.7, 16.5], [8.55, 8.55], color="#334155" if dark else "#94a3b8", lw=1.2)
    
    # PES Logo Emblem placeholder (circular compass / emblem)
    center_x, center_y = 15.6, 9.0
    circle = patches.Circle((center_x, center_y), 0.42, facecolor="#f8fafc" if dark else "#ffffff",
                            edgecolor="#0284c7", lw=2, zorder=5)
    ax.add_patch(circle)
    # 8-point compass star in emblem
    star_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    for i, a in enumerate(star_angles):
        color = "#ea580c" if i % 2 == 0 else "#0284c7"
        ax.plot([center_x, center_x + 0.35 * (1 if a in [0, 45, 315] else (-1 if a in [135, 180, 225] else 0))],
                [center_y, center_y + 0.35 * (1 if a in [45, 90, 135] else (-1 if a in [225, 270, 315] else 0))],
                color=color, lw=1.6, zorder=6)
    ax.text(16.15, 9.15, "PES", fontsize=15, fontweight="black", color="#0f2942" if not dark else "#f8fafc", zorder=7)
    ax.text(16.15, 8.82, "UNIVERSITY", fontsize=7.5, fontweight="bold", color="#64748b" if not dark else "#94a3b8", zorder=7)

def draw_pes_footer(ax, slide_no, dark=False):
    ax.text(16.5, 0.4, f"PES UNIVERSITY  •  UE24AM320A  •  {slide_no}", ha="right", va="center",
            fontsize=9, color="#64748b" if not dark else "#94a3b8", fontfamily="sans-serif")

# ==============================================================================
# SLIDE 13: PRIMARY MODEL • SEGFORMER (MiT-B0 / VISION TRANSFORMER)
# ==============================================================================
def render_slide_13(dark=False):
    fig, ax = plt.subplots(figsize=(17.78, 10), dpi=300) # 16:9
    ax.set_xlim(0, 17.78)
    ax.set_ylim(0, 10)
    ax.axis("off")
    
    bg_canvas = "#0b0f19" if dark else "#ffffff"
    fig.patch.set_facecolor(bg_canvas)
    ax.set_facecolor(bg_canvas)
    
    draw_pes_header(ax, "13", "Proposed Architecture / Block Diagram",
                    "PRIMARY MODEL • SEGFORMER (MiT-B0 / VISION TRANSFORMER)", dark=dark)
    draw_pes_footer(ax, "13", dark=dark)
    
    # -------------------------------------------------------------
    # LEFT PANEL: Architectural Diagram (Dark Canvas)
    # -------------------------------------------------------------
    panel_bg = "#0f172a" if dark else "#111827"
    panel_border = "#38bdf8" if dark else "#1e293b"
    p_box = patches.FancyBboxPatch((0.7, 0.8), 9.6, 7.2, boxstyle="round,pad=0.1,rounding_size=0.15",
                                   facecolor=panel_bg, edgecolor=panel_border, linewidth=1.5, zorder=2)
    ax.add_patch(p_box)
    
    # Diagram Title inside panel
    ax.text(1.0, 7.7, "SegFormer (MiT-B0) + CrossScaleSatelliteFusionAdapter",
            fontsize=10.5, fontweight="bold", color="#38bdf8", zorder=3)
    
    # 1. Input Patch Box
    inp = patches.FancyBboxPatch((3.5, 7.0), 4.0, 0.5, boxstyle="round,pad=0.05,rounding_size=0.08",
                                facecolor="#1e293b", edgecolor="#64748b", linewidth=1.2, zorder=4)
    ax.add_patch(inp)
    ax.text(5.5, 7.25, "Input UAV RGB Patch  (3 × 256 × 256)", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#f8fafc", zorder=5)
    
    # 2. Four Encoder Stages (Left to right or downward cascade)
    stages = [
        ("Stage 1 (MiT-B0)", "Overlapped Patch Merge (4×4, s=4)\nEfficient Self-Attention (R=8) + Mix-FFN\nOutput: 32 × 64 × 64", 1.2, 5.7, 3.8, 0.95),
        ("Stage 2 (MiT-B0)", "Overlapped Patch Merge (2×2, s=2)\nEfficient Self-Attention (R=4) + Mix-FFN\nOutput: 64 × 32 × 32", 1.2, 4.45, 3.8, 0.95),
        ("Stage 3 (MiT-B0)", "Overlapped Patch Merge (2×2, s=2)\nEfficient Self-Attention (R=2) + Mix-FFN\nOutput: 160 × 16 × 16", 1.2, 3.2, 3.8, 0.95),
        ("Stage 4 (MiT-B0)", "Overlapped Patch Merge (2×2, s=2)\nEfficient Self-Attention (R=1) + Mix-FFN\nOutput: 256 × 8 × 8", 1.2, 1.95, 3.8, 0.95),
    ]
    
    for title, desc, sx, sy, sw, sh in stages:
        b = patches.FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0.05,rounding_size=0.08",
                                   facecolor="#1e293b", edgecolor="#0284c7", linewidth=1.4, zorder=4)
        ax.add_patch(b)
        ax.text(sx + sw/2, sy + sh - 0.22, title, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="#38bdf8", zorder=5)
        ax.text(sx + sw/2, sy + 0.32, desc, ha="center", va="center",
                fontsize=7.2, color="#cbd5e1", zorder=5)
        
    # Vertical arrows connecting stages
    ax.annotate("", xy=(3.1, 5.7), xytext=(3.1, 6.95),
                arrowprops=dict(facecolor="#0284c7", edgecolor="#0284c7", width=1.2, headwidth=4))
    for i in range(len(stages)-1):
        y_from = stages[i][3]
        y_to = stages[i+1][3] + stages[i+1][5]
        ax.annotate("", xy=(3.1, y_to), xytext=(3.1, y_from),
                    arrowprops=dict(facecolor="#0284c7", edgecolor="#0284c7", width=1.2, headwidth=4))

    # 3. All-MLP Decoder Column (Right side of inner canvas)
    # Linear Projections & Multi-Level Fusion
    mlp_proj = patches.FancyBboxPatch((5.8, 4.8), 4.2, 1.35, boxstyle="round,pad=0.05,rounding_size=0.08",
                                      facecolor="#1e293b", edgecolor="#10b981", linewidth=1.4, zorder=4)
    ax.add_patch(mlp_proj)
    ax.text(7.9, 5.85, "All-MLP Decoder Multi-Level Fusion", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#34d399", zorder=5)
    ax.text(7.9, 5.3, "Linear MLP Projections: C_i -> C = 256\nBilinear 4× Upsample to H/4 × W/4 (64×64)\nConcat (1024 ch) -> Linear Fusion (256 ch)",
            ha="center", va="center", fontsize=7.2, color="#cbd5e1", zorder=5)

    # Connections from each stage to Decoder (Multi-level skip flow)
    for title, desc, sx, sy, sw, sh in stages:
        out_y = sy + sh/2
        ax.annotate("", xy=(5.8, 5.3), xytext=(sx + sw, out_y),
                    arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=1.2, linestyle="--"))

    # Phase-II Sentinel-2 Satellite Cross-Attention Bridge
    sat_box = patches.FancyBboxPatch((5.8, 3.2), 4.2, 1.3, boxstyle="round,pad=0.05,rounding_size=0.08",
                                     facecolor="#2e1065", edgecolor="#a855f7", linewidth=1.6, zorder=4)
    ax.add_patch(sat_box)
    ax.text(7.9, 4.15, "PHASE-II SATELLITE CROSS-ATTENTION BRIDGE", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color="#c084fc", zorder=5)
    ax.text(7.9, 3.62, "Sentinel-2 Multi-Spectral Tokens (B2, B3, B4, B8, B8A)\nQ = UAV Tokens,  K, V = Satellite Tokens\nMulti-Head Cross-Attention + Gated Residual (gamma=0.0)",
            ha="center", va="center", fontsize=7.0, color="#e9d5ff", zorder=5)

    # Arrow from MLP fusion down to Satellite Cross-Attention
    ax.annotate("", xy=(7.9, 4.5), xytext=(7.9, 4.8),
                arrowprops=dict(facecolor="#a855f7", edgecolor="#a855f7", width=1.4, headwidth=4))

    # MC Dropout & Final Classification
    cls_box = patches.FancyBboxPatch((5.8, 1.3), 4.2, 1.6, boxstyle="round,pad=0.05,rounding_size=0.08",
                                     facecolor="#1e293b", edgecolor="#f59e0b", linewidth=1.4, zorder=4)
    ax.add_patch(cls_box)
    ax.text(7.9, 2.55, "Uncertainty & Classification Head", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#fbbf24", zorder=5)
    ax.text(7.9, 2.15, "Monte Carlo Dropout (p = 0.2, T = 8 passes)\n1×1 Conv Classifier (256 -> 3 classes)\nBilinear Upsample -> 3 × 256 × 256",
            ha="center", va="center", fontsize=7.2, color="#cbd5e1", zorder=5)
    ax.text(7.9, 1.55, "Output: [0] Soil/BG  |  [1] Crop  |  [2] Weed", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color="#4ade80", zorder=5)

    # Arrow down from Satellite Adapter to Cls
    ax.annotate("", xy=(7.9, 2.9), xytext=(7.9, 3.2),
                arrowprops=dict(facecolor="#f59e0b", edgecolor="#f59e0b", width=1.4, headwidth=4))

    # -------------------------------------------------------------
    # RIGHT PANEL: Specification Cards
    # -------------------------------------------------------------
    rx = 10.7
    rw = 6.4
    cards = [
        ("INPUT", "3 × 256 × 256 UAV RGB Patch (GSD: 1.0 cm/px)",
         "#eff6ff" if not dark else "#1e293b", "#3b82f6", "#1d4ed8" if not dark else "#60a5fa"),
        ("ENCODER (VISION TRANSFORMER)", "Hierarchical Mix-Transformer (MiT-B0), 4 stages with Overlapped Patch Merging & Spatial-Reduction Self-Attention (R=8, 4, 2, 1)",
         "#eff6ff" if not dark else "#1e293b", "#0284c7", "#0369a1" if not dark else "#38bdf8"),
        ("POSITIONAL-FREE DESIGN", "Mix-FFN with 3×3 depthwise convolutions removes fixed positional embeddings -> scale-invariant to flight altitudes & patch resolutions",
         "#ecfdf5" if not dark else "#1e293b", "#10b981", "#047857" if not dark else "#34d399"),
        ("DECODER", "All-MLP lightweight decoder: multi-scale feature unification into 256 channels + Monte Carlo Dropout (p=0.2) for predictive entropy",
         "#eff6ff" if not dark else "#1e293b", "#0284c7", "#0369a1" if not dark else "#38bdf8"),
        ("EFFICIENCY ADVANTAGE", "3.98M params (vs 14.33M U-Net, 72% reduction) | 13.17 ms CPU latency (2.7× faster) | 15.2 MB memory footprint",
         "#fef2f2" if not dark else "#1e293b", "#ef4444", "#b91c1c" if not dark else "#f87171"),
        ("PHASE-II SATELLITE BRIDGE", "Multi-Head Cross-Attention Adapter fusing Sentinel-2 10m/20m multi-spectral tokens (B2, B3, B4, B8, B8A) into UAV feature maps",
         "#fff7ed" if not dark else "#2a1b14", "#f97316", "#c2410c" if not dark else "#fb923c"),
        ("OUTPUT", "3 × 256 × 256 Pixel Map: Class 0 (Soil), Class 1 (Crop), Class 2 (Weed)",
         "#f0fdf4" if not dark else "#1e293b", "#22c55e", "#15803d" if not dark else "#4ade80"),
    ]
    
    card_y = 7.15
    for tag, body, bg_c, border_c, tag_c in cards:
        ch = 0.88 if len(body) > 75 else 0.72
        cb = patches.FancyBboxPatch((rx, card_y - ch), rw, ch, boxstyle="round,pad=0.06,rounding_size=0.1",
                                    facecolor=bg_c, edgecolor=border_c, linewidth=1.5, zorder=3)
        ax.add_patch(cb)
        ax.text(rx + 0.25, card_y - 0.22, tag, fontsize=8.5, fontweight="bold", color=tag_c, zorder=4)
        ax.text(rx + 0.25, card_y - 0.52, body, fontsize=8.0, color="#1e293b" if not dark else "#cbd5e1",
                wrap=True, zorder=4)
        card_y -= (ch + 0.18)

    plt.tight_layout()
    suffix = "_dark" if dark else ""
    out_path = os.path.join(REPORTS_DIR, f"segformer_architecture_detail_slide{suffix}.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    # Also save copy to brain
    brain_out = os.path.join(BRAIN_DIR, f"segformer_architecture_detail_slide{suffix}.png")
    plt.savefig(brain_out, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[SUCCESS] Saved SegFormer Architecture Detail Slide to: {out_path}")

# ==============================================================================
# SLIDE 12: SYSTEM PIPELINE • 5 STAGES (WITH SEGFORMER)
# ==============================================================================
def render_slide_12(dark=False):
    fig, ax = plt.subplots(figsize=(17.78, 10), dpi=300)
    ax.set_xlim(0, 17.78)
    ax.set_ylim(0, 10)
    ax.axis("off")
    
    bg_canvas = "#0b0f19" if dark else "#ffffff"
    fig.patch.set_facecolor(bg_canvas)
    ax.set_facecolor(bg_canvas)
    
    draw_pes_header(ax, "12", "Proposed Architecture / Block Diagram",
                    "SYSTEM PIPELINE  •  5 STAGES", dark=dark)
    draw_pes_footer(ax, "12", dark=dark)
    
    # 5 Stages Cards in Upper Center Area
    stages_info = [
        ("1. DATA INGESTION & PREPROCESSING", [
            ("UAV Aerial Flights", "WeedMap RGB Tiles"),
            ("Boundary Filtering", "Patch Extractor (256×256)"),
            ("Augmentation & Weights", "[0.009, 1.90, 1.09]")
        ], 0.7, 4.3, 3.6, 3.6, "#0284c7"),
        
        ("2. PROPOSED AI SEGMENTATION ENGINE", [
            ("Mix-Transformer (MiT-B0)", "Hierarchical 4-Stage ViT"),
            ("Overlapped Patch Merging", "Efficient Self-Attention (R=8..1)"),
            ("All-MLP Decoder", "Cross-Scale Satellite Adapter"),
            ("Softmax Pixel Classifier", "Soil, Crop, Weed (1×1 Conv)")
        ], 4.5, 4.3, 4.9, 3.6, "#10b981"),
        
        ("3. TRUSTWORTHY AI MODULE", [
            ("Class-Masked Grad-CAM", "Crop vs Weed Attributions (linear_fuse)"),
            ("Monte Carlo Dropout (T=8)", "Predictive Entropy H(x)")
        ], 9.6, 5.8, 3.8, 2.1, "#6366f1"),
        
        ("4. SPATIAL ANALYTICS & DECISION SUPPORT", [
            ("GSD Physical Area (m²)", "1.0 cm/px Calibration"),
            ("Infestation Heatmaps", "Low, Mod, High Risk"),
            ("Prescription Guidance", "Targeted Spot-Spraying")
        ], 9.6, 3.4, 3.8, 2.2, "#f59e0b"),
        
        ("5. DASHBOARD & RELATIONAL AUDITING", [
            ("Streamlit Web UI", "7 Interactive Tabs"),
            ("SQLite DB", "outputs/agrifusion.db")
        ], 13.6, 4.3, 3.5, 3.6, "#ec4899"),
    ]
    
    for s_title, sub_nodes, sx, sy, sw, sh, color in stages_info:
        # Stage Outer Container
        sg = patches.FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0.08,rounding_size=0.15",
                                    facecolor="#1e293b" if dark else "#f8fafc",
                                    edgecolor=color, linewidth=1.8, linestyle="--", zorder=2)
        ax.add_patch(sg)
        ax.text(sx + 0.2, sy + sh - 0.28, s_title, fontsize=8.5, fontweight="bold",
                color=color, fontfamily="sans-serif", zorder=3)
        
        # Inner node boxes
        ny = sy + sh - 0.85
        node_h = (sh - 1.1) / len(sub_nodes)
        for n_head, n_sub in sub_nodes:
            nb = patches.FancyBboxPatch((sx + 0.2, ny - node_h + 0.1), sw - 0.4, node_h - 0.12,
                                        boxstyle="round,pad=0.05,rounding_size=0.08",
                                        facecolor="#0f172a" if dark else "#ffffff",
                                        edgecolor="#475569" if dark else "#cbd5e1",
                                        linewidth=1.2, zorder=3)
            ax.add_patch(nb)
            ax.text(sx + sw/2, ny - node_h/2 + 0.12, n_head, ha="center", va="center",
                    fontsize=8.0, fontweight="bold", color="#f8fafc" if dark else "#0f172a", zorder=4)
            ax.text(sx + sw/2, ny - node_h/2 - 0.15, n_sub, ha="center", va="center",
                    fontsize=7.0, color="#94a3b8" if dark else "#64748b", zorder=4)
            ny -= node_h

    # Connective arrows between stages
    ax.annotate("", xy=(4.5, 6.1), xytext=(4.3, 6.1),
                arrowprops=dict(facecolor="#0284c7", edgecolor="#0284c7", width=1.5, headwidth=5))
    ax.annotate("", xy=(9.6, 6.85), xytext=(9.4, 6.85),
                arrowprops=dict(facecolor="#10b981", edgecolor="#10b981", width=1.5, headwidth=5))
    ax.annotate("", xy=(9.6, 4.5), xytext=(9.4, 4.5),
                arrowprops=dict(facecolor="#10b981", edgecolor="#10b981", width=1.5, headwidth=5))
    ax.annotate("", xy=(13.6, 5.0), xytext=(13.4, 5.0),
                arrowprops=dict(facecolor="#f59e0b", edgecolor="#f59e0b", width=1.5, headwidth=5))

    # Bottom Stage Indicator Badges (Matching slide 12 in photo)
    bottom_badges = [
        ("1 Data Ingestion & Prep", 0.7, 3.1, "#f1f5f9" if not dark else "#1e293b", "#64748b" if not dark else "#cbd5e1"),
        ("2 AI Segmentation Engine (SegFormer)", 4.0, 4.8, "#0f2942" if not dark else "#0284c7", "#ffffff"),
        ("3 Trustworthy AI", 9.0, 2.5, "#f1f5f9" if not dark else "#1e293b", "#64748b" if not dark else "#cbd5e1"),
        ("4 Spatial Analytics", 11.7, 2.6, "#f1f5f9" if not dark else "#1e293b", "#64748b" if not dark else "#cbd5e1"),
        ("5 Dashboard & Auditing", 14.5, 2.6, "#f1f5f9" if not dark else "#1e293b", "#64748b" if not dark else "#cbd5e1"),
    ]
    for b_title, bx, bw, b_bg, b_fg in bottom_badges:
        bb = patches.FancyBboxPatch((bx, 1.2), bw, 0.65, boxstyle="round,pad=0.08,rounding_size=0.1",
                                    facecolor=b_bg, edgecolor="none", zorder=3)
        ax.add_patch(bb)
        ax.text(bx + bw/2, 1.52, b_title, ha="center", va="center",
                fontsize=9.0, fontweight="bold", color=b_fg, zorder=4)

    plt.tight_layout()
    suffix = "_dark" if dark else ""
    out_path = os.path.join(REPORTS_DIR, f"system_architecture_block_diagram_slide{suffix}.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    brain_out = os.path.join(BRAIN_DIR, f"system_architecture_block_diagram_slide{suffix}.png")
    plt.savefig(brain_out, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[SUCCESS] Saved Pipeline 5-Stages Slide to: {out_path}")

# ==============================================================================
# SLIDE 14: ARCHITECTURE HIGHLIGHTS (WITH SEGFORMER)
# ==============================================================================
def render_slide_14(dark=False):
    fig, ax = plt.subplots(figsize=(17.78, 10), dpi=300)
    ax.set_xlim(0, 17.78)
    ax.set_ylim(0, 10)
    ax.axis("off")
    
    bg_canvas = "#0b0f19" if dark else "#ffffff"
    fig.patch.set_facecolor(bg_canvas)
    ax.set_facecolor(bg_canvas)
    
    draw_pes_header(ax, "14", "Proposed Architecture / Block Diagram",
                    "ARCHITECTURE HIGHLIGHTS", dark=dark)
    draw_pes_footer(ax, "14", dark=dark)
    
    highlights = [
        ("End-to-End Pipeline Modularization",
         "Divides the workflow into 5 cohesive stages: (1) Patch Ingestion & Preprocessing, (2) Deep Semantic Segmentation, (3) Trustworthy AI (XAI & Uncertainty), (4) Geospatial Field Analytics, and (5) Web Deployment & Relational Auditing.",
         "#0284c7", "grid"),
        
        ("Proposed Primary Model (SegFormer MiT-B0)",
         "Employs a modern Vision Transformer (NeurIPS 2021) with hierarchical Mix-Transformer encoder, overlapped patch merging, All-MLP decoder, and zero positional embeddings. Delivers 3.98M parameters (72% reduction vs U-Net) and 13.17 ms/patch CPU latency (2.7× faster).",
         "#10b981", "brain"),
         
        ("Phase-II Cross-Scale Satellite Fusion Bridge",
         "Integrated CrossScaleSatelliteFusionAdapter inside the decoder features multi-head cross-attention and gated residual addition, allowing seamless fusion of 10m Sentinel-2 multi-spectral tokens (B2, B3, B4, B8, B8A) without altering base network weights.",
         "#f97316", "satellite"),
         
        ("Explainable & Safe Agronomic Diagnostics",
         "Combines class-masked Grad-CAM (isolating subtle crop leaf vs weed cluster attributions without soil background gradient contamination) and Monte Carlo Dropout (T = 8 passes) to flag ambiguous boundaries before generating spraying maps.",
         "#6366f1", "eye"),
         
        ("Relational Storage & Compliance",
         "Persists all inference telemetry, infestation heatmaps, and physical area metrics (m²) into an SQLite database (agrifusion.db) with instantaneous audit export.",
         "#0f766e", "db")
    ]
    
    hy = 7.3
    hw = 16.0
    for title, desc, icon_col, icon_type in highlights:
        hh = 1.05
        # Main card
        card_bg = "#1e293b" if dark else "#f8fafc"
        border_col = icon_col if "Phase-II" in title else ("#334155" if dark else "#e2e8f0")
        card = patches.FancyBboxPatch((0.9, hy - hh), hw, hh, boxstyle="round,pad=0.08,rounding_size=0.12",
                                      facecolor=card_bg, edgecolor=border_col,
                                      linewidth=2.0 if "Phase-II" in title else 1.2, zorder=2)
        ax.add_patch(card)
        
        # Circular Icon Badge
        circle = patches.Circle((1.55, hy - hh/2), 0.32, facecolor=icon_col, edgecolor="none", zorder=3)
        ax.add_patch(circle)
        
        # Text
        ax.text(2.1, hy - 0.32, title, fontsize=11, fontweight="bold",
                color="#f8fafc" if dark else "#0f2942", zorder=3)
        ax.text(2.1, hy - 0.72, desc, fontsize=8.8, color="#94a3b8" if dark else "#334155",
                wrap=True, zorder=3)
        
        hy -= (hh + 0.22)
        
    plt.tight_layout()
    suffix = "_dark" if dark else ""
    out_path = os.path.join(REPORTS_DIR, f"architecture_highlights_slide{suffix}.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    brain_out = os.path.join(BRAIN_DIR, f"architecture_highlights_slide{suffix}.png")
    plt.savefig(brain_out, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[SUCCESS] Saved Architecture Highlights Slide to: {out_path}")

# ==============================================================================
# SLIDE 15: FURTHER IMPLEMENTATION & LIVE SYSTEM DEMO
# ==============================================================================
def render_slide_15(dark=False):
    fig, ax = plt.subplots(figsize=(17.78, 10), dpi=300)
    ax.set_xlim(0, 17.78)
    ax.set_ylim(0, 10)
    ax.axis("off")
    
    bg_canvas = "#0b0f19" if dark else "#ffffff"
    fig.patch.set_facecolor(bg_canvas)
    ax.set_facecolor(bg_canvas)
    
    draw_pes_header(ax, "15", "Further Implementation & Live System Demonstration",
                    "WORK DONE TILL NOW  &  PHASE-II SCOPE", dark=dark)
    draw_pes_footer(ax, "15", dark=dark)
    
    # LEFT BOX: Part 1 Live Demonstration
    box1 = patches.FancyBboxPatch((0.7, 1.4), 7.6, 6.7, boxstyle="round,pad=0.08,rounding_size=0.15",
                                  facecolor="#0f2937" if dark else "#ecfdf5",
                                  edgecolor="#10b981", linewidth=1.8, zorder=2)
    ax.add_patch(box1)
    ax.text(1.0, 7.65, "PHASE-I  •  COMPLETED & LIVE DEMO", fontsize=8.5, fontweight="bold",
            color="#059669" if not dark else "#34d399", zorder=3)
    ax.text(1.0, 7.25, "Part 1: Live Demonstration Highlights (Work Done Till Now)",
            fontsize=11.5, fontweight="bold", color="#0f2942" if not dark else "#f8fafc", zorder=3)
    
    p1_bullets = [
        "End-to-End UAV Inference Engine: Real-time sliding-window patch inference (256×256, stride 192) on unseen field orthomosaics with whole-field mask reconstruction.",
        "Empirical Multi-Model Benchmark: Modern SegFormer (MiT-B0) adopted as new primary model (3.98M params, 13.17 ms latency) benchmarked against ResNet18 U-Net (14.33M), DeepLabV3+ (26.74M), and Basic CNN (1.89M).",
        "Trustworthy AI Diagnostic Suite: Live visual inspection of Class-Masked Grad-CAM leaf attributions and Monte Carlo Dropout (T = 8) predictive boundary entropy.",
        "Field Spatial Analytics & Relational Store: Dynamic N×N weed infestation risk zoning (Low, Moderate, High) backed by an SQLite audit database (agrifusion.db) with CSV export."
    ]
    by = 6.6
    for b in p1_bullets:
        ax.text(1.1, by, "•", fontsize=11, fontweight="bold", color="#10b981", zorder=3)
        ax.text(1.35, by, b, fontsize=8.6, color="#1e293b" if not dark else "#cbd5e1", wrap=True, zorder=3)
        by -= 1.15
        
    # Badges at bottom of Left Box
    badges = [
        ("SegFormer", "Primary Model"),
        ("3.98 M", "Parameters (72% ↓)"),
        ("13.17 ms", "CPU Latency (2.7× ↑)"),
        ("Flight 003", "Held-out Test Field")
    ]
    b_x = 1.0
    for val, lbl in badges:
        bb = patches.FancyBboxPatch((b_x, 1.6), 1.65, 0.9, boxstyle="round,pad=0.04,rounding_size=0.08",
                                    facecolor="#ffffff" if not dark else "#1e293b",
                                    edgecolor="#10b981", linewidth=1.2, zorder=3)
        ax.add_patch(bb)
        ax.text(b_x + 0.82, 2.15, val, ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="#047857" if not dark else "#34d399", zorder=4)
        ax.text(b_x + 0.82, 1.82, lbl, ha="center", va="center", fontsize=6.8,
                color="#64748b" if not dark else "#94a3b8", zorder=4)
        b_x += 1.85

    # Center Flow Arrow
    ax.annotate("", xy=(9.0, 4.7), xytext=(8.5, 4.7),
                arrowprops=dict(facecolor="#38bdf8", edgecolor="#38bdf8", width=3, headwidth=8))

    # RIGHT BOX: Part 2 Further Implementation Roadmap
    box2 = patches.FancyBboxPatch((9.2, 1.4), 7.8, 6.7, boxstyle="round,pad=0.08,rounding_size=0.15",
                                  facecolor="#2a1b14" if dark else "#fff7ed",
                                  edgecolor="#f97316", linewidth=1.8, zorder=2)
    ax.add_patch(box2)
    ax.text(9.5, 7.65, "PHASE-II  •  FURTHER IMPLEMENTATION", fontsize=8.5, fontweight="bold",
            color="#c2410c" if not dark else "#fb923c", zorder=3)
    ax.text(9.5, 7.25, "Part 2: Further Implementation Roadmap (Phase-II Scope)",
            fontsize=11.5, fontweight="bold", color="#0f2942" if not dark else "#f8fafc", zorder=3)
            
    p2_bullets = [
        "Sentinel-2 Multi-Spectral Co-Registration: Acquire and align 10 m GSD Sentinel-2 surface reflectance bands (B2, B3, B4, B8, B8A) matching the Rheinbach and Eschikon UAV field footprints.",
        "Activation of Cross-Scale Fusion Adapter: Connect multi-spectral satellite embeddings into the pre-built CrossScaleSatelliteFusionAdapter via multi-head cross-attention tokens.",
        "Multi-Dataset Cross-Domain Transfer: Ingest and evaluate zero-shot domain transfer on University of Bonn PhenoBench (1024×1024 ultra-crisp RGB + pixel-level weed/crop masks).",
        "Multi-Temporal Phenology Tracking: Track seasonal vegetation index time series (NDVI) alongside episodic UAV flights to monitor crop growth dynamics and weed resurgence.",
        "Edge Drone Companion Optimization: Quantize the fused SegFormer model via ONNX/TensorRT (FP16) for real-time onboard inference on embedded robotic sprayers (NVIDIA Jetson Orin Nano)."
    ]
    by = 6.6
    for b in p2_bullets:
        ax.text(9.6, by, "•", fontsize=11, fontweight="bold", color="#f97316", zorder=3)
        ax.text(9.85, by, b, fontsize=8.6, color="#1e293b" if not dark else "#cbd5e1", wrap=True, zorder=3)
        by -= 0.95

    plt.tight_layout()
    suffix = "_dark" if dark else ""
    out_path = os.path.join(REPORTS_DIR, f"further_implementation_roadmap_slide{suffix}.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    brain_out = os.path.join(BRAIN_DIR, f"further_implementation_roadmap_slide{suffix}.png")
    plt.savefig(brain_out, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[SUCCESS] Saved Further Implementation Roadmap Slide to: {out_path}")

if __name__ == "__main__":
    for dark in [False, True]:
        render_slide_12(dark=dark)
        render_slide_13(dark=dark)
        render_slide_14(dark=dark)
        render_slide_15(dark=dark)
