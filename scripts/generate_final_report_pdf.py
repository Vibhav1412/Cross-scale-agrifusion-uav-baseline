"""
generate_final_report_pdf.py
Generates an exhaustive, publication-grade academic PDF report documenting the entire
Cross-Scale UAV-Satellite Data Fusion Capstone Project (PES University PW26_YBS_02).

Includes:
- Full system architecture & motivation
- Comprehensive 5-model benchmark suite (Test mIoU, Dice, Crop/Weed IoU, latency, footprint)
- SEPARATE Architecture & Computational Specifications Table (from Streamlit UI)
- SEPARATE Individual Model Training Dynamics & Loss Convergence Graphs (SegFormer, ResNet50, ResNet18, DeepLabV3+, Basic CNN)
- Leakage-free dataset splitting protocol (WeedMap)
- Primary SegFormer (MiT-B0) design & Phase-II Satellite Fusion Adapter
- Class-Masked Target Grad-CAM & Monte Carlo Dropout Uncertainty
- 4-Tier Variable Rate Application (VRA) Prescription Engine & Economics
- Commercial drone mission exports (DJI Agras CSV, GeoJSON, compliance cert)
- PhenoBench (CVPR 2023) benchmark integration
- Defense cheat sheet for review panel
"""

import os
import sys
import shutil
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
    KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PDF = os.path.join(PROJECT_ROOT, "outputs", "reports", "FINAL_CAPSTONE_RESEARCH_REPORT.pdf")
DOCS_PDF = os.path.join(PROJECT_ROOT, "docs", "FINAL_CAPSTONE_RESEARCH_REPORT.pdf")
ARTIFACT_DIR = r"C:\Users\Admin\.gemini\antigravity\brain\b2741856-4f5d-40f0-858d-90c0bffb4502"

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and draw total page count
    along with running header and footer on every page except the cover.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber > 1:
            self.saveState()
            self.setFont("Helvetica-Bold", 7.5)
            self.setFillColor(colors.HexColor("#475569"))
            
            # Running Header
            self.drawString(45, 752, "CROSS-SCALE UAV–SATELLITE DATA FUSION")
            self.setFont("Helvetica", 7.5)
            self.setFillColor(colors.HexColor("#64748b"))
            self.drawRightString(567, 752, "PES UNIVERSITY | CAPSTONE PW26_YBS_02")
            
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.6)
            self.line(45, 745, 567, 745)
            
            # Running Footer
            self.line(45, 42, 567, 42)
            self.drawString(45, 30, "AI-Based Precision Agriculture: Crop & Weed Monitoring | Guide: Prof. Yoga B S")
            self.drawRightString(567, 30, f"Page {self._pageNumber} of {page_count}")
            self.restoreState()

def get_proportional_image(path, target_width=510, max_height=260):
    """Loads image and scales it proportionally to fit within printable bounds."""
    if not os.path.exists(path):
        return None
    try:
        with Image.open(path) as img:
            w, h = img.size
            aspect = h / w
            calc_height = target_width * aspect
            if calc_height > max_height:
                calc_height = max_height
                target_width = calc_height / aspect
            return RLImage(path, width=target_width, height=calc_height)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def build_pdf():
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    os.makedirs(os.path.dirname(DOCS_PDF), exist_ok=True)

    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=54,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    
    # Custom Academic Palette
    C_PRIMARY = colors.HexColor("#0f172a")     # Deep Slate 900
    C_SECONDARY = colors.HexColor("#0369a1")   # Deep Sky Blue
    C_ACCENT = colors.HexColor("#059669")      # Emerald 600
    C_DARK = colors.HexColor("#1e293b")        # Slate 800
    C_MUTED = colors.HexColor("#64748b")       # Slate 500
    C_LIGHT = colors.HexColor("#f8fafc")       # Off-white
    C_CARD_BG = colors.HexColor("#f1f5f9")     # Card fill
    C_BORDER = colors.HexColor("#e2e8f0")      # Divider gray

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=C_PRIMARY,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=C_SECONDARY,
        spaceAfter=10
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=C_DARK
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=C_PRIMARY,
        spaceBefore=11,
        spaceAfter=5,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=C_SECONDARY,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=C_DARK,
        spaceAfter=5
    )
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )
    callout_style = ParagraphStyle(
        'Callout_Text',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=C_DARK
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=C_DARK
    )
    table_cell_center = ParagraphStyle(
        'TableCellCenter',
        parent=table_cell,
        alignment=1
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold',
        textColor=C_PRIMARY
    )
    table_cell_bold_center = ParagraphStyle(
        'TableCellBoldCenter',
        parent=table_cell_bold,
        alignment=1
    )
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=10,
        textColor=C_MUTED,
        alignment=1,
        spaceBefore=2,
        spaceAfter=6
    )

    story = []

    # =========================================================================
    # COVER / HEADER BANNER
    # =========================================================================
    story.append(Paragraph("Cross-Scale UAV–Satellite Data Fusion", title_style))
    story.append(Paragraph("AI-Based Precision Agriculture for Crop & Weed Monitoring | Comprehensive Research & Defense Report", subtitle_style))
    
    # Metadata Block Table
    meta_table_data = [
        [
            Paragraph("<b>Project ID:</b> PW26_YBS_02<br/><b>Institution:</b> PES University, Bengaluru<br/><b>Department:</b> CSE (AI & Machine Learning)<br/><b>Course:</b> Capstone Project (Sem V, 2024–2028)", meta_style),
            Paragraph("<b>Faculty Guide:</b> Prof. Yoga B S<br/><b>Student Authors:</b><br/>• Arnav Choudhary (PES1UG24AM050)<br/>• Vibhav M (PES1UG24CS527)<br/>• Vainik (PES1UG24CS513)<br/>• Manish Bala (PES1UG24CS263)", meta_style)
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[250, 260])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_CARD_BG),
        ('BOX', (0,0), (-1,-1), 1, C_BORDER),
        ('PADDING', (0,0), (-1,-1), 7),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # Executive Callout Box
    summary_box_data = [[
        Paragraph(
            "<b>EXECUTIVE SUMMARY & CORE RESEARCH HIGHLIGHTS:</b><br/>"
            "This report documents the comprehensive research, architecture evolution, empirical benchmarks, and deployment artifacts for "
            "Capstone Project <b>PW26_YBS_02</b>. Established initially as a ResNet18 U-Net baseline for UAV crop-weed segmentation, the system "
            "has transitioned to <b>SegFormer (MiT-B0 Vision Transformer)</b> as the primary architecture based on review feedback. SegFormer "
            "achieves <b>54.33% Test mIoU</b> (+3.37% over ResNet18, +3.23% over ResNet50) on strictly held-out UAV flights with only <b>3.98M "
            "parameters</b> (11× smaller and 7.1× faster than ResNet50). The framework integrates <b>Class-Masked Grad-CAM</b>, <b>Monte Carlo "
            "Dropout Predictive Entropy</b>, a commercial-grade <b>Variable Rate Application (VRA) Prescription Engine</b> saving up to 88% "
            "chemical volume, an interactive <b>8-Tab Streamlit Dashboard</b>, and active integration with the <b>PhenoBench (CVPR 2023)</b> benchmark.",
            callout_style
        )
    ]]
    summary_box = Table(summary_box_data, colWidths=[510])
    summary_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f0fdf4")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#86efac")),
        ('LINELEFT', (0,0), (-1,-1), 4, C_ACCENT),
        ('PADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(summary_box)
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 1: THE SCALE GAP & RESEARCH MOTIVATION
    # =========================================================================
    story.append(Paragraph("1. Research Motivation & The Remote Sensing Scale Gap", h1_style))
    story.append(Paragraph(
        "Modern precision agriculture faces a fundamental operational trade-off in remote sensing imagery. <b>Unmanned Aerial Vehicles (UAVs)</b> "
        "provide ultra-high spatial resolution (Ground Sampling Distance, GSD &approx; 1.0 cm/pixel), enabling individual leaf segmentation, crop "
        "canopy inspection, and weed-seedling discrimination. However, UAV missions suffer from strict battery limitations, small spatial footprints "
        "(&lt; 20 hectares per sortie), and high operational labor costs. Conversely, <b>Earth Observation Satellites</b> (such as Copernicus Sentinel-2) "
        "provide continuous, synoptic regional coverage (10–30 m/pixel) with short revisit intervals (5 days), but suffer from severe mixed-pixel "
        "spectral ambiguity where individual plants cannot be resolved.",
        body_style
    ))
    story.append(Paragraph(
        "<b>The Capstone Objective:</b> Establish a centimetre-accurate UAV ground reference baseline and develop a modular <b>Cross-Scale Attention "
        "Fusion Adapter</b> that bridges macro-scale satellite multi-spectral features with micro-scale UAV feature maps.",
        body_style
    ))
    
    img_arch = get_proportional_image(os.path.join(PROJECT_ROOT, "outputs/reports/system_architecture_block_diagram_slide.png"), target_width=510, max_height=185)
    if img_arch:
        story.append(img_arch)
        story.append(Paragraph("Figure 1: End-to-End System Architecture Block Diagram showing UAV pipeline, SegFormer core, Phase-II Satellite Adapter, and Actionable Farm Analytics.", caption_style))

    # =========================================================================
    # SECTION 2: ARCHITECTURAL EVOLUTION (WHAT CHANGED & WHY)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("2. Architecture Evolution: What Changed Since Review 1", h1_style))
    story.append(Paragraph(
        "Following Review 1 with project mentor <b>Prof. Yoga B S</b>, the panel recommended upgrading beyond legacy CNN backbones (ResNet18) "
        "to a contemporary vision architecture with self-attention mechanisms capable of capturing global field row geometries and naturally "
        "integrating multi-modal satellite tokens. Table 1 summarizes the complete paradigm evolution across project phases:",
        body_style
    ))

    evolution_table_data = [
        [
            Paragraph("Project Subsystem", table_header),
            Paragraph("Review 1 State (Previous)", table_header),
            Paragraph("Current State (Upgraded)", table_header),
            Paragraph("Engineering Justification & Value", table_header)
        ],
        [
            Paragraph("<b>Primary Architecture</b>", table_cell_bold),
            Paragraph("ResNet18 U-Net (CNN)<br/>14.33M parameters", table_cell),
            Paragraph("<b>SegFormer (MiT-B0 ViT)</b><br/>3.98M parameters", table_cell),
            Paragraph("Vision Transformer captures long-range crop row patterns without positional embedding limits; 72% smaller, +3.37% mIoU.", table_cell)
        ],
        [
            Paragraph("<b>Heavy Baseline</b>", table_cell_bold),
            Paragraph("None (Only basic CNN & DeepLab)", table_cell),
            Paragraph("<b>ResNet50 U-Net</b><br/>43.88M parameters", table_cell),
            Paragraph("Rigorous 'David vs. Goliath' benchmark proving CNN depth scaling fails on fine agricultural vegetation.", table_cell)
        ],
        [
            Paragraph("<b>Explainable AI</b>", table_cell_bold),
            Paragraph("Global Grad-CAM (unfocused class blending)", table_cell),
            Paragraph("<b>Class-Masked Target Grad-CAM</b>", table_cell),
            Paragraph("Mathematically isolates Crop vs. Weed gradient flow; produces razor-sharp class-specific activation maps.", table_cell)
        ],
        [
            Paragraph("<b>Uncertainty Metric</b>", table_cell_bold),
            Paragraph("Heuristic softmax thresholding", table_cell),
            Paragraph("<b>Monte Carlo Dropout Entropy</b>", table_cell),
            Paragraph("10 stochastic forward passes quantify Bayesian epistemic uncertainty; flags occluded weed boundaries.", table_cell)
        ],
        [
            Paragraph("<b>Agronomic Prescription</b>", table_cell_bold),
            Paragraph("Basic pixel count percentage", table_cell),
            Paragraph("<b>4-Tier Variable Rate Application (VRA)</b>", table_cell),
            Paragraph("Commercial drone spray mission generator; exports DJI Agras CSV, GeoJSON, and financial/CO2 savings audit.", table_cell)
        ],
        [
            Paragraph("<b>Data Benchmark</b>", table_cell_bold),
            Paragraph("WeedMap flights only", table_cell),
            Paragraph("<b>WeedMap + PhenoBench (CVPR 2023)</b>", table_cell),
            Paragraph("Curated setup for zero-shot domain transfer between Swiss (ETH) and German (Univ. of Bonn) sugar beet fields.", table_cell)
        ],
        [
            Paragraph("<b>User Interface</b>", table_cell_bold),
            Paragraph("CLI + 7-Tab basic UI", table_cell),
            Paragraph("<b>8-Tab High-Contrast Dark Theme</b>", table_cell),
            Paragraph("Multi-model selector, interactive VRA economics slider, GIS downloads, running as background daemon.", table_cell)
        ]
    ]
    t_evo = Table(evolution_table_data, colWidths=[90, 110, 120, 190])
    t_evo.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, C_CARD_BG]),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_evo)
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 3: THE 5-MODEL BENCHMARK SUITE & DAVID VS GOLIATH STUDY
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("3. Empirical Benchmark Suite & The 'David vs. Goliath' Study", h1_style))
    story.append(Paragraph(
        "To rigorously validate the new architecture, all five deep learning models were trained and evaluated under strictly identical conditions "
        "on the held-out test split (Flight `003`, 149 unseen tiles, 9,704,970 test pixels). Table 2 presents the test generalization metrics:",
        body_style
    ))

    benchmark_table_data = [
        [
            Paragraph("Model Architecture", table_header),
            Paragraph("Params", table_header),
            Paragraph("Model Size", table_header),
            Paragraph("CPU Latency", table_header),
            Paragraph("Val mIoU", table_header),
            Paragraph("Test mIoU", table_header),
            Paragraph("Crop IoU", table_header),
            Paragraph("Weed IoU", table_header),
            Paragraph("Macro Dice", table_header)
        ],
        [
            Paragraph("<b>SegFormer (MiT-B0) [Primary]</b>", table_cell_bold),
            Paragraph("<b>3.98 M</b>", table_cell_center),
            Paragraph("<b>15.20 MB</b>", table_cell_center),
            Paragraph("<b>13.96 ms</b>", table_cell_center),
            Paragraph("<b>54.34%</b>", table_cell_center),
            Paragraph("<b>54.33%</b>", table_cell_bold_center),
            Paragraph("<b>42.13%</b>", table_cell_bold_center),
            Paragraph("<b>23.44%</b>", table_cell_center),
            Paragraph("<b>65.32%</b>", table_cell_bold_center)
        ],
        [
            Paragraph("ResNet50 U-Net [Heavy Baseline]", table_cell),
            Paragraph("43.88 M", table_cell_center),
            Paragraph("167.39 MB", table_cell_center),
            Paragraph("99.89 ms", table_cell_center),
            Paragraph("60.48%", table_cell_center),
            Paragraph("51.10%", table_cell_center),
            Paragraph("36.76%", table_cell_center),
            Paragraph("20.10%", table_cell_center),
            Paragraph("61.80%", table_cell_center)
        ],
        [
            Paragraph("DeepLabV3+ (ResNet18)", table_cell),
            Paragraph("16.60 M", table_cell_center),
            Paragraph("63.34 MB", table_cell_center),
            Paragraph("34.34 ms", table_cell_center),
            Paragraph("52.40%", table_cell_center),
            Paragraph("51.70%", table_cell_center),
            Paragraph("41.11%", table_cell_center),
            Paragraph("17.73%", table_cell_center),
            Paragraph("62.16%", table_cell_center)
        ],
        [
            Paragraph("ResNet18 U-Net [Phase-I Baseline]", table_cell),
            Paragraph("14.33 M", table_cell_center),
            Paragraph("54.67 MB", table_cell_center),
            Paragraph("34.80 ms", table_cell_center),
            Paragraph("56.45%", table_cell_center),
            Paragraph("50.96%", table_cell_center),
            Paragraph("28.96%", table_cell_center),
            Paragraph("26.01%", table_cell_center),
            Paragraph("61.72%", table_cell_center)
        ],
        [
            Paragraph("Basic CNN Segmenter", table_cell),
            Paragraph("0.80 M", table_cell_center),
            Paragraph("3.06 MB", table_cell_center),
            Paragraph("35.29 ms", table_cell_center),
            Paragraph("43.10%", table_cell_center),
            Paragraph("42.27%", table_cell_center),
            Paragraph("20.83%", table_cell_center),
            Paragraph("10.81%", table_cell_center),
            Paragraph("50.50%", table_cell_center)
        ]
    ]
    t_bench = Table(benchmark_table_data, colWidths=[120, 42, 48, 52, 46, 50, 50, 50, 52])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#f0fdf4"), colors.white, C_CARD_BG, colors.white, C_CARD_BG]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # NEW SEPARATE TABLE: ARCHITECTURE & COMPUTATIONAL SPECIFICATIONS
    # -------------------------------------------------------------------------
    story.append(Paragraph("<b>Table 3: Architecture & Computational Specifications</b>", h2_style))
    story.append(Paragraph(
        "Profiles the exact computational memory footprint, parameter counts, batch inference latency ($B=4$), single-image inference latency, "
        "and output tensor resolutions for autonomous edge deployment:",
        body_style
    ))

    specs_table_data = [
        [
            Paragraph("Architecture", table_header),
            Paragraph("Parameters (M)", table_header),
            Paragraph("Size (MB)", table_header),
            Paragraph("Latency B=4 (ms)", table_header),
            Paragraph("Per-Image Latency (ms)", table_header),
            Paragraph("Output Resolution", table_header)
        ],
        [
            Paragraph("<b>SegFormer (MiT-B0) [New Primary Proposed]</b>", table_cell_bold),
            Paragraph("<b>3.98</b>", table_cell_bold_center),
            Paragraph("<b>15.20</b>", table_cell_bold_center),
            Paragraph("<b>55.85</b>", table_cell_bold_center),
            Paragraph("<b>13.96</b>", table_cell_bold_center),
            Paragraph("256 &times; 256 &times; 3", table_cell_center)
        ],
        [
            Paragraph("U-Net (ResNet50) [Heavyweight Baseline]", table_cell),
            Paragraph("43.88", table_cell_center),
            Paragraph("167.39", table_cell_center),
            Paragraph("399.54", table_cell_center),
            Paragraph("99.89", table_cell_center),
            Paragraph("256 &times; 256 &times; 3", table_cell_center)
        ],
        [
            Paragraph("U-Net (ResNet18) [Phase-I Baseline]", table_cell),
            Paragraph("14.33", table_cell_center),
            Paragraph("54.67", table_cell_center),
            Paragraph("134.09", table_cell_center),
            Paragraph("33.52", table_cell_center),
            Paragraph("256 &times; 256 &times; 3", table_cell_center)
        ],
        [
            Paragraph("DeepLabV3+ (ResNet18)", table_cell),
            Paragraph("16.60", table_cell_center),
            Paragraph("63.34", table_cell_center),
            Paragraph("148.66", table_cell_center),
            Paragraph("37.16", table_cell_center),
            Paragraph("256 &times; 256 &times; 3", table_cell_center)
        ],
        [
            Paragraph("Basic CNN Segmenter", table_cell),
            Paragraph("0.80", table_cell_center),
            Paragraph("3.06", table_cell_center),
            Paragraph("140.10", table_cell_center),
            Paragraph("35.03", table_cell_center),
            Paragraph("256 &times; 256 &times; 3", table_cell_center)
        ]
    ]
    t_specs = Table(specs_table_data, colWidths=[175, 65, 60, 70, 70, 70])
    t_specs.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#f0fdf4"), colors.white, C_CARD_BG, colors.white, C_CARD_BG]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t_specs)
    story.append(Spacer(1, 6))

    # Narrative on David vs Goliath
    story.append(Paragraph("<b>The 'David vs. Goliath' Empirical Proof:</b>", h2_style))
    story.append(Paragraph(
        "A central hypothesis tested during this phase was whether increasing convolutional network depth (from ResNet18 to ResNet50) "
        "improves precision agricultural segmentation. Our empirical results deliver a definitive answer:<br/>"
        "• <b>ResNet50 vs. ResNet18:</b> Quadrupling convolutional depth (14.33M &rarr; 43.88M parameters) yielded a marginal +0.14% mIoU "
        "(50.96% &rarr; 51.10%) while <b>tripling latency (34.8 ms &rarr; 99.9 ms)</b> and quadrupling memory (54.7 MB &rarr; 167.4 MB). "
        "Deeper CNNs suffer from spatial feature degradation over fine, sub-centimeter leaf edges.<br/>"
        "• <b>SegFormer vs. ResNet50:</b> SegFormer (3.98M params) <b>outperforms ResNet50 by +3.23% mIoU and +5.37% Crop IoU</b> "
        "while being <b>11.0× smaller</b> and <b>7.1× faster</b> on CPU inference. SegFormer's Multi-Scale Self-Attention preserves "
        "hierarchical spatial detail and learns continuous geometric dependencies between crop rows across variable UAV camera angles.",
        body_style
    ))

    img_bar = get_proportional_image(os.path.join(PROJECT_ROOT, "outputs/graphs/models_comparison_barchart.png"), target_width=510, max_height=145)
    if img_bar:
        story.append(img_bar)
        story.append(Paragraph("Figure 2: Empirical Benchmark comparison across all 5 architectures on held-out Flight 003 test tiles.", caption_style))

    # =========================================================================
    # SECTION 3.1: SEPARATE INDIVIDUAL MODEL TRAINING DYNAMICS & LOSS CONVERGENCE
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("3.1 Individual Model Training Dynamics & Loss Convergence", h1_style))
    story.append(Paragraph(
        "Each architecture was trained across 10 epochs using identical optimization schemes (Adam optimizer, Cosine Annealing learning rate schedule, "
        "and normalized median frequency balancing hybrid loss). Below are the individual training and validation convergence dynamics for each model:",
        body_style
    ))

    # Model 1: SegFormer (MiT-B0)
    story.append(Paragraph("<b>(a) SegFormer (MiT-B0) [Primary Proposed Architecture]</b>", h2_style))
    story.append(Paragraph(
        "SegFormer exhibits rapid loss descent and stable multi-class validation metric progression. The pre-trained Mix-Transformer backbone provides "
        "rich visual feature priors, enabling crop IoU to saturate above 42% by Epoch 6 while maintaining smooth loss decay without overfitting.",
        body_style
    ))
    img_seg_curve = get_proportional_image(os.path.join(PROJECT_ROOT, "outputs/graphs/segformer_training_curves.png"), target_width=500, max_height=165)
    if img_seg_curve:
        story.append(img_seg_curve)
        story.append(Paragraph("Figure 3a: SegFormer (MiT-B0) Training Loss (left) and Validation mIoU / Crop IoU / Weed IoU progression (right) across 10 epochs.", caption_style))

    story.append(Spacer(1, 6))

    # Model 2: ResNet50 U-Net
    story.append(Paragraph("<b>(b) U-Net (ResNet50) [Heavyweight Comparative Baseline]</b>", h2_style))
    story.append(Paragraph(
        "ResNet50 reaches low training loss (0.5099) and high in-domain validation mIoU (60.48%), but suffers an acute generalization drop on unseen "
        "held-out Flight 003 (51.10% Test mIoU). Its 43.88M parameters overfit to field-specific soil textures, confirming the limitations of convolutional depth scaling.",
        body_style
    ))
    img_r50_curve = get_proportional_image(os.path.join(PROJECT_ROOT, "outputs/graphs/resnet50_training_curves.png"), target_width=500, max_height=165)
    if img_r50_curve:
        story.append(img_r50_curve)
        story.append(Paragraph("Figure 3b: ResNet50 U-Net Training Loss (left) and Validation mIoU / Crop IoU / Weed IoU progression (right) across 10 epochs.", caption_style))

    story.append(PageBreak())

    # Model 3: ResNet18 U-Net
    story.append(Paragraph("<b>(c) U-Net (ResNet18) [Phase-I Baseline]</b>", h2_style))
    story.append(Paragraph(
        "ResNet18 U-Net converges smoothly within 10 epochs, balancing crop and weed learning via hierarchical skip connections. However, its local "
        "receptive field plateaus around 50.96% test mIoU, struggling with fragmented weed foliage between crop rows.",
        body_style
    ))
    img_unet_curve = get_proportional_image(os.path.join(PROJECT_ROOT, "outputs/graphs/unet_training_curves.png"), target_width=500, max_height=165)
    if img_unet_curve:
        story.append(img_unet_curve)
        story.append(Paragraph("Figure 3c: ResNet18 U-Net Training Loss (left) and Validation mIoU / Crop IoU / Weed IoU progression (right) across 10 epochs.", caption_style))

    story.append(Spacer(1, 6))

    # Model 4: DeepLabV3+
    story.append(Paragraph("<b>(d) DeepLabV3+ (ResNet18) [Atrous Spatial Pyramid Pooling]</b>", h2_style))
    story.append(Paragraph(
        "DeepLabV3+ leverages multi-rate dilated convolutions (ASPP) to capture broad crop canopy context (achieving 41.11% Crop IoU). However, "
        "dilated filters cause gridding artifacts on sub-centimeter weed pixels, capping weed IoU at 17.73%.",
        body_style
    ))
    img_deep_curve = get_proportional_image(os.path.join(PROJECT_ROOT, "outputs/graphs/deeplab_training_curves.png"), target_width=500, max_height=165)
    if img_deep_curve:
        story.append(img_deep_curve)
        story.append(Paragraph("Figure 3d: DeepLabV3+ Training Loss (left) and Validation mIoU / Crop IoU / Weed IoU progression (right) across 10 epochs.", caption_style))

    story.append(PageBreak())

    # Model 5: Basic CNN Segmenter
    story.append(Paragraph("<b>(e) Basic CNN Segmenter [Ablation Reference Baseline]</b>", h2_style))
    story.append(Paragraph(
        "Lacking multi-scale skip connections or attention mechanisms, the Basic CNN segmenter suffers severe semantic degradation during bottleneck "
        "downsampling. Weed IoU reaches only 10.81%, demonstrating the critical necessity of advanced multi-scale feature aggregation.",
        body_style
    ))
    img_cnn_curve = get_proportional_image(os.path.join(PROJECT_ROOT, "outputs/graphs/baseline_cnn_training_curves.png"), target_width=500, max_height=165)
    if img_cnn_curve:
        story.append(img_cnn_curve)
        story.append(Paragraph("Figure 3e: Basic CNN Segmenter Training Loss (left) and Validation mIoU / Crop IoU / Weed IoU progression (right) across 10 epochs.", caption_style))

    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 4: DATASET & LEAKAGE-FREE SPLITTING PROTOCOL
    # =========================================================================
    story.append(Paragraph("4. Dataset & Leakage-Free Geographic Splitting Protocol", h1_style))
    story.append(Paragraph(
        "To prevent spatial data leakage and biased evaluation, tiles from the <b>WeedMap</b> benchmark (Sa et al., MDPI Remote Sensing 2018) "
        "were split strictly by <b>independent UAV flights and geographic locations</b> across Eschikon (Switzerland) and Rheinbach (Germany). "
        "No spatial patches from the test flight were ever seen during training or validation.",
        body_style
    ))

    data_split_data = [
        [
            Paragraph("Dataset Split", table_header),
            Paragraph("Flight IDs Included", table_header),
            Paragraph("Tile Count", table_header),
            Paragraph("Percentage", table_header),
            Paragraph("Labeled Pixels", table_header),
            Paragraph("Role & Geographic Condition", table_header)
        ],
        [
            Paragraph("<b>Training Set</b>", table_cell_bold),
            Paragraph("Flight 000, 002, 004", table_cell),
            Paragraph("368 tiles", table_cell_center),
            Paragraph("57.6%", table_cell_center),
            Paragraph("63,590,400 px", table_cell_center),
            Paragraph("Baseline multi-condition learning across multiple dates and sun angles.", table_cell)
        ],
        [
            Paragraph("<b>Validation Set</b>", table_cell_bold),
            Paragraph("Flight 001", table_cell),
            Paragraph("122 tiles", table_cell_center),
            Paragraph("19.1%", table_cell_center),
            Paragraph("21,081,600 px", table_cell_center),
            Paragraph("Hyperparameter tuning, early stopping, and checkpoint selection.", table_cell)
        ],
        [
            Paragraph("<b>Held-Out Test Set</b>", table_cell_bold),
            Paragraph("<b>Flight 003</b>", table_cell_bold),
            Paragraph("149 tiles", table_cell_bold_center),
            Paragraph("23.3%", table_cell_bold_center),
            Paragraph("25,747,200 px", table_cell_bold_center),
            Paragraph("<b>Strictly unseen flight campaign</b> for cross-field domain generalization.", table_cell)
        ]
    ]
    t_data = Table(data_split_data, colWidths=[80, 85, 55, 55, 75, 160])
    t_data.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, C_CARD_BG, colors.HexColor("#fef2f2")]),
        ('PADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_data)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Extreme Class Imbalance Mitigation:</b> The dataset exhibits an extreme 99:1 background-to-vegetation ratio: "
        "Soil/Background constitutes <b>98.76%</b> ($109.0M$ px), Sugar Beet Crop constitutes <b>0.45%</b> ($500K$ px), and Weeds constitute <b>0.79%</b> ($873K$ px). "
        "To prevent the network from collapsing into a trivial majority-class predictor, we formulated a dual objective function: "
        "&nbsp;<i>&Lscr;<sub>total</sub> = 0.5 &middot; &Lscr;<sub>WCE</sub> + 0.5 &middot; &Lscr;<sub>Dice</sub></i>, where class weights "
        "<b>w</b> = [0.0087, 1.9017, 1.0896] were computed via normalized Median Frequency Balancing.",
        body_style
    ))

    # =========================================================================
    # SECTION 5: SEGFORMER ARCHITECTURE & SATELLITE FUSION ADAPTER
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("5. Primary Proposed Architecture: SegFormer (MiT-B0)", h1_style))
    story.append(Paragraph(
        "SegFormer (Xie et al., NeurIPS 2021) overcomes the core limitation of standard Vision Transformers (such as ViT) "
        "which require fixed-resolution positional embeddings. SegFormer incorporates four architectural breakthroughs:",
        body_style
    ))
    story.append(Paragraph("• <b>Hierarchical Transformer Encoder:</b> Generates multi-scale feature maps at {1/4, 1/8, 1/16, 1/32} of the original image resolution, matching the multi-scale hierarchy of CNN feature pyramids without rigid inductive bias.", bullet_style))
    story.append(Paragraph("• <b>Overlapped Patch Merging:</b> Preserves local continuity around plant boundaries by merging overlapping 3&times;3 patches (stride 2) rather than non-overlapping patches, preventing boundary artifacting.", bullet_style))
    story.append(Paragraph("• <b>Sequence Reduction Attention (SRA):</b> Reduces the spatial dimension of key and value vectors by a reduction ratio <i>R</i>, slashing multi-head self-attention computational complexity from <i>O(N<sup>2</sup>)</i> to <i>O(N<sup>2</sup>/R)</i>.", bullet_style))
    story.append(Paragraph("• <b>Lightweight All-MLP Decoder:</b> Unifies multi-level feature channels via standard MLP layers and upsamples them to <i>H/4 &times; W/4</i>, avoiding heavy deconvolutional decoders and enabling real-time CPU execution (13.96 ms).", bullet_style))

    img_seg = get_proportional_image(os.path.join(PROJECT_ROOT, "outputs/reports/segformer_architecture_detail_slide.png"), target_width=510, max_height=185)
    if img_seg:
        story.append(img_seg)
        story.append(Paragraph("Figure 4: SegFormer (MiT-B0) architectural pipeline with Hierarchical Encoder, SRA, All-MLP Decoder, and Phase-II Satellite Adapter.", caption_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Phase-II Cross-Scale Satellite Fusion Adapter:</b>", h2_style))
    story.append(Paragraph(
        "Built directly into <code>models/segformer.py</code>, the <code>CrossScaleSatelliteFusionAdapter</code> module bridges the spatial gap between "
        "low-resolution multi-spectral satellite features (e.g. Sentinel-2 Level-2A, 10m GSD, 5 bands: Red, Green, Blue, NIR Band 8, RedEdge Band 5) "
        "and high-resolution UAV feature maps (1cm GSD). The satellite tensor is linearly projected to the decoder feature dimension, "
        "cross-attended with UAV tokens, and injected as a residual prior into the prediction head: "
        "&nbsp;<i>z<sub>fused</sub> = &gamma; &middot; CrossAttn(Q = z<sub>UAV</sub>, K = z<sub>Sat</sub>, V = z<sub>Sat</sub>) + z<sub>UAV</sub></i>.",
        body_style
    ))

    # =========================================================================
    # SECTION 6: EXPLAINABLE AI & UNCERTAINTY ESTIMATION
    # =========================================================================
    story.append(Spacer(1, 6))
    story.append(Paragraph("6. Explainable AI & Bayesian Uncertainty Estimation", h1_style))
    story.append(Paragraph(
        "To provide trust and safety for autonomous robotic spraying, the pipeline incorporates two complementary validation layers:",
        body_style
    ))
    story.append(Paragraph(
        "<b>1. Class-Masked Target Grad-CAM (Fixed from Review 1):</b><br/>"
        "In Review 1, Grad-CAM pooled gradients across all output channels simultaneously, resulting in blurry, ambiguous heatmaps. "
        "We re-engineered the explainability engine with class-specific backward targeting. Gradients are isolated exclusively for target class <i>c</i>: "
        "&nbsp;&alpha;<sub>k</sub><sup>c</sup> = (1/Z) &sum;<sub>i</sub> &sum;<sub>j</sub> (&part; Y<sup>c</sup> / &part; A<sub>ij</sub><sup>k</sup>). "
        "When querying Crop (Class 1), activations concentrate tightly on sugar beet leaves; when querying Weed (Class 2), activations isolate inter-row weeds.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. Monte Carlo Dropout Predictive Uncertainty:</b><br/>"
        "By retaining dropout active during inference across <i>T = 10</i> stochastic forward passes, the model estimates Bayesian epistemic uncertainty. "
        "Pixel-wise Shannon entropy is computed: &nbsp;<i>H(p) = -&sum;<sub>c</sub> p&#772;<sub>c</sub> log<sub>2</sub>(p&#772;<sub>c</sub>)</i>. "
        "High-entropy zones pinpoint occluded plant boundaries and soil-weed transitions, preventing false-positive herbicide discharge.",
        body_style
    ))

    img_xai = get_proportional_image(os.path.join(PROJECT_ROOT, "outputs/reports/gradcam_sample_explanation.png"), target_width=510, max_height=125)
    if img_xai:
        story.append(img_xai)
        story.append(Paragraph("Figure 5: Class-Masked Grad-CAM verifying distinct physiological activations for Sugar Beet (Crop) vs. Weeds.", caption_style))

    img_unc = get_proportional_image(os.path.join(PROJECT_ROOT, "outputs/reports/mc_dropout_uncertainty_sample.png"), target_width=510, max_height=125)
    if img_unc:
        story.append(img_unc)
        story.append(Paragraph("Figure 6: Monte Carlo Dropout Uncertainty (Predictive Entropy Map) identifying high-risk decision boundaries.", caption_style))

    # =========================================================================
    # SECTION 7: PRECISION PRESCRIPTION & HERBICIDE COST-SAVINGS (VRA)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("7. Precision Agronomic Prescription & Economics Engine (VRA)", h1_style))
    story.append(Paragraph(
        "<b>Add-On 3 Full Implementation:</b> Machine learning segmentation is only valuable if it drives physical actuation in the field. "
        "We developed an autonomous <b>Variable Rate Application (VRA)</b> prescription engine that converts pixel-level weed masks into "
        "commercial drone spraying missions with quantifiable economic and ecological benefits.",
        body_style
    ))

    tier_table_data = [
        [
            Paragraph("Tier", table_header),
            Paragraph("Prescription Strategy", table_header),
            Paragraph("Weed Density", table_header),
            Paragraph("Nozzle Actuation", table_header),
            Paragraph("PWM Flow", table_header),
            Paragraph("Multiplier", table_header),
            Paragraph("Agronomic Objective", table_header)
        ],
        [
            Paragraph("<b>Tier 0</b>", table_cell_bold),
            Paragraph("<b>Zero-Spray Zone</b>", table_cell_bold),
            Paragraph("0.0% &ndash; 1.0%", table_cell),
            Paragraph("<font color='#16a34a'><b>OFF (Complete Shutoff)</b></font>", table_cell),
            Paragraph("0%", table_cell_center),
            Paragraph("0.0&times; (0 L/ha)", table_cell_center),
            Paragraph("Healthy crop/clean soil; chemical discharge prohibited.", table_cell)
        ],
        [
            Paragraph("<b>Tier 1</b>", table_cell_bold),
            Paragraph("Micro-Pulse Targeted", table_cell),
            Paragraph("1.0% &ndash; 5.0%", table_cell),
            Paragraph("Pulsed Micro-Dose", table_cell),
            Paragraph("30%", table_cell_center),
            Paragraph("0.3&times; (60 L/ha)", table_cell_center),
            Paragraph("Early-stage emergent weeds; low-volume spot-suppression.", table_cell)
        ],
        [
            Paragraph("<b>Tier 2</b>", table_cell_bold),
            Paragraph("Moderate Standard", table_cell),
            Paragraph("5.0% &ndash; 15.0%", table_cell),
            Paragraph("Continuous Active", table_cell),
            Paragraph("70%", table_cell_center),
            Paragraph("0.7&times; (140 L/ha)", table_cell_center),
            Paragraph("Established inter-row weed clusters; targeted eradication.", table_cell)
        ],
        [
            Paragraph("<b>Tier 3</b>", table_cell_bold),
            Paragraph("<font color='#dc2626'><b>Curative Maximum</b></font>", table_cell_bold),
            Paragraph("&gt; 15.0%", table_cell),
            Paragraph("Full-Throttle Flow", table_cell),
            Paragraph("100%", table_cell_center),
            Paragraph("1.0&times; (200 L/ha)", table_cell_center),
            Paragraph("Heavy localized infestation; maximum curative dosage.", table_cell)
        ]
    ]
    t_tier = Table(tier_table_data, colWidths=[45, 105, 65, 95, 45, 65, 90])
    t_tier.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#f0fdf4"), colors.white, C_CARD_BG, colors.HexColor("#fef2f2")]),
        ('PADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_tier)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Rigorous Field Economics & Environmental Spared Metrics:</b><br/>"
        "Testing the VRA engine on held-out flight 003 field tiles under realistic agronomic parameters (5.0 ha field, 200 L/ha base broadcast rate, "
        "₹650/L chemical cost, active ingredient concentration 350 g/L) demonstrated dramatic advantages over blanket broadcast spraying:<br/>"
        "• <b>Nozzle Shutoff Rate:</b> <b>22.3% to 85.0%</b> of the field area required zero chemical application.<br/>"
        "• <b>Volume Slashed:</b> Chemical consumption was reduced by <b>29.1% to 88.4%</b> (saving 291 to 884 Liters per 5 ha).<br/>"
        "• <b>Direct Financial Savings:</b> Farmers save between <b>₹1,89,150 to ₹5,74,600 ($2,238 to $6,799 USD)</b> per 5 hectares per spray cycle.<br/>"
        "• <b>Active Environmental Spared:</b> Prevented <b>101.9 kg to 309.4 kg</b> of toxic synthetic active chemical leaching into groundwater.<br/>"
        "• <b>Carbon Offset:</b> Saved <b>131.0 kg to 397.8 kg CO<sub>2</sub>e</b> in chemical manufacturing and drone battery endurance.",
        body_style
    ))

    img_presc = get_proportional_image(os.path.join(PROJECT_ROOT, "outputs/reports/cli_prescription_map.png"), target_width=510, max_height=160)
    if img_presc:
        story.append(img_presc)
        story.append(Paragraph("Figure 7: Multi-Panel Autonomous Prescription Report: RGB Aerial Tile (left), Continuous Weed Density Heatmap (middle), and 4-Color Actuation Grid with Economic KPI summary (right).", caption_style))

    story.append(Paragraph(
        "<b>Commercial Autonomous Drone Export Formats:</b><br/>"
        "1. <b>DJI Agras CSV Mission (<code>dji_agras_drone_mission.csv</code>):</b> Waypoints with GPS Lat/Lon coordinates, target flight altitude (3.0 m), flight speed (3.5 m/s), nozzle PWM duty cycle (0–100%), and prescription dosage rate for direct upload to DJI Agras T30/T40/T50 handheld controllers.<br/>"
        "2. <b>GIS Vector GeoJSON (<code>agrifusion_vra_prescription.geojson</code>):</b> Vector polygons with embedded tier metadata for QGIS, ArcGIS, and Climate FieldView.<br/>"
        "3. <b>Agronomic Audit Certificate (<code>agrifusion_environmental_compliance.json</code>):</b> Cryptographically auditable compliance record meeting EU Farm to Fork and ICAR Precision Weeding directives.",
        body_style
    ))

    # =========================================================================
    # SECTION 8: SOFTWARE ENGINEERING & INTERACTIVE WEB DASHBOARD
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("8. Software Engineering & Interactive Deployment", h1_style))
    story.append(Paragraph(
        "The project follows strict modular software engineering practices. All data pipelines, model definitions, inference engines, and database stores "
        "are encapsulated into reusable modules across the repository:",
        body_style
    ))

    repo_structure_data = [
        [
            Paragraph("Module / Directory", table_header),
            Paragraph("Key Responsibilities & Components", table_header),
            Paragraph("Technologies / Formats", table_header)
        ],
        [
            Paragraph("<code>models/segformer.py</code>", table_cell_bold),
            Paragraph("Primary SegFormer (MiT-B0) implementation with All-MLP Decoder and built-in CrossScaleSatelliteFusionAdapter.", table_cell),
            Paragraph("PyTorch, HuggingFace Transformers", table_cell)
        ],
        [
            Paragraph("<code>models/unet.py</code>", table_cell_bold),
            Paragraph("ResNet18 and ResNet50 U-Net architectures with 4-stage skip connections and satellite hook.", table_cell),
            Paragraph("PyTorch, Torchvision", table_cell)
        ],
        [
            Paragraph("<code>src/prescription.py</code>", table_cell_bold),
            Paragraph("4-tier VRA zoning, economic/environmental calculators, DJI Agras CSV and GeoJSON exporters.", table_cell),
            Paragraph("NumPy, Pandas, GeoJSON", table_cell)
        ],
        [
            Paragraph("<code>src/database.py</code>", table_cell_bold),
            Paragraph("Relational audit logging store (<code>outputs/agrifusion.db</code>) with predictions and prescriptions tables.", table_cell),
            Paragraph("SQLite3", table_cell)
        ],
        [
            Paragraph("<code>src/explainability.py</code>", table_cell_bold),
            Paragraph("Class-Masked target Grad-CAM engine with backward hooks on bottleneck and transformer blocks.", table_cell),
            Paragraph("PyTorch Autograd, Matplotlib", table_cell)
        ],
        [
            Paragraph("<code>src/uncertainty.py</code>", table_cell_bold),
            Paragraph("Monte Carlo Dropout sampling engine computing predictive mean, variance, and Shannon entropy.", table_cell),
            Paragraph("PyTorch, SciPy", table_cell)
        ],
        [
            Paragraph("<code>app/app.py</code>", table_cell_bold),
            Paragraph("Interactive 8-tab Streamlit dashboard running live at http://localhost:8501 with high-contrast dark theme.", table_cell),
            Paragraph("Streamlit, Uvicorn, CSS3", table_cell)
        ]
    ]
    t_repo = Table(repo_structure_data, colWidths=[120, 260, 130])
    t_repo.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, C_CARD_BG]),
        ('PADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_repo)
    story.append(Spacer(1, 8))

    # Gallery of predictions
    img_gal = get_proportional_image(os.path.join(PROJECT_ROOT, "outputs/predictions/segformer_test_predictions_gallery.png"), target_width=510, max_height=215)
    if img_gal:
        story.append(img_gal)
        story.append(Paragraph("Figure 8: SegFormer (MiT-B0) Qualitative Prediction Gallery on held-out test tiles (Flight 003): Input RGB (left), Ground Truth (middle), Predicted Mask (right).", caption_style))

    # =========================================================================
    # SECTION 9: PHENOBENCH INTEGRATION & FUTURE ROADMAP
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("9. PhenoBench Integration & Cross-Dataset Domain Transfer", h1_style))
    story.append(Paragraph(
        "To establish academic prestige and test domain robustness, the project is actively integrating the <b>PhenoBench benchmark (CVPR 2023)</b> "
        "created by the University of Bonn and PhenoRob. Sourced from <code>https://www.phenobench.org/data/PhenoBench-v110.zip</code> (7.63 GB), "
        "PhenoBench provides high-resolution 1024&times;1024 UAV imagery of sugar beet fields with pixel-level ground truth masks.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Zero-Shot Cross-Dataset Validation Protocol:</b><br/>"
        "Because both WeedMap and PhenoBench target <b>Sugar Beet (<i>Beta vulgaris</i>)</b> with identical 3-class ontologies "
        "(<code>0: Soil</code>, <code>1: Crop</code>, <code>2: Weed</code>), our trained models can be tested zero-shot on PhenoBench "
        "without re-training. This evaluates how well SegFormer's self-attention representations generalize across distinct geographic locations, "
        "sensors, and lighting (ETH Zurich flights in Switzerland &rarr; University of Bonn fields in Germany).",
        body_style
    ))

    img_road = get_proportional_image(os.path.join(PROJECT_ROOT, "outputs/reports/further_implementation_roadmap_slide.png"), target_width=510, max_height=175)
    if img_road:
        story.append(img_road)
        story.append(Paragraph("Figure 9: Future Implementation Roadmap showing Phase-II Multi-Modal Satellite Fusion, Multi-Crop Expansion (Wheat & Rice), and Edge TPU Deployment.", caption_style))

    # =========================================================================
    # SECTION 10: DEFENSE GUIDE & FREQUENTLY ASKED QUESTIONS
    # =========================================================================
    story.append(Spacer(1, 6))
    story.append(Paragraph("10. Presentation Defense & Viva Cheat Sheet", h1_style))
    story.append(Paragraph(
        "Below are concise, authoritative answers prepared for the project evaluation panel:",
        body_style
    ))

    defense_qa = [
        ("Q1: Why did you transition from ResNet18 U-Net to SegFormer?",
         "ResNet18 relies on local convolutional inductive bias with fixed receptive fields, missing global canopy alignment across parallel crop rows. SegFormer employs Multi-Scale Self-Attention without positional embedding limits, capturing both fine leaf edges and field-wide geometry. It achieves higher test mIoU (54.33% vs 50.96%) with 72% fewer parameters (3.98M vs 14.33M) and faster CPU inference (13.96 ms vs 34.80 ms)."),
        ("Q2: Why does SegFormer beat the 43.88M parameter ResNet50 model?",
         "This confirms the 'David vs. Goliath' finding: in agricultural UAV imagery, scaling CNN depth adds parameter bloat without improving spatial context. Fine weed seedlings occupy only 5–15 pixels; deep residual bottlenecks downsample and blur these features, degrading weed IoU (20.10% on ResNet50 vs 23.44% on SegFormer)."),
        ("Q3: How do you address extreme 99:1 background class imbalance?",
         "We used normalized Median Frequency Balancing to derive class weights w = [0.0087, 1.9017, 1.0896] and trained on a combined loss L = 0.5·L_WCE + 0.5·L_Dice. This penalizes false negatives on rare crop (0.45%) and weed (0.79%) pixels without destabilizing gradient flow."),
        ("Q4: How does your system interface with physical agricultural drones?",
         "Through the Variable Rate Application (VRA) engine. It aggregates pixel predictions into a 16x16 field grid, assigns 4 discrete spray dosage tiers (0%, 30%, 70%, 100%), and exports waypoint mission CSVs directly compatible with DJI Agras T30/T40/T50 flight controllers.")
    ]

    for q, a in defense_qa:
        story.append(Paragraph(f"<b>{q}</b>", h2_style))
        story.append(Paragraph(a, body_style))

    # Build Document
    print(f"Compiling publication-grade PDF report to: {OUTPUT_PDF}...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print("PDF build complete!")

    # Copy to docs/ and artifact directory for immediate user download
    shutil.copyfile(OUTPUT_PDF, DOCS_PDF)
    print(f"Copied to: {DOCS_PDF}")
    
    artifact_dest = os.path.join(ARTIFACT_DIR, "FINAL_CAPSTONE_RESEARCH_REPORT.pdf")
    shutil.copyfile(OUTPUT_PDF, artifact_dest)
    print(f"Copied to Artifact Directory: {artifact_dest}")

if __name__ == "__main__":
    build_pdf()
