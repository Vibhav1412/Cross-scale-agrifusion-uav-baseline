"""
Phase 8: Streamlit Web Dashboard for UAV Crop & Weed Monitoring
Cross-Scale UAV-Satellite Data Fusion (PES University Capstone PW26_YBS_02)
Features:
- Image Upload & Inference with sliding window patch reconstruction.
- Side-by-Side Segmentation & Class Overlay.
- Weed & Crop Coverage Analytics (% and physical m^2).
- Infestation Heatmap & Risk Zoning.
- Explainable AI (Grad-CAM for Crop vs. Weed).
- Uncertainty Estimation (Predictive Entropy via Monte Carlo Dropout).
- Model Performance Benchmarks & Generalization Reports.
- Relational SQLite History & Audit Logging.
"""

import os
import sys
import time
import io
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torchvision.transforms.functional as TF

from models.unet import build_unet
from src.predict import load_inference_model, predict_large_image, create_prediction_overlay, colorize_mask
from src.analytics import compute_field_coverage, generate_infestation_heatmap, generate_decision_support_report
from src.explainability import generate_xai_visualization
from src.uncertainty import estimate_uncertainty_mc_dropout, generate_uncertainty_visualization
from src.database import log_prediction, get_prediction_history, clear_prediction_history

st.set_page_config(
    page_title="CrossScale AgriFusion | UAV Crop-Weed Baseline",
    page_icon="🌱",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f3f4f6;
        border-radius: 8px;
        padding: 1rem;
        border-left: 5px solid #10b981;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    ckpt_path = os.path.join(PROJECT_ROOT, "outputs/checkpoints/unet_best.pth")
    if not os.path.exists(ckpt_path):
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dev = load_inference_model(ckpt_path, device=device)
    return model, dev

model, device = get_model()

# Header
st.markdown('<div class="main-header">🌱 Cross-Scale UAV–Satellite Data Fusion</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Based Precision Agriculture for Crop & Weed Monitoring | PES University (PW26_YBS_02)</div>', unsafe_allow_html=True)

# Tabs
tab_overview, tab_inference, tab_heatmaps, tab_xai, tab_uncertainty, tab_history, tab_benchmarks = st.tabs([
    "📋 Project Overview",
    "🔍 UAV Segmentation",
    "🗺️ Infestation Heatmaps",
    "💡 Explainable AI (Grad-CAM)",
    "📊 Uncertainty Estimation",
    "📜 History & Database Logs",
    "📈 Model Benchmarks"
])

# -------------------------------------------------------------
# TAB 1: OVERVIEW
# -------------------------------------------------------------
with tab_overview:
    st.subheader("Capstone Research Framework: The Scale Gap")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        **Academic Context:**
        - **Institution:** PES University, Department of CSE (AI & ML)
        - **Project ID:** `PW26_YBS_02` (Semester V, 2024–2028)
        - **Project Guide:** Prof. Yoga B S
        - **Team:** Arnav Choudhary, Vibhav M, Vainik, Manish Bala
        
        **The Problem Statement & Scale Gap:**
        Traditional precision agriculture relies either on high-resolution UAV imagery with limited spatial coverage or satellite imagery with large regional coverage but low spatial detail. This baseline provides the **centimetre-level UAV ground reference** for subsequent cross-scale fusion with Sentinel-2 satellite data in Phase-II.
        
        **Current Baseline Architecture:**
        - Primary Model: **ResNet18 U-Net** (14.33M parameters) with skip connections.
        - Training Dataset: **WeedMap** (MicaSense RedEdge-M UAV Flights over Sugar Beet).
        - Multi-Class Formulation: `0: Soil/Background`, `1: Sugar Beet Crop`, `2: Weed`.
        - Database Storage: **SQLite Relational Store** (`outputs/agrifusion.db`) for prediction history and field audit logs.
        - Modular Extension: Built-in `CrossScaleFusionAdapter` for satellite feature injection.
        """)
    with col2:
        if os.path.exists(os.path.join(PROJECT_ROOT, "outputs/reports/augmented_batch_sample.png")):
            st.image(os.path.join(PROJECT_ROOT, "outputs/reports/augmented_batch_sample.png"), caption="Sample Training Batch & Ground Truth Overlays", use_container_width=True)

# -------------------------------------------------------------
# TAB 2: INFERENCE
# -------------------------------------------------------------
with tab_inference:
    st.subheader("UAV Image Inference & Semantic Segmentation")
    
    col_up, col_ctrl = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader("Upload UAV RGB Image (.png, .jpg)", type=["png", "jpg", "jpeg"])
        use_sample = st.checkbox("Or use sample from held-out test split (Flight 003)", value=(uploaded_file is None))
        
    with col_ctrl:
        st.write("**Inference Parameters:**")
        patch_size = st.selectbox("Patch Size", [256], index=0)
        stride = st.slider("Sliding Window Stride", min_value=128, max_value=256, value=192, step=32)
        gsd_cm = st.number_input("Ground Sample Distance (cm/px)", value=1.0, step=0.1)

    image_pil = None
    image_name = "test_sample_003_frame0008.png"
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_name = uploaded_file.name
    elif use_sample:
        demo_dir = os.path.join(PROJECT_ROOT, "demo_samples")
        sample_options = {
            "Frame 0008 (High Weed Infestation)": "frame0008.png",
            "Frame 0076 (Prominent Crop Rows)": "frame0076.png",
            "Frame 0110 (Mixed Crop & Weed Canopy)": "frame0110.png",
            "Frame 0103 (Clustered Weed Patches)": "frame0103.png",
            "Frame 0144 (Moderate Mixed Field)": "frame0144.png"
        }
        chosen_label = st.selectbox("Choose Sample Test Tile from Flight 003:", list(sample_options.keys()), index=0)
        chosen_file = sample_options[chosen_label]
        sample_path = os.path.join(demo_dir, chosen_file)
        if not os.path.exists(sample_path):
            sample_path = os.path.join("C:/Users/Admin/Downloads/Tiles/Tiles/RedEdge/003/tile/RGB", chosen_file)
        if os.path.exists(sample_path):
            image_pil = Image.open(sample_path).convert("RGB")
            image_name = chosen_file
            st.caption(f"Loaded test sample: `{sample_path}`")
            
    if image_pil is not None and model is not None:
        if st.button("🚀 Run UAV Segmentation", type="primary"):
            with st.spinner("Executing sliding-window patch inference & reconstruction..."):
                t0 = time.time()
                pred_mask, prob_map = predict_large_image(model, image_pil, device, patch_size=patch_size, stride=stride)
                inference_time_ms = (time.time() - t0) * 1000
                
            img_np = np.array(image_pil)
            overlay, color_mask = create_prediction_overlay(img_np, pred_mask, alpha=0.5)
            coverage = compute_field_coverage(pred_mask, gsd_cm=gsd_cm)
            
            # Generate quick heatmap summary for database logging
            _, _, _, heatmap_summary = generate_infestation_heatmap(pred_mask, grid_size=16)
            decision_report = generate_decision_support_report(coverage, heatmap_summary)
            
            # Log to SQLite Database
            record_id = log_prediction(
                image_name=image_name,
                coverage=coverage,
                heatmap_summary=heatmap_summary,
                decision_report=decision_report,
                inference_time_ms=inference_time_ms
            )
            
            # Save in session state for other tabs
            st.session_state["image_pil"] = image_pil
            st.session_state["img_np"] = img_np
            st.session_state["pred_mask"] = pred_mask
            st.session_state["prob_map"] = prob_map
            st.session_state["coverage"] = coverage
            st.session_state["decision_report"] = decision_report
            
            st.success(f"Segmentation complete and logged to SQLite Database (Record #{record_id})")
            
            # Display Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Weed Coverage", f"{coverage['weed_coverage_pct']}%", f"{coverage['physical_weed_area_m2']} m²")
            m2.metric("Crop Coverage", f"{coverage['crop_coverage_pct']}%", f"{coverage['physical_crop_area_m2']} m²")
            m3.metric("Weed-to-Crop Ratio", f"{coverage['weed_to_crop_ratio']:.2f}")
            m4.metric("Inference Latency", f"{inference_time_ms:.1f} ms")
            
            # Side-by-side images
            c1, c2, c3 = st.columns(3)
            c1.image(img_np, caption="Original UAV RGB", use_container_width=True)
            c2.image(color_mask, caption="Predicted Mask (Green:Crop, Red:Weed)", use_container_width=True)
            c3.image(overlay, caption="Overlay on Field", use_container_width=True)
            
    elif model is None:
        st.warning("Model checkpoint not found. Please train the baseline model first.")

# -------------------------------------------------------------
# TAB 3: HEATMAPS & DECISION SUPPORT
# -------------------------------------------------------------
with tab_heatmaps:
    st.subheader("Spatial Weed Infestation Heatmaps & Risk Zoning")
    if "pred_mask" in st.session_state:
        pred_mask = st.session_state["pred_mask"]
        coverage = st.session_state["coverage"]
        
        c_grid, c_thresh = st.columns([1, 2])
        with c_grid:
            grid_res = st.slider("Spatial Grid Resolution", min_value=8, max_value=32, value=16, step=4)
        with c_thresh:
            low_t = st.number_input("Low Risk Threshold (%)", value=5.0)
            mod_t = st.number_input("Moderate Risk Threshold (%)", value=15.0)
            
        smooth_heatmap, density_map, risk_map, summary = generate_infestation_heatmap(
            pred_mask, grid_size=grid_res, thresholds={"low": low_t, "moderate": mod_t}
        )
        report = generate_decision_support_report(coverage, summary)
        
        # Display Decision Support Card
        st.markdown(f"""
        <div class="metric-card">
            <h4>📋 Agronomic Decision Support Assessment (Priority: {report['assessment_priority']})</h4>
            <p><strong>Recommendation:</strong> {report['recommended_action']}</p>
            <p><strong>High-Risk Hot Zones:</strong> {report['high_risk_zones']} field zones exceed economic threshold.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        
        ch1, ch2 = st.columns(2)
        with ch1:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(smooth_heatmap, cmap="jet")
            ax.set_title("Continuous Weed Density (%)")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close()
            
        with ch2:
            fig, ax = plt.subplots(figsize=(6, 5))
            try:
                cmap_risk = mpl.colormaps["YlOrRd"].resampled(3)
            except Exception:
                cmap_risk = plt.cm.get_cmap("YlOrRd", 3)
            im = ax.imshow(risk_map, cmap=cmap_risk, vmin=0, vmax=2)
            ax.set_title(f"Field Risk Zoning (High Risk: {summary['high_infestation_zones']} zones)")
            ax.axis("off")
            cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2], fraction=0.046, pad=0.04)
            cbar.ax.set_yticklabels(['Low (<5%)', 'Moderate (5-15%)', 'High (>15%)'])
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Run segmentation in the 'UAV Segmentation' tab first to generate spatial heatmaps.")

# -------------------------------------------------------------
# TAB 4: EXPLAINABLE AI
# -------------------------------------------------------------
with tab_xai:
    st.subheader("Explainable AI: Class-Specific Grad-CAM")
    st.markdown("Visualizes which visual spatial patterns the ResNet18 encoder prioritizes for the **Crop** class vs. the **Weed** class.")
    
    if "image_pil" in st.session_state and model is not None:
        if st.button("Generate Grad-CAM Visualizations", key="btn_xai"):
            img_pil = st.session_state["image_pil"].resize((256, 256))
            img_tensor = TF.normalize(TF.to_tensor(img_pil), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(device)
            img_display = np.array(img_pil)
            
            with st.spinner("Computing class activation maps..."):
                crop_cam, weed_cam, crop_ov, weed_ov = generate_xai_visualization(model, img_tensor, img_display)
                
            x1, x2, x3 = st.columns(3)
            x1.image(img_display, caption="Target UAV Patch (256x256)", use_container_width=True)
            x2.image(crop_ov, caption="Grad-CAM: Crop Class Attribution", use_container_width=True)
            x3.image(weed_ov, caption="Grad-CAM: Weed Class Attribution", use_container_width=True)
    else:
        st.info("Run segmentation in the 'UAV Segmentation' tab first to enable XAI visualizations.")

# -------------------------------------------------------------
# TAB 5: UNCERTAINTY ESTIMATION
# -------------------------------------------------------------
with tab_uncertainty:
    st.subheader("Predictive Uncertainty Estimation via Monte Carlo Dropout")
    st.markdown("Quantifies pixel-wise predictive entropy $H(p)$ across stochastic passes to identify ambiguous crop-weed boundaries.")
    
    if "image_pil" in st.session_state and model is not None:
        passes = st.slider("Monte Carlo Forward Passes", min_value=4, max_value=16, value=8, step=2)
        unc_threshold = st.slider("Uncertainty Decision Threshold", min_value=0.1, max_value=0.8, value=0.35, step=0.05)
        
        if st.button("Run Uncertainty Analysis", key="btn_unc"):
            img_pil = st.session_state["image_pil"].resize((256, 256))
            img_tensor = TF.normalize(TF.to_tensor(img_pil), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(device)
            img_display = np.array(img_pil)
            
            with st.spinner(f"Executing {passes} stochastic Monte Carlo passes..."):
                mean_p, entropy, variance, pred_m = estimate_uncertainty_mc_dropout(model, img_tensor, num_passes=passes)
                unc_summary = generate_uncertainty_visualization(img_display, pred_m, entropy, uncertainty_threshold=unc_threshold)
                
            u1, u2, u3 = st.columns(3)
            u1.metric("Mean Predictive Entropy", f"{unc_summary['mean_entropy']:.4f}")
            u2.metric("Max Entropy", f"{unc_summary['max_entropy']:.4f}")
            u3.metric("High-Uncertainty Boundary Pixels", f"{unc_summary['high_uncertainty_pixel_pct']}%")
            
            f1, f2 = st.columns(2)
            with f1:
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(entropy, cmap="plasma", vmin=0, vmax=1.0)
                ax.set_title("Predictive Entropy Heatmap [0, 1]")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig)
                plt.close()
                
            with f2:
                fig, ax = plt.subplots(figsize=(6, 5))
                unc_overlay = img_display.copy()
                unc_overlay[entropy > unc_threshold] = [255, 255, 0] # Yellow
                ax.imshow(unc_overlay)
                ax.set_title(f"Uncertain Boundaries (Yellow: {unc_summary['high_uncertainty_pixel_pct']}%)")
                ax.axis("off")
                st.pyplot(fig)
                plt.close()
    else:
        st.info("Run segmentation in the 'UAV Segmentation' tab first.")

# -------------------------------------------------------------
# TAB 6: PREDICTION HISTORY & DATABASE
# -------------------------------------------------------------
with tab_history:
    st.subheader("📜 Relational Prediction History & Field Audit Database")
    st.markdown("All inference runs and field coverage analyses are persisted in the **SQLite Database** (`outputs/agrifusion.db`).")
    
    col_hist_action, col_clear = st.columns([4, 1])
    with col_hist_action:
        df_history = get_prediction_history(limit=100)
        st.write(f"Total Logged Field Inspections: **{len(df_history)}**")
    with col_clear:
        if st.button("🗑️ Clear History"):
            clear_prediction_history()
            st.rerun()

    if len(df_history) > 0:
        st.dataframe(df_history, use_container_width=True)
        
        # Downloadable CSV report
        csv_buffer = io.StringIO()
        df_history.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Inspection Audit Report (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"agrifusion_field_audit_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No inference runs recorded yet. Upload and segment an image in the 'UAV Segmentation' tab to generate your first audit record.")

# -------------------------------------------------------------
# TAB 7: BENCHMARKS
# -------------------------------------------------------------
with tab_benchmarks:
    st.subheader("Model Benchmarks, Comparative Analytics & Generalization")
    
    comp_csv_path = os.path.join(PROJECT_ROOT, "outputs/reports/models_benchmark_comparison.csv")
    comp_chart_path = os.path.join(PROJECT_ROOT, "outputs/graphs/models_comparison_barchart.png")
    arch_summary_path = os.path.join(PROJECT_ROOT, "outputs/reports/model_architectures_summary.csv")
    
    if os.path.exists(comp_csv_path):
        st.markdown("#### 🏆 Multi-Model Benchmark Comparison (Held-Out Flight 003)")
        df_comp = pd.read_csv(comp_csv_path)
        st.dataframe(df_comp, use_container_width=True)
        
    if os.path.exists(comp_chart_path):
        st.image(comp_chart_path, caption="Figure: Held-Out Test Generalization Performance by Architecture", use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### 📈 Individual Model Training Dynamics & Loss Convergence")
    
    col_u, col_d, col_b = st.columns(3)
    with col_u:
        st.markdown("**Proposed ResNet18 U-Net**")
        u_curve = os.path.join(PROJECT_ROOT, "outputs/graphs/unet_training_curves.png")
        if os.path.exists(u_curve):
            st.image(u_curve, use_container_width=True)
    with col_d:
        st.markdown("**DeepLabV3+**")
        d_curve = os.path.join(PROJECT_ROOT, "outputs/graphs/deeplab_training_curves.png")
        if os.path.exists(d_curve):
            st.image(d_curve, use_container_width=True)
    with col_b:
        st.markdown("**Baseline CNN**")
        b_curve = os.path.join(PROJECT_ROOT, "outputs/graphs/baseline_cnn_training_curves.png")
        if os.path.exists(b_curve):
            st.image(b_curve, use_container_width=True)
            
    if os.path.exists(arch_summary_path):
        st.markdown("---")
        st.markdown("#### 🏗️ Architecture & Computational Specifications")
        st.dataframe(pd.read_csv(arch_summary_path), use_container_width=True)
