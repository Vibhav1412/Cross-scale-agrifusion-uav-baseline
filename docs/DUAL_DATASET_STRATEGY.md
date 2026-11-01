# Capstone Project: Cross-Scale UAV–Satellite Data Fusion
## Dual-Dataset Strategy: WeedMap + PhenoBench Integration & Cross-Domain Generalization

**Project ID:** PW26_YBS_02 | **Institution:** PES University  
**Focus:** Multi-Dataset Training, Cross-Domain Generalization, and Label Unification  

---

## 1. Research Motivation

In our Capstone Phase-I literature synthesis, **Research Gap 2** explicitly highlights:
> *"Limited cross-crop and cross-region validation. Existing approaches suffer from limited transferability across seasons, locations, sensor heights, and soil conditions."*

Using **both WeedMap and PhenoBench** directly transforms our UAV baseline from a single-dataset benchmark into a **cross-domain generalized framework**.

---

## 2. Label Taxonomy Harmonization

Both datasets monitor sugar beet fields, making their semantic concepts completely compatible. We unify them into a standardized 3-class scheme:

| Unified Class ID | Semantic Class | WeedMap Label | PhenoBench Label | Handling Strategy |
| :---: | :---: | :---: | :---: | :---: |
| **0** | **Soil / Background** | `0` (Background) | `0` (Soil) | Exact match |
| **1** | **Crop (Sugar Beet)** | `1` (Crop) | `1` (Crop) + `3` (Partial crop) | Direct map (class 3 merged or masked) |
| **2** | **Weed** | `2` (Weed) | `2` (Weed) + `4` (Partial weed) | Direct map (class 4 merged or masked) |

---

## 3. Staged Development Workflow

### Stage 1: Initial Baseline with WeedMap (Current Step)
- **Why First?**
  1. Faster iteration and lightweight download footprint.
  2. Ground-truth georeferencing (UTM/WGS84 orthomosaics) directly establishes the geospatial anchor for Phase-II Sentinel-2 satellite fusion.
  3. Clean 3-class segmentation without partial-visibility ambiguities.
- **Tasks:** Build and verify U-Net, DeepLabV3+, Basic CNN; establish coverage calculation, heatmaps, XAI, and uncertainty estimation.

### Stage 2: Scaling and Cross-Domain Generalization with PhenoBench
- **Why Second?**
  1. Expands training data with 2,179 ultra-high-resolution tiles ($1024 \times 1024$).
  2. Enables **Zero-Shot Domain Transfer Testing**:
     - *Experiment A:* Model trained on WeedMap $\rightarrow$ Evaluated on PhenoBench (unseen sensor, unseen field).
     - *Experiment B:* Model trained on PhenoBench $\rightarrow$ Evaluated on WeedMap.
  3. **Combined Multi-Domain Training**: Train a single unified model on both datasets simultaneously to evaluate whether cross-sensor diversity improves segmentation robustness on complex field boundaries.

---

## 4. Technical Resolution Alignment

- **Input Dimension Standardization:**
  - WeedMap tiles ($480 \times 360$) and PhenoBench tiles ($1024 \times 1024$) are both processed through a standardized patch pipeline (e.g., $256 \times 256$ or $512 \times 512$ random/sliding crops).
- **Sensor Invariance:**
  - Robust color jitter (brightness, contrast, hue, saturation) and normalization ensure the model learns leaf morphology and spatial canopy structure rather than camera-specific radiometric artifacts.

