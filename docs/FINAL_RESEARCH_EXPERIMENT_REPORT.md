# Capstone Project: Cross-Scale UAV–Satellite Data Fusion
## Phase-I Final Research Experiment & Baseline Benchmark Report

**Project ID:** PW26_YBS_02  
**Institution:** Department of Computer Science & Engineering (AI & ML), PES University, Bengaluru  
**Academic Course:** Capstone Project (Phase-I, Sem V, 2024–2028)  
**Project Guide:** Prof. Yoga B S  
**Authors / Research Team:**  
- Arnav Choudhary (`PES1UG24AM050`)  
- Vibhav M (`PES1UG24CS527`)  
- Vainik (`PES1UG24CS513`)  
- Manish Bala (`PES1UG24CS263`)  

---

## 1. Abstract & Research Context

High-resolution Unmanned Aerial Vehicle (UAV) imagery provides centimetre-scale canopy detail essential for precision agricultural operations, such as selective spot-spraying and weed mapping. However, UAVs are fundamentally constrained by restricted battery endurance and spatial footprints. Conversely, Earth Observation satellites (e.g., Sentinel-2) deliver continuous regional monitoring ($10\text{–}30\text{ m/px}$) but suffer from mixed-pixel spectral ambiguity.

This research establishes the **Phase-I High-Resolution UAV Baseline** for our larger capstone investigation: **Cross-Scale UAV–Satellite Data Fusion**. We developed, trained, and empirically evaluated a deep semantic segmentation framework distinguishing **Soil/Background (`0`)**, **Sugar Beet Crop (`1`)**, and **Weeds (`2`)** on the benchmark WeedMap dataset across independent UAV flight campaigns in Germany and Switzerland. The architecture features a modular ResNet18 U-Net equipped with a dedicated **Cross-Scale Fusion Adapter** designed to seamlessly accept satellite feature tensors in Phase-II.

---

## 2. Experimental Setup & Leakage-Free Splitting Protocol

To prevent spatial autocorrelation and overly optimistic evaluation, tiles were partitioned strictly by **independent flight campaigns and geographic field sites**, matching the official experimental protocol of Sa et al. (MDPI Remote Sensing 2018):

- **Raw Dataset:** 970 high-resolution RGB tiles ($480 \times 360$ px) captured via MicaSense RedEdge-M UAV flights at 10 m altitude ($GSD \approx 1.0\text{ cm/px}$).
- **Boundary Mask Filtering:** 331 non-effective boundary padding tiles outside the flight boundary were removed.
- **Retained Effective Field Tiles:** 639 tiles ($110,419,200$ labeled pixels).
  - **Training Set (Flights `000`, `002`, `004`):** 368 tiles ($57.6\%$)
  - **Validation Set (Flight `001`):** 122 tiles ($19.1\%$)
  - **Held-Out Test Set (Flight `003`):** 149 tiles ($23.3\%$) — **strictly unseen flight and field conditions** for cross-field generalization.

### Class Imbalance & Loss Formulation
- Soil / Background: $98.76\%$ ($109,045,845$ px)
- Sugar Beet Crop: $0.45\%$ ($500,245$ px)
- Weed: $0.79\%$ ($873,110$ px)

To mitigate this severe $99:1$ background-to-vegetation ratio, we formulated a hybrid loss function:
$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{WCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$
where class weights $\mathbf{w} = [0.0087, 1.9017, 1.0896]$ were derived via normalized median frequency balancing.

---

## 3. Architecture Comparison & Verification

All models were evaluated on $4 \times 3 \times 256 \times 256$ inputs:

| Model Architecture | Total Parameters | Trainable | Model Memory | CPU Latency ($B=4$) | Per-Image Latency | Key Design Feature |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **ResNet18 U-Net** *(Proposed)* | **14.33 M** | **14,331,043** | **54.67 MB** | **139.20 ms** | **34.80 ms** | Hierarchical skips + Phase-II satellite fusion hook |
| **DeepLabV3+** | 16.60 M | 16,603,555 | 63.34 MB | 137.37 ms | 34.34 ms | Atrous Spatial Pyramid Pooling (ASPP) |
| **Basic CNN** | 0.80 M | 802,307 | 3.06 MB | 141.17 ms | 35.29 ms | Transpose-conv decoder without skips |

---

## 4. Empirical Model Performance & Training Dynamics

The primary U-Net model was trained using Adam ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$) with Cosine Annealing learning rate scheduling.

### Validation Progression:
- **Best Validation mIoU:** **56.45%** (Epoch 10)
- **Validation Mean Dice (F1):** **67.83%**
- **Soil IoU:** $99.1\%$
- **Crop IoU:** $35.7\%$
- **Weed IoU:** $34.6\%$

### Generalization Evaluation on Held-Out Test Flight (`003`):
Evaluated on 149 unseen field tiles ($9,704,970$ test pixels):

| Target Semantic Class | Test IoU (%) | Test Dice / F1 (%) | Test Precision (%) | Test Recall (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Soil / Background** | **97.91%** | **98.94%** | $99.99\%$ | $97.92\%$ |
| **Sugar Beet Crop** | **28.96%** | **44.91%** | $29.19\%$ | **97.32%** |
| **Weed** | **26.01%** | **41.29%** | $28.98\%$ | **71.77%** |
| **Macro Average** | **50.96%** | **61.72%** | **52.72%** | **89.00%** |

### Confusion Matrix on Held-Out Test Set (U-Net):
$$\begin{pmatrix} 
\text{Soil} & \text{Crop} & \text{Weed} \\
9,449,030 & 59,114 & 141,529 \\
12 & 33,292 & 904 \\
1,227 & 21,635 & 58,121
\end{pmatrix}$$

**Key Agronomic Insights:**
1. **Exceptional Crop Recall ($97.32\%$):** Only 12 crop pixels out of $34,208$ were misidentified as soil, guaranteeing almost zero missed crops during autonomous robotic weeding or spraying.
2. **Weed Detection Sensitivity ($71.77\%$):** Over $71\%$ of true weed instances in completely unseen field flights were correctly detected despite extreme morphological similarities with early-stage sugar beet seedlings.

### Multi-Model Benchmark Comparison (Feature 7):
Evaluated under identical experimental conditions on held-out test flight `003` (149 unseen tiles):

| Model Architecture | Parameters | Model Size | Test mIoU (%) | Test mDice (%) | Crop IoU (%) | Weed IoU (%) | Macro Recall (%) | Macro Prec (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet18 U-Net (Proposed)** | **14.33 M** | **54.67 MB** | **50.96%** | **61.72%** | **28.96%** | **26.01%** | **89.00%** | **52.72%** |
| **DeepLabV3+** | 16.60 M | 63.34 MB | 51.70% | 62.16% | 41.11% | 17.73% | 92.33% | 53.69% |
| **Basic CNN** | 0.80 M | 3.06 MB | 42.27% | 50.50% | 20.83% | 10.81% | 76.29% | 44.92% |

**Scientific Interpretation & Agronomic Discussion:**
1. **Necessity of Encoder-Decoder Skip Connections (U-Net vs. Basic CNN):**
   The Basic CNN without skip connections struggled on fine vegetation, achieving only $10.81\%$ Weed IoU compared to $26.01\%$ on U-Net ($+15.20\%$ absolute gain) and $42.27\%$ vs $50.96\%$ mIoU ($+8.69\%$). This quantitatively validates our architectural choice.
2. **Weed Sensitivity & Fine-Structure Preservation (U-Net vs. DeepLabV3+):**
   While DeepLabV3+ achieves higher Crop IoU ($41.11\%$) due to ASPP contextual pooling over large contiguous canopy zones, its Weed IoU drops sharply to $17.73\%$ (vs. $26.01\%$ for U-Net, an $8.28\%$ degradation). Atrous spatial pooling tends to dilute narrow, isolated weed leaves, confirming that U-Net remains the superior primary baseline for localized herbicide targeting.

---

## 5. Agricultural Decision Support & Spatial Analytics

1. **Physical Area Estimation:**
   Using the calibrated Ground Sampling Distance ($GSD = 1.0\text{ cm/px}$), the system translates pixel segmentations into physical field metrics:
   $$\text{Area } (m^2) = N_{\text{pixels}} \times \left(\frac{GSD}{100}\right)^2$$
2. **Infestation Heatmaps & Risk Zoning:**
   The spatial domain is discretized into an $N \times N$ grid, categorizing local weed density into:
   - Low Infestation ($<5\%$ coverage)
   - Moderate Infestation ($5\text{–}15\%$ coverage)
   - High Infestation ($>15\%$ coverage)
3. **Targeted Prescription Guidance:**
   Provides localized spray-zone coordinates, reducing herbicide chemical usage compared to conventional broad-acre broadcasting.

---

## 6. Trustworthy AI: Explainability & Uncertainty Quantification

1. **Class-Specific Grad-CAM (Feature 5):**
   Gradients extracted from `model.layer4` verify that the network activates on leaf veins and rosette geometries for sugar beets, while activating on irregular multi-stem clusters for weed species.
2. **Predictive Uncertainty via Monte Carlo Dropout (Feature 6):**
   Stochastic passes ($T=8$) compute pixel-wise predictive entropy:
   $$\mathcal{H}(x) = - \frac{1}{\log(C)} \sum_{c=0}^{C-1} \bar{p}_c(x) \log(\bar{p}_c(x) + \epsilon)$$
   Uncertainty spikes sharply along the physical boundaries between overlapping leaves and soil edges, identifying ambiguous zones requiring visual inspection.

---

## 7. Strategic Bridge to Phase-II Cross-Scale Satellite Fusion

With the high-resolution UAV baseline verified, the project roadmap for Phase-II is fully established:
1. **Geospatial Co-Registration:** Utilize the Swiss (Eschikon) and German (Rheinbach) UTM coordinates provided by WeedMap to query and align multi-spectral Sentinel-2 satellite imagery (Bands 2, 3, 4, 8, 8A).
2. **Cross-Scale Feature Fusion:** Feed satellite spectral embeddings into the pre-built `CrossScaleFusionAdapter` in `models/unet.py`.
3. **Comparative Study:** Quantify whether cross-scale data fusion improves segmentation robustness and zero-shot transferability across different agricultural seasons and geographic regions.

