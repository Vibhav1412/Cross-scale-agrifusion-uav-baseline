# Capstone Project: Cross-Scale UAV–Satellite Data Fusion
## Phase 4: Data Preprocessing, Leakage-Free Splitting & DataLoader Verification

**Project ID:** PW26_YBS_02 | **Institution:** PES University | **Department:** B.Tech CSE (AI & ML)  
**Dataset:** WeedMap (MicaSense RedEdge-M UAV Flights)  

---

## 1. Flight-Level Leakage-Free Splitting Protocol

To prevent spatial autocorrelation and data leakage between neighboring spatial patches, tiles are grouped strictly by **independent UAV flight campaigns and field dates**, matching the official experimental protocol established in the WeedMap research paper (Sa et al., MDPI Remote Sensing 2018):

| Split Manifest | Source Subsets | Raw Tiles | Retained Effective Tiles | Role & Evaluation Purpose |
| :---: | :---: | :---: | :---: | :---: |
| **`train.csv`** | `000`, `002`, `004` | 590 | **368** | Primary baseline model training |
| **`val.csv`** | `001` | 176 | **122** | Checkpoint selection, hyperparameter tuning & early stopping |
| **`test.csv`** | `003` | 204 | **149** | **Held-Out Cross-Field Generalization Test** (Unseen flight) |
| **Total** | `000`–`004` | 970 | **639** | Filtered out 331 empty/boundary border padding tiles |

---

## 2. Empirical Class Distribution & Imbalance Analysis

Across all 639 effective field tiles ($110,419,200$ labeled pixels):

| Class ID | Semantic Class | Pixel Count | Area Coverage (%) | Normalized Loss Weight ($\sum = 3.0$) |
| :---: | :---: | :---: | :---: | :---: |
| **0** | **Soil / Background** | $109,045,845$ | **98.76%** | $0.0087$ |
| **1** | **Crop (Sugar Beet)** | $500,245$ | **0.45%** | $1.9017$ |
| **2** | **Weed** | $873,110$ | **0.79%** | $1.0896$ |

### Class Imbalance Handling:
- The severe imbalance (vegetation constitutes ~1.24% of the total pixel area) reflects realistic early-season field conditions.
- To counter this, our training pipeline combines:
  1. **Weighted Cross-Entropy Loss** utilizing the computed median frequency balancing weights.
  2. **Multi-Class Soft Dice Loss** ($0.5 \times \text{CE} + 0.5 \times \text{Dice}$) to ensure gradients remain sensitive to small weed and crop clusters.

---

## 3. Synchronized Data Augmentation Pipeline

Implemented in `src/augmentation.py`:
- **Spatial Transforms (Synchronized):**
  - Random $256 \times 256$ crop for training; deterministic center crop for validation/testing.
  - Random horizontal flip ($p = 0.5$).
  - Random vertical flip ($p = 0.5$).
  - Random orthogonal rotation ($90^\circ, 180^\circ, 270^\circ$, $p = 0.5$).
- **Mask Integrity Guarantee:**
  - Nearest-neighbor interpolation is strictly enforced for masks to guarantee no fractional label interpolation occurs between classes $\{0, 1, 2\}$.
- **Photometric Transforms (Radiometric Invariance):**
  - Brightness adjustment ($\pm 20\%$) and contrast adjustment ($\pm 20\%$) applied strictly to the 3-channel RGB image.
- **Normalization:** ImageNet mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`).

---

## 4. DataLoader Pipeline Verification

The unit test script `src/test_dataloader.py` verified:
- **Image Tensor:** `torch.Size([8, 3, 256, 256])`, `dtype=torch.float32`.
- **Mask Tensor:** `torch.Size([8, 256, 256])`, `dtype=torch.int64`.
- **Label Integrity:** Unique values in all batches $\subseteq \{0, 1, 2\}$.
- **Batch Throughput:** $133.31\text{ ms}$ per 8-image batch on CPU.
- **Visual Verification Output:** [`outputs/reports/augmented_batch_sample.png`](outputs/reports/augmented_batch_sample.png).

