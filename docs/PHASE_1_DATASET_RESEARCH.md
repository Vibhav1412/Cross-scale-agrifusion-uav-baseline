# Capstone Project: Cross-Scale UAV–Satellite Data Fusion
## Phase 1: UAV Agricultural Dataset Research, Evaluation & Selection Report

**Academic Context:**
- **Institution:** PES University, Department of Computer Science and Engineering (AI & ML)
- **Course:** Capstone Project (Phase-I, Sem V, 2024–2028)
- **Project ID:** PW26_YBS_02
- **Project Title:** Cross-Scale UAV–Satellite Data Fusion: AI-Based Precision Agriculture for Crop Monitoring
- **Project Guide:** Prof. Yoga B S
- **Team Members:**
  - Arnav Choudhary (PES1UG24AM050)
  - Vibhav M (PES1UG24CS527)
  - Vainik (PES1UG24CS513)
  - Manish Bala (PES1UG24CS263)

---

## 1. Executive Summary & Objective

The primary objective of the overall capstone project is to bridge the "Scale Gap" in precision agriculture: connecting high-resolution, centimetre-scale Unmanned Aerial Vehicle (UAV) imagery (2–10 cm ground resolution) with scalable, multi-temporal satellite imagery (e.g., Sentinel-2 at 10–30 m resolution). 

For the current development phase (Phase-I), we are constructing a complete, modular, and reproducible **UAV-only semantic segmentation baseline** for crop and weed monitoring. This baseline will later serve as the ground reference and local-scale component for the cross-scale fusion framework.

This report independently evaluates publicly available UAV agricultural datasets against 13 rigorous criteria to determine the most viable, research-grade dataset for our baseline implementation.

---

## 2. In-Depth Candidate Dataset Profiles

### Candidate 1: PhenoBench (University of Bonn)

1. **Dataset Name:** PhenoBench: A Large Dataset and Benchmarks for Semantic Image Interpretation in the Field
2. **Official Download / Source Link:** 
   - Official Portal: [https://www.phenobench.org/](https://www.phenobench.org/)
   - Benchmark Platform: [https://www.codabench.org/competitions/1066/](https://www.codabench.org/competitions/1066/)
   - Code & Devkit: [https://github.com/PRBonn/phenobench](https://github.com/PRBonn/phenobench)
3. **Publication & Source Information:** 
   - J. Weyler, F. Magistri, P. Roggiolani, R. Sodhi, C. Stachniss, "PhenoBench: A Large Dataset and Benchmarks for Semantic Image Interpretation in the Field", *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023.
4. **Number of Images:** 
   - Total of 2,960 high-resolution image crops ($1024 \times 1024$ pixels) extracted from 100-megapixel global UAV orthomosaics.
   - Standard split: 1,407 Training images, 772 Validation images, 781 Test images (withheld labels).
5. **Image Resolution:** $1024 \times 1024$ pixels per patch (Ground Sampling Distance ~0.5–1 cm/pixel).
6. **Image Type:** High-resolution RGB images.
7. **Available Annotation Types:** 
   - Pixel-level semantic segmentation masks.
   - Plant-level instance segmentation masks.
   - Leaf-level instance segmentation masks (>30,000 annotated crop leaves).
8. **Number of Classes:** 5 semantic classes:
   - `0`: Soil / Background
   - `1`: Crop (Sugar Beet)
   - `2`: Weed
   - `3`: Partial-crop (<50% visible plant)
   - `4`: Partial-weed (<50% visible weed)
9. **Segmentation Masks Available:** Yes, complete pixel-wise ground truth `.png` masks for Train and Validation sets.
10. **Metadata Availability:** 
    - Temporal / Growth Stage metadata: Collected across multiple days and years (2020–2022) spanning emergence to closed canopy.
    - Spatial / Field metadata: Contains multiple distinct fields; validation set includes images from fields unseen during training.
11. **License & Restrictions:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0). Completely open for research.
12. **Suitability for Deep Learning:** Excellent. Purpose-built for modern deep learning architectures (U-Net, DeepLabV3+, Mask2Former). $1024 \times 1024$ tiles easily downsampled or cropped to $256 \times 256$ or $512 \times 512$ for efficient GPU/MPS training.
13. **Limitations:** 
    - Test set masks are hidden on Codabench (requires submitting online, or locally carving out a portion of the 772 validation images for local test evaluation).
    - Archive size is relatively large (~7.6 GB for `PhenoBench-v110.zip`).
    - Does not provide geo-referenced TIFF orthomosaics directly out-of-the-box (patches are non-georeferenced RGB crops).

---

### Candidate 2: WeedMap (ETH Zurich & University of Bonn)

1. **Dataset Name:** WeedMap: A Large-Scale Semantic Weed Mapping Dataset
2. **Official Download / Source Link:** 
   - ETH Zurich Project Portal: [https://projects.asl.ethz.ch/datasets/doku.php?id=weedmap](https://projects.asl.ethz.ch/datasets/doku.php?id=weedmap)
   - ETH Research Collection (Direct DOI): [https://doi.org/10.3929/ethz-c-000788571](https://doi.org/10.3929/ethz-c-000788571)
   - GitHub Repository: [https://github.com/viariasv/weedMap](https://github.com/viariasv/weedMap)
3. **Publication & Source Information:** 
   - I. Sa, M. Popović, R. Khanna, Z. Chen, P. Lottes, F. Liebisch, J. Nieto, C. Stachniss, A. Walter, R. Siegwart, "weedMap: A Large-Scale Semantic Weed Mapping Framework Using Aerial Multispectral Imaging and Deep Neural Network for Precision Farming", *Remote Sensing (MDPI)*, 10(9), 1423, 2018.
4. **Number of Images:** 
   - 8 comprehensive flight datasets (numbered `000` to `007`) spanning sugar beet and sunflower fields.
   - Contains both full-field georeferenced orthomosaics and hundreds of pre-tiled image crops per dataset (typically $480 \times 360$ px tiles, totaling >1,200 tiles across sets).
5. **Image Resolution:** 
   - Tiled crops: $480 \times 360$ pixels.
   - Full orthomosaics: Up to tens of thousands of pixels with a Ground Sampling Distance (GSD) of ~1.0–1.5 cm/pixel.
6. **Image Type:** 
   - Multispectral (MicaSense RedEdge-M 5-band: Blue, Green, Red, RedEdge, NIR; Parrot Sequoia 4-band: Green, Red, RedEdge, NIR).
   - Readily composited into standard 3-channel RGB imagery.
7. **Available Annotation Types:** Pixel-level semantic segmentation masks.
8. **Number of Classes:** 3 core classes:
   - `0`: Background (Soil, non-vegetation)
   - `1`: Crop (Sugar beet or Sunflower)
   - `2`: Weed
9. **Segmentation Masks Available:** Yes, dense pixel-wise ground-truth label maps matching every orthomosaic and pre-extracted tile.
10. **Metadata Availability:** 
    - Full spatial metadata: Georeferenced coordinates (UTM / WGS84) for entire field orthomosaics.
    - Location metadata: Multiple experimental field sites in Eschikon (Switzerland) and Bonn (Germany).
    - Sensor calibration logs and flight trajectory coordinates available.
11. **License & Restrictions:** Creative Commons Attribution 4.0 International (CC BY 4.0). Completely free for research.
12. **Suitability for Deep Learning:** Outstanding. Tiled datasets are lightweight, directly compatible with PyTorch `Dataset` loaders, and standard 3-class segmentation aligns 1:1 with BCE/Dice loss formulations.
13. **Limitations:** 
    - Pre-extracted tiles are $480 \times 360$ (not square powers-of-two, requiring resizing or padding to $256 \times 256$ or $512 \times 512$).
    - Handling raw multi-channel GeoTIFFs requires GDAL / Rasterio if working from full orthomosaics rather than pre-cut tiles.

---

### Candidate 3: WeedsGalore (GFZ German Research Centre & Uni Bonn)

1. **Dataset Name:** WeedsGalore: A Multispectral and Multitemporal UAV-based Dataset for Crop and Weed Segmentation
2. **Official Download / Source Link:** 
   - GitHub Repository: [https://github.com/GFZ/weedsgalore](https://github.com/GFZ/weedsgalore)
   - CVF Open Access / arXiv: [https://arxiv.org/abs/2410.02422](https://arxiv.org/abs/2410.02422)
3. **Publication & Source Information:** 
   - E. Celikkan et al., "WeedsGalore: A Multispectral and Multitemporal UAV-based Dataset for Crop and Weed Segmentation in Agricultural Maize Fields", *IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 2025.
4. **Number of Images:** 
   - 35 full-field orthomosaics across different flights.
   - Core tiled dataset contains hundreds of $600 \times 600$ pixel tiles cropped with structured overlaps.
5. **Image Resolution:** $600 \times 600$ pixels per tile (GSD ~1.2 cm/pixel).
6. **Image Type:** 5-band multispectral (Red, Green, Blue, RedEdge, NIR) with RGB extraction supported.
7. **Available Annotation Types:** Pixel-level semantic segmentation and instance segmentation masks.
8. **Number of Classes:** Multi-class weed taxonomy:
   - Soil / Background
   - Maize (target crop)
   - Specific weed species: Common amaranth (*Amaranthus retroflexus*), Barnyard grass (*Echinochloa crus-galli*), Quickweed (*Galinsoga parviflora*), and generic "other weeds".
9. **Segmentation Masks Available:** Yes, pixel-level annotation masks provided for all tiles and full fields.
10. **Metadata Availability:** 
    - Multi-temporal flight dates spanning early to late vegetative maize growth.
    - Geospatial coordinates (GeoTIFF) available for field orthomosaics.
11. **License & Restrictions:** Creative Commons Attribution (CC BY). Free for academic and research use.
12. **Suitability for Deep Learning:** Very high. Clean GitHub tooling, modern PyTorch loaders provided by authors.
13. **Limitations:** 
    - Severe class imbalance across specialized weed species (requires complex loss reweighting or collapsing into a binary weed class).
    - Very recent release (WACV 2025), so fewer established external open-source benchmarks exist compared to PhenoBench or WeedMap.

---

### Candidate 4: CoFly-WeedDB (CERTH-ITI & DUTH)

1. **Dataset Name:** CoFly-WeedDB: A UAV Image Dataset for Weed Detection and Species Identification
2. **Official Download / Source Link:** 
   - Zenodo Repository: [https://doi.org/10.5281/zenodo.6697343](https://doi.org/10.5281/zenodo.6697343)
   - GitHub Repository: [https://github.com/CoFly-Project/CoFly-WeedDB](https://github.com/CoFly-Project/CoFly-WeedDB)
3. **Publication & Source Information:** 
   - M. Krestenitis et al., "CoFly-WeedDB: A UAV image dataset for weed detection and species identification", *Remote Sensing*, 2022.
4. **Number of Images:** 201 fully annotated high-resolution images.
5. **Image Resolution:** $1280 \times 720$ pixels.
6. **Image Type:** RGB imagery (captured with DJI Phantom 4 Pro, 20 MP sensor at 5 m altitude).
7. **Available Annotation Types:** Polygon annotations and 1D pixel masks (`labels_1d/`).
8. **Number of Classes:** 4 classes: Background (`0`), Johnson grass (`1`), Purslane (`2`), Field bindweed (`3`).
9. **Segmentation Masks Available:** Yes, exact single-channel PNG segmentation masks.
10. **Metadata Availability:** UAV telemetry (altitude, speed, camera angle), but limited to a single cotton field.
11. **License & Restrictions:** Creative Commons Attribution 4.0 International (CC BY 4.0).
12. **Suitability for Deep Learning:** Moderate. Fast download (~436 MB), ready-to-train 1D masks.
13. **Limitations:** Only 201 images (too small for robust deep learning without artificial augmentation); crop (cotton) is bundled into background rather than segmented as a distinct crop class; no multi-field or cross-season flight data.

---

## 3. Systematic Dataset Comparison Matrix

| Evaluation Criteria | **Candidate 1: PhenoBench** | **Candidate 2: WeedMap** | **Candidate 3: WeedsGalore** | **Candidate 4: CoFly-WeedDB** |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Target Crop** | Sugar Beet | Sugar Beet & Sunflower | Maize (Corn) | Cotton |
| **Spectral Channels** | RGB (Ultra-high res) | RGB + Multispectral (5-band) | RGB + Multispectral (5-band) | RGB |
| **Total Labeled Units** | 2,179 labeled tiles ($1024 \times 1024$) | >1,200 labeled tiles + 8 full orthomosaics | >1,000 tiles + 35 full orthomosaics | 201 annotated images ($1280 \times 720$) |
| **Target Classes** | Crop, Weed, Soil, Partial Crop/Weed | Crop, Weed, Background | Crop, 4 Weed Classes, Soil | 3 Weed species, Background |
| **Crop vs. Weed Separation?** | **Yes (Direct)** | **Yes (Direct)** | **Yes (Direct)** | No (Crop merged in background) |
| **Georeferencing / Coordinates?** | No (Isolated crops) | **Yes (Full UTM / WGS84)** | **Yes (Full GeoTIFF)** | Limited (GPS flight metadata only) |
| **Multi-Field Diversity?** | Yes (Separate test fields) | Yes (Swiss & German fields) | Yes (Multiple field plots) | No (Single field) |
| **Multi-Temporal / Multi-Flight?**| Yes (Multiple growth stages) | Yes (8 separate flights) | Yes (Multi-temporal timeline) | No (Single early-growth flight) |
| **Download Footprint** | ~7.6 GB | ~5.36 GB (all 8 datasets) | ~0.4 GB (core) / 12 GB (full) | ~436 MB |
| **License** | CC BY-SA 4.0 | CC BY 4.0 | CC BY | CC BY 4.0 |
| **Readiness for U-Net / DeepLab** | **Immediate (PyTorch ready)**| **Immediate (PyTorch ready)** | High (PyTorch ready) | Moderate (Requires patch tiling) |
| **Relevance to Phase-II Satellite Fusion** | Medium (Lacks geocoordinates)| **Highest (Pre-georeferenced)** | High (Pre-georeferenced) | Low (No coordinate grid) |

---

## 4. Final Recommendation & Strategic Justification

### Recommended Primary Dataset: **WeedMap (ETH Zurich / Uni Bonn)**
*Alternative Recommended Option (Pure RGB Focus): **PhenoBench** (Uni Bonn)*

### Why WeedMap is the Best Fit for this Capstone:

1. **Direct Alignment with Capstone Novelty & Roadmap (Slides 4, 11, 14):**
   Our overarching capstone is *"Cross-Scale UAV–Satellite Data Fusion"*. In Phase-II, we must align UAV orthomosaics with Sentinel-2 satellite pixels over identical geographic regions. **WeedMap explicitly provides georeferenced orthomosaics with calibrated UTM coordinates** alongside pre-cut tiles. This means the baseline we build now will directly port into the cross-scale fusion pipeline without having to synthetically manufacture geospatial coordinates.

2. **Standard 3-Class Formulation (Soil, Crop, Weed):**
   WeedMap provides clean, unambiguous 3-class segmentation:
   - Class 0: Background / Soil
   - Class 1: Crop (Sugar Beet / Sunflower)
   - Class 2: Weed
   This perfectly satisfies Phase 2 of our project (crop vs weed coverage metrics, infestation heatmaps).

3. **Multi-Field and Multi-Flight Generalization (Slide 16):**
   With 8 separate datasets (`000` through `007`) spanning different fields in Switzerland (Eschikon) and Germany (Bonn), WeedMap enables true **Cross-Field Generalization Testing** (training on sets `000`–`004` and evaluating on unseen field sets `005`–`007`), addressing Research Gap 2 identified in our Phase-I presentation.

4. **RGB + Multispectral Flexibility:**
   While we will train our Phase-I baseline strictly on the 3-channel RGB imagery, having the RedEdge and NIR channels available provides a natural bridge to satellite spectral indices (e.g. NDVI, NDRE) in Phase-II.

---

## 5. Next Steps for Dataset Acquisition

To proceed with Step 3 (Inspection and Preprocessing), the user can download either:
1. **Option A (Recommended for Full Capstone Pipeline): WeedMap**
   - Download link: [ETH Research Collection DOI: 10.3929/ethz-c-000788571](https://doi.org/10.3929/ethz-c-000788571)
   - Specifically the pre-tiled RGB/multispectral datasets (e.g., RedEdge or Sequoia tiles).
2. **Option B (Recommended for High-Res Pure RGB Benchmark): PhenoBench**
   - Download link: [https://www.phenobench.org/](https://www.phenobench.org/) (`PhenoBench-v110.zip`, ~7.6 GB).

No model implementation will begin until the user confirms the dataset selection and provides the downloaded data directory structure.

