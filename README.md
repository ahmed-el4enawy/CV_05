# CV_05: Face Detection & Recognition (PCA / Eigenfaces)

**Team number:** 12
**Team members:**
* Ahmed Salah Geoshy Elshenawy
* Abdullah Mohammad Khalifa
* Mohamed Elsayed Attallah

---

## Executive Summary

This project implements a complete face detection and recognition pipeline using PCA (Principal Component Analysis) / Eigenfaces, built entirely from scratch in **C++17**. The system supports face detection on both color and grayscale images, trains eigenface models on standard face datasets, performs recognition via nearest-neighbor in eigenface space, and evaluates performance with ROC curves and comprehensive metrics. Legacy feature detection (Harris, SIFT, matching) from CV_03 is preserved. The engine is integrated into a modern Django web interface via `pybind11`.

---

## Table of Contents

1. [Tasks Implemented](#1-tasks-implemented)
2. [System Architecture](#2-system-architecture)
3. [Installation](#3-installation)
4. [Usage Instructions](#4-usage-instructions)
5. [Technical Specifications](#5-technical-specifications)
6. [File Descriptions](#6-file-descriptions)

---

## 1. Tasks Implemented

### A) Face Detection (from scratch)
* **Color images:** Skin-color segmentation in YCbCr color space with morphological cleanup (erosion, dilation, opening, closing) and connected component labeling for bounding box extraction.
* **Grayscale images:** Edge-density sliding window with symmetry analysis using integral images.
* Both paths include non-maximum suppression (IoU-based) on bounding boxes.

### B) Face Recognition — PCA / Eigenfaces (from scratch)
* **Training:** Flatten face images → compute mean face → center data → covariance trick (AᵀA) → Jacobi eigendecomposition → extract top-k eigenfaces → project training faces.
* **Recognition:** Project test face into eigenspace → L2 nearest-neighbor → threshold-based acceptance/rejection.
* **Preprocessing:** Histogram equalization, bilinear resize to standard dimensions.

### C) Performance Evaluation
* **ROC Curve:** Sweep distance threshold, compute TPR/FPR, plot with AUC.
* **Metrics:** Rank-1 accuracy, Equal Error Rate (EER), confusion matrix.
* **Visualization:** Publication-quality plots via matplotlib.

### D) Legacy Features (from CV_03)
* Harris Corner Detection, λ⁻ Corner Detection
* SIFT Feature Descriptors (full pipeline)
* Feature Matching (SSD + NCC)

---

## 2. System Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                  Web Interface (Browser)                        │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ 6 Tab Layout   │  │ Control Panel│  │ Results Dashboard  │  │
│  │ Face/Recog/ROC │  │ (Params)     │  │ (Images + Metrics) │  │
│  └──────┬─────────┘  └──────┬───────┘  └──────────▲─────────┘  │
└─────────│────────────────────│─────────────────────│────────────┘
          │ (File Upload)      │ (FormData)          │ (JSON)
┌─────────▼────────────────────▼─────────────────────┴────────────┐
│              Django Backend (Python / views.py)                  │
│  - Routes to C++ engine via pybind11                            │
│  - face_utils.py: dataset loading, ROC, evaluation              │
└───────────────────────────┬─────────────────────────────────────┘
                            │ (pybind11)
┌───────────────────────────▼─────────────────────────────────────┐
│              C++ Core Engine (cv_core.pyd)                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  cv_custom_algorithms.h (Legacy CV_03):                 │    │
│  │  • Gaussian Blur  • Sobel  • Harris  • λ⁻  • SIFT     │    │
│  │  • SSD/NCC matching  • Drawing primitives               │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  cv_face_algorithms.h (NEW CV_05):                      │    │
│  │  • Skin detection (YCbCr)  • Morphological ops          │    │
│  │  • Connected components    • Face detection              │    │
│  │  • Histogram equalization  • Jacobi eigendecomposition   │    │
│  │  • PCA / Eigenfaces training & recognition               │    │
│  └─────────────────────────────────────────────────────────┘    │
│  OpenCV used ONLY for: cv::Mat, cv::imread, cv::imwrite        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Installation

### Prerequisites
* Python 3.12+
* OpenCV 4.x (C++ binaries installed and added to `PATH`)
* A C++17 compiler (MSVC, GCC, or Clang)

### Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Build the C++ engine
python build_cv_core.py build_ext --inplace

# Download ORL dataset into datasets/orl_faces/
# Expected structure: datasets/orl_faces/s1/1.pgm ... s40/10.pgm

# Run the server
python manage.py runserver 8000
```

Open `http://localhost:8000` in your browser.

---

## 4. Usage Instructions

### Tab 1: Face Detection
1. Upload an image (color or grayscale).
2. Adjust min/max face size.
3. Click **"Detect Faces"** to locate faces with bounding boxes.

### Tab 2: Face Recognition
1. Select a dataset from the dropdown.
2. Configure train/test split and PCA components.
3. Click **"Train Model"** to build the eigenface model.
4. Upload a test face and click **"Recognize"** to identify it.

### Tab 3: Evaluation & ROC
1. Select dataset and parameters.
2. Click **"Run Full Evaluation"** for complete pipeline.
3. View accuracy, AUC, EER, ROC curve, confusion matrix, and eigenfaces.

### Tabs 4-6: Legacy (Harris, SIFT, Matching)
Same as CV_03 — see previous documentation.

---

## 5. Technical Specifications

### Eigenface Training (Covariance Trick)
Given M training faces of N pixels each:
1. Flatten and stack into matrix X (N×M)
2. Compute mean: μ = (1/M) Σxᵢ
3. Center: A = X − μ
4. Small covariance: L = AᵀA (M×M instead of N×N)
5. Eigendecompose L via Jacobi method → (λᵢ, vᵢ)
6. Recover true eigenvectors: uᵢ = Avᵢ / ‖Avᵢ‖
7. Keep top-k by eigenvalue → Eigenfaces U
8. Project: wᵢ = Uᵀ(xᵢ − μ)

### Face Recognition
1. Project test face: w = Uᵀ(x − μ)
2. Find nearest training projection: argmin ‖w − wᵢ‖₂
3. Accept if distance < threshold, reject otherwise

### Skin Detection (YCbCr)
```
Y  =  0.299R + 0.587G + 0.114B
Cb = −0.169R − 0.331G + 0.500B + 128
Cr =  0.500R − 0.419G − 0.081B + 128
Skin: 77 ≤ Cb ≤ 127, 133 ≤ Cr ≤ 173, Y > 30
```

---

## 6. File Descriptions

| File | Description |
|------|-------------|
| `cv_face_algorithms.h` | **NEW:** From-scratch face detection (skin, morphology, CC), PCA/Eigenfaces (Jacobi eigen, training, recognition) |
| `cv_custom_algorithms.h` | Legacy CV_03: Gaussian blur, Sobel, Harris, λ⁻, SIFT, SSD/NCC matching, drawing |
| `cv_core.cpp` | pybind11 module: 9 functions (5 legacy + 4 new face functions) |
| `face_utils.py` | **NEW:** Dataset loading, stratified splitting, ROC computation, plotting, full evaluation |
| `build_cv_core.py` | setuptools build script for compiling the C++ module |
| `detector/views.py` | Django API views: 11 endpoints (6 legacy + 5 new) |
| `detector/templates/detector/home.html` | Single-page UI with 6-tab layout |
| `static/css/style.css` | Dark theme with cyan/green accents |
| `static/js/app.js` | Frontend logic: tabs, upload, API calls, visualization |
| `cv_project/settings.py` | Django configuration |
| `cv_project/urls.py` | URL routing |
| `datasets/` | Face dataset storage (ORL/AT&T, etc.) |

---

**Institution:** Cairo University, Faculty of Engineering
**Department:** Systems & Biomedical Engineering
**Course:** Computer Vision (Assignment 5)
