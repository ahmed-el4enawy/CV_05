# CV_03: Feature Detection, Descriptors & Matching

**Team number:** 12  
**Team members:**
* Ahmed Salah Geoshy Elshenawy  
* Abdullah Mohammad Khalifa  
* Mohamed Elsayed Attallah  

---

## Executive Summary

This project implements a complete feature detection, description, and matching pipeline for grayscale and color images. Developed entirely from scratch in **C++17**, the system bypasses standard OpenCV algorithmic functions in favor of first-principles implementations. It features custom Harris corner detection, λ⁻ (lambda-minus) corner detection, Scale Invariant Feature Transform (SIFT) descriptors, and feature matching using both Sum of Squared Differences (SSD) and Normalized Cross Correlation (NCC). All algorithms report computation times. The engine is integrated into a modern Django web interface via `pybind11`.

---

## Table of Contents

1. [Tasks Implemented](#1-tasks-implemented)
2. [System Architecture](#3-system-architecture)
3. [Installation](#4-installation)
4. [Usage Instructions](#5-usage-instructions)
5. [Technical Specifications](#6-technical-specifications)
6. [File Descriptions](#7-file-descriptions)

---

## 1. Tasks Implemented

### A) Feature Detection
* **Harris Corner Detection**: Extract unique features using the Harris operator. The response function R = det(M) − k·trace(M)² is computed from the structure tensor with Gaussian-windowed gradient products.
* **λ⁻ (Lambda-Minus) Detection**: Extract corners using the smallest eigenvalue of the structure tensor: λ⁻ = (trace − √(trace² − 4·det)) / 2. Both methods include non-maximum suppression and report computation time.

### B) SIFT Descriptors
* **Scale Invariant Feature Transform**: Generate feature descriptors from scratch:
  - Gaussian scale space construction (octaves × scales)
  - Difference of Gaussians (DoG) computation
  - Keypoint detection as DoG extrema (3×3×3 neighborhood)
  - Edge and low-contrast rejection via Hessian ratio test
  - Dominant orientation assignment (36-bin histogram)
  - 4×4×8 = 128-dimensional descriptor with trilinear interpolation
  - Descriptor normalization with 0.2 clamp
* Reports computation time.

### C) Feature Matching
* **Sum of Squared Differences (SSD)**: Match descriptors between image pairs using SSD distance with Lowe's ratio test for robust matching.
* **Normalized Cross Correlation (NCC)**: Match using zero-mean NCC with threshold and ratio test.
* Both methods report matching computation time.

---

## 2. System Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                  Web Interface (Browser)                        │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ Tab Navigation │  │ Control Panel│  │ Results Dashboard  │  │
│  │ (3 Tabs)       │  │ (Sliders)    │  │ (Images + Timing)  │  │
│  └──────┬─────────┘  └──────┬───────┘  └──────────▲─────────┘  │
└─────────│────────────────────│─────────────────────│────────────┘
          │ (File Upload)      │ (FormData)          │ (JSON)
┌─────────▼────────────────────▼─────────────────────┴────────────┐
│              Django Backend (Python / views.py)                  │
│  - Handles image upload, routes to C++ engine via pybind11      │
│  - Returns JSON with result image URLs and timing data          │
└───────────────────────────┬─────────────────────────────────────┘
                            │ (pybind11)
┌───────────────────────────▼─────────────────────────────────────┐
│              C++ Core Engine (cv_core.pyd)                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  From-scratch algorithms (cv_custom_algorithms.h):      │    │
│  │  • Gaussian Blur (separable)  • Sobel gradients         │    │
│  │  • Harris corners             • λ⁻ corners              │    │
│  │  • SIFT (full pipeline)       • SSD matching            │    │
│  │  • NCC matching               • Drawing primitives      │    │
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

# Run the server
python manage.py runserver 8000
```

Open `http://localhost:8000` in your browser.

---

## 4. Usage Instructions

### Tab 1: Feature Detection
1. Upload an image (grayscale or color).
2. Adjust Harris parameters (k, threshold, NMS radius) and λ⁻ parameters.
3. Click **"Detect Features"** to run both detectors simultaneously.
4. View detected corners with corner count and computation time badges.

### Tab 2: SIFT Descriptors
1. Upload an image.
2. Adjust SIFT parameters (octaves, scales, contrast/edge thresholds).
3. Click **"Generate Descriptors"** to extract SIFT keypoints.
4. View keypoints with orientation arrows, count, descriptor dimension, and timing.

### Tab 3: Feature Matching
1. Upload **two images**.
2. Adjust matching thresholds (SSD ratio / NCC threshold).
3. Click **"Match (SSD)"** or **"Match (NCC)"**.
4. View side-by-side match visualization with per-stage timing breakdown.

---

## 5. Technical Specifications

### Harris Corner Response
$$R = \det(M) - k \cdot \text{trace}(M)^2$$

where M is the structure tensor:
$$M = \begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix}$$

### Lambda-Minus
$$\lambda^- = \frac{\text{trace}(M) - \sqrt{\text{trace}(M)^2 - 4 \cdot \det(M)}}{2}$$

### SIFT Descriptor
128-dimensional vector from a 4×4 grid of 8-bin orientation histograms, normalized and clamped at 0.2.

---

## 6. File Descriptions

| File | Description |
|------|-------------|
| `cv_custom_algorithms.h` | From-scratch C++ implementations: Gaussian blur, Sobel, Harris, λ⁻, SIFT (scale space, DoG, descriptors), SSD/NCC matching, drawing primitives |
| `cv_core.cpp` | pybind11 module exposing 5 functions with std::chrono timing |
| `build_cv_core.py` | setuptools build script for compiling the C++ module |
| `detector/views.py` | Django API views for all algorithms |
| `detector/templates/detector/home.html` | Single-page UI with 3-tab layout |
| `static/css/style.css` | Dark theme with green/cyan accents |
| `static/js/app.js` | Frontend logic: tabs, upload, sliders, API calls, zoom |
| `cv_project/settings.py` | Django configuration |
| `cv_project/urls.py` | URL routing for 6 API endpoints |

---

**Institution:** Cairo University, Faculty of Engineering  
**Department:** Systems & Biomedical Engineering  
**Course:** Computer Vision (Assignment 3)
