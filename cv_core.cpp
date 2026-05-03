/**
 * cv_core – C++ computer vision operations for CV_03.
 *
 * Assignment 3: Feature Detection, SIFT Descriptors & Feature Matching
 * All algorithmic functions are implemented from scratch in cv_custom_algorithms.h.
 * OpenCV is used ONLY for cv::Mat, cv::imread, and cv::imwrite (image I/O).
 *
 * Exposed to Python via pybind11.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <chrono>

#include "cv_custom_algorithms.h"

namespace py = pybind11;

/* ───────────────────── helpers ─────────────────────────────────── */
namespace {

cv::Mat load_image(const std::string& path, const std::string& mode)
{
    int flag = cv::IMREAD_COLOR;
    if (mode == "gray")       flag = cv::IMREAD_GRAYSCALE;
    else if (mode == "unchanged") flag = cv::IMREAD_UNCHANGED;

    cv::Mat img = cv::imread(path, flag);
    if (img.empty())
        throw std::runtime_error("Could not read image: " + path);
    return img;
}

void ensure_parent_dir(const std::string& path)
{
    std::filesystem::path p(path);
    if (p.has_parent_path())
        std::filesystem::create_directories(p.parent_path());
}

std::string save_image(const std::string& path, const cv::Mat& img)
{
    ensure_parent_dir(path);
    if (!cv::imwrite(path, img))
        throw std::runtime_error("Failed to write image: " + path);
    return path;
}

// Timer utility
struct Timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start;
    Timer() : start(clock::now()) {}

    double elapsed_ms() const {
        auto end = clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

} // anon

/* ═══════════════════════════════════════════════════════════════════
 *  1. Harris Corner Detection
 * ═══════════════════════════════════════════════════════════════════ */

py::dict harris_detect(const std::string& in,
                       const std::string& out,
                       double k_param,
                       double threshold,
                       int nms_radius)
{
    Timer timer;

    cv::Mat color = load_image(in, "color");
    cv::Mat gray = custom::to_grayscale(color);

    auto corners = custom::harris_corners(gray, k_param, threshold, nms_radius);

    // Draw corners on image
    cv::Mat canvas = color.clone();
    for (auto& kp : corners) {
        custom::draw_circle(canvas, {(int)kp.x, (int)kp.y}, 4,
                           cv::Scalar(0, 0, 255), 1);
        custom::draw_cross(canvas, {(int)kp.x, (int)kp.y}, 3,
                          cv::Scalar(0, 255, 0), 1);
    }

    save_image(out, canvas);
    double elapsed = timer.elapsed_ms();

    py::dict result;
    result["output"] = out;
    result["corner_count"] = (int)corners.size();
    result["time_ms"] = elapsed;

    // Return corner coordinates
    py::list corner_list;
    for (auto& kp : corners) {
        py::dict cp;
        cp["x"] = (int)kp.x;
        cp["y"] = (int)kp.y;
        cp["response"] = kp.response;
        corner_list.append(cp);
    }
    result["corners"] = corner_list;

    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 *  2. Lambda-Minus (λ⁻) Corner Detection
 * ═══════════════════════════════════════════════════════════════════ */

py::dict lambda_minus_detect(const std::string& in,
                             const std::string& out,
                             double threshold,
                             int nms_radius)
{
    Timer timer;

    cv::Mat color = load_image(in, "color");
    cv::Mat gray = custom::to_grayscale(color);

    auto corners = custom::lambda_minus_corners(gray, threshold, nms_radius);

    // Draw corners
    cv::Mat canvas = color.clone();
    for (auto& kp : corners) {
        custom::draw_circle(canvas, {(int)kp.x, (int)kp.y}, 4,
                           cv::Scalar(255, 0, 0), 1);
        custom::draw_cross(canvas, {(int)kp.x, (int)kp.y}, 3,
                          cv::Scalar(0, 255, 255), 1);
    }

    save_image(out, canvas);
    double elapsed = timer.elapsed_ms();

    py::dict result;
    result["output"] = out;
    result["corner_count"] = (int)corners.size();
    result["time_ms"] = elapsed;

    py::list corner_list;
    for (auto& kp : corners) {
        py::dict cp;
        cp["x"] = (int)kp.x;
        cp["y"] = (int)kp.y;
        cp["response"] = kp.response;
        corner_list.append(cp);
    }
    result["corners"] = corner_list;

    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 *  3. SIFT Feature Descriptors
 * ═══════════════════════════════════════════════════════════════════ */

py::dict sift_detect(const std::string& in,
                     const std::string& out,
                     int num_octaves,
                     int scales_per_octave,
                     double contrast_threshold,
                     double edge_threshold)
{
    Timer timer;

    cv::Mat color = load_image(in, "color");
    cv::Mat gray = custom::to_grayscale(color);

    custom::SIFTParams params;
    params.num_octaves = num_octaves;
    params.scales_per_octave = scales_per_octave;
    params.contrast_threshold = contrast_threshold;
    params.edge_threshold = edge_threshold;

    std::vector<custom::KeyPoint> keypoints;
    std::vector<std::vector<double>> descriptors;

    custom::sift_detect_and_describe(gray, keypoints, descriptors, params);

    // Draw keypoints with orientation
    cv::Mat canvas = custom::draw_keypoints(color, keypoints,
                                             cv::Scalar(0, 255, 0), true);

    save_image(out, canvas);
    double elapsed = timer.elapsed_ms();

    py::dict result;
    result["output"] = out;
    result["keypoint_count"] = (int)keypoints.size();
    result["descriptor_dim"] = keypoints.empty() ? 128 : (int)descriptors[0].size();
    result["time_ms"] = elapsed;

    // Return keypoint info
    py::list kp_list;
    for (size_t i = 0; i < keypoints.size(); ++i) {
        py::dict kd;
        kd["x"] = (int)keypoints[i].x;
        kd["y"] = (int)keypoints[i].y;
        kd["scale"] = keypoints[i].scale;
        kd["orientation"] = keypoints[i].orientation * 180.0 / custom::PI;
        kp_list.append(kd);
    }
    result["keypoints"] = kp_list;

    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 *  4. Feature Matching — SSD
 * ═══════════════════════════════════════════════════════════════════ */

py::dict match_features_ssd(const std::string& in1,
                            const std::string& in2,
                            const std::string& out,
                            double ratio_threshold,
                            int num_octaves,
                            double contrast_threshold)
{
    Timer timer_total;

    cv::Mat color1 = load_image(in1, "color");
    cv::Mat gray1 = custom::to_grayscale(color1);
    cv::Mat color2 = load_image(in2, "color");
    cv::Mat gray2 = custom::to_grayscale(color2);

    // SIFT on image 1
    Timer timer_sift1;
    custom::SIFTParams params;
    params.num_octaves = num_octaves;
    params.contrast_threshold = contrast_threshold;
    std::vector<custom::KeyPoint> kps1;
    std::vector<std::vector<double>> desc1;
    custom::sift_detect_and_describe(gray1, kps1, desc1, params);
    double sift1_ms = timer_sift1.elapsed_ms();

    // SIFT on image 2
    Timer timer_sift2;
    std::vector<custom::KeyPoint> kps2;
    std::vector<std::vector<double>> desc2;
    custom::sift_detect_and_describe(gray2, kps2, desc2, params);
    double sift2_ms = timer_sift2.elapsed_ms();

    // SSD matching
    Timer timer_match;
    auto matches = custom::match_ssd(desc1, desc2, ratio_threshold);
    double match_ms = timer_match.elapsed_ms();

    // Draw matches
    cv::Mat canvas = custom::draw_matches(color1, kps1, color2, kps2, matches);
    save_image(out, canvas);

    double total_ms = timer_total.elapsed_ms();

    py::dict result;
    result["output"] = out;
    result["kp_count_1"] = (int)kps1.size();
    result["kp_count_2"] = (int)kps2.size();
    result["match_count"] = (int)matches.size();
    result["sift1_time_ms"] = sift1_ms;
    result["sift2_time_ms"] = sift2_ms;
    result["match_time_ms"] = match_ms;
    result["total_time_ms"] = total_ms;

    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 *  5. Feature Matching — NCC
 * ═══════════════════════════════════════════════════════════════════ */

py::dict match_features_ncc(const std::string& in1,
                            const std::string& in2,
                            const std::string& out,
                            double ncc_threshold,
                            int num_octaves,
                            double contrast_threshold)
{
    Timer timer_total;

    cv::Mat color1 = load_image(in1, "color");
    cv::Mat gray1 = custom::to_grayscale(color1);
    cv::Mat color2 = load_image(in2, "color");
    cv::Mat gray2 = custom::to_grayscale(color2);

    // SIFT on image 1
    Timer timer_sift1;
    custom::SIFTParams params;
    params.num_octaves = num_octaves;
    params.contrast_threshold = contrast_threshold;
    std::vector<custom::KeyPoint> kps1;
    std::vector<std::vector<double>> desc1;
    custom::sift_detect_and_describe(gray1, kps1, desc1, params);
    double sift1_ms = timer_sift1.elapsed_ms();

    // SIFT on image 2
    Timer timer_sift2;
    std::vector<custom::KeyPoint> kps2;
    std::vector<std::vector<double>> desc2;
    custom::sift_detect_and_describe(gray2, kps2, desc2, params);
    double sift2_ms = timer_sift2.elapsed_ms();

    // NCC matching
    Timer timer_match;
    auto matches = custom::match_ncc(desc1, desc2, ncc_threshold);
    double match_ms = timer_match.elapsed_ms();

    // Draw matches
    cv::Mat canvas = custom::draw_matches(color1, kps1, color2, kps2, matches);
    save_image(out, canvas);

    double total_ms = timer_total.elapsed_ms();

    py::dict result;
    result["output"] = out;
    result["kp_count_1"] = (int)kps1.size();
    result["kp_count_2"] = (int)kps2.size();
    result["match_count"] = (int)matches.size();
    result["sift1_time_ms"] = sift1_ms;
    result["sift2_time_ms"] = sift2_ms;
    result["match_time_ms"] = match_ms;
    result["total_time_ms"] = total_ms;

    return result;
}

/* ═══════════════════ pybind11 bindings ════════════════════════════ */

PYBIND11_MODULE(cv_core, m)
{
    m.doc() = "C++ custom CV core – Harris, SIFT & Feature Matching for CV_03";

    m.def("harris_detect", &harris_detect,
          "Harris corner detection (from scratch)",
          py::arg("input_path"), py::arg("output_path"),
          py::arg("k") = 0.04, py::arg("threshold") = 1e6,
          py::arg("nms_radius") = 5);

    m.def("lambda_minus_detect", &lambda_minus_detect,
          "Lambda-minus corner detection (from scratch)",
          py::arg("input_path"), py::arg("output_path"),
          py::arg("threshold") = 1e4,
          py::arg("nms_radius") = 5);

    m.def("sift_detect", &sift_detect,
          "SIFT keypoint detection and descriptor generation (from scratch)",
          py::arg("input_path"), py::arg("output_path"),
          py::arg("num_octaves") = 4,
          py::arg("scales_per_octave") = 3,
          py::arg("contrast_threshold") = 0.04,
          py::arg("edge_threshold") = 10.0);

    m.def("match_features_ssd", &match_features_ssd,
          "Match features between two images using SSD",
          py::arg("input_path_1"), py::arg("input_path_2"),
          py::arg("output_path"),
          py::arg("ratio_threshold") = 0.75,
          py::arg("num_octaves") = 4,
          py::arg("contrast_threshold") = 0.04);

    m.def("match_features_ncc", &match_features_ncc,
          "Match features between two images using NCC",
          py::arg("input_path_1"), py::arg("input_path_2"),
          py::arg("output_path"),
          py::arg("ncc_threshold") = 0.7,
          py::arg("num_octaves") = 4,
          py::arg("contrast_threshold") = 0.04);
}
