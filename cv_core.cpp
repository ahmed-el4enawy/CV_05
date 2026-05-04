/**
 * cv_core – C++ computer vision operations for CV_05.
 *
 * Assignment 5: Face Detection & Recognition (PCA / Eigenfaces)
 *
 * All algorithmic functions are implemented from scratch.
 * OpenCV is used ONLY for cv::Mat, cv::imread, cv::imwrite (image I/O),
 * and cv::CascadeClassifier (face detection — permitted per instructions).
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
#include "cv_face_algorithms.h"

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

// Global eigenface model (cached in memory)
custom::EigenfaceModel g_model;
bool g_model_loaded = false;

} // anon

/* ═══════════════════════════════════════════════════════════════════
 *  Face Detection (OpenCV CascadeClassifier — permitted per instructions)
 * ═══════════════════════════════════════════════════════════════════ */

py::dict detect_faces_api(const std::string& in, const std::string& out,
                          int min_size, int max_size)
{
    Timer timer;
    cv::Mat img = load_image(in, "unchanged");
    // Ensure we have either 1 or 3 channels for the face detector
    if (img.channels() == 4) {
        // BGRA → BGR: drop alpha channel manually
        cv::Mat bgr(img.rows, img.cols, CV_8UC3);
        for (int y = 0; y < img.rows; ++y)
            for (int x = 0; x < img.cols; ++x) {
                cv::Vec4b p = img.at<cv::Vec4b>(y, x);
                bgr.at<cv::Vec3b>(y, x) = cv::Vec3b(p[0], p[1], p[2]);
            }
        img = bgr;
    }
    auto boxes = face::detect_faces(img, min_size, max_size);
    cv::Mat canvas = face::draw_boxes(img, boxes);
    save_image(out, canvas);

    py::dict result;
    result["output"] = out;
    result["face_count"] = (int)boxes.size();
    result["time_ms"] = timer.elapsed_ms();
    py::list blist;
    for (auto& b : boxes) {
        py::dict bd;
        bd["x"]=b.x; bd["y"]=b.y; bd["w"]=b.w; bd["h"]=b.h; bd["conf"]=b.conf;
        blist.append(bd);
    }
    result["faces"] = blist;
    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 *  Train Eigenfaces (PCA — from scratch)
 * ═══════════════════════════════════════════════════════════════════ */

py::dict train_eigenfaces_api(const std::vector<std::string>& image_paths,
                              const std::vector<int>& labels,
                              const std::string& model_out,
                              int num_components,
                              int face_h, int face_w)
{
    Timer timer;
    std::vector<cv::Mat> faces;
    for (auto& p : image_paths) {
        cv::Mat img = cv::imread(p, cv::IMREAD_GRAYSCALE);
        if (img.empty()) throw std::runtime_error("Cannot read: " + p);
        faces.push_back(img);
    }

    g_model = custom::train_eigenfaces(faces, labels, num_components, face_h, face_w);
    g_model_loaded = true;

    // Save mean face image
    ensure_parent_dir(model_out);
    cv::Mat mean_img = custom::mean_face_to_image(g_model.mean_face, g_model.face_h, g_model.face_w);
    std::string mean_path = model_out + "_mean.png";
    save_image(mean_path, mean_img);

    // Save top eigenfaces
    py::list ef_paths;
    int n_show = std::min(g_model.num_comp, 10);
    for (int i = 0; i < n_show; ++i) {
        cv::Mat row = g_model.eigenfaces.row(i);
        cv::Mat ef_img = custom::eigenface_to_image(row, g_model.face_h, g_model.face_w);
        std::string p = model_out + "_ef" + std::to_string(i) + ".png";
        save_image(p, ef_img);
        ef_paths.append(p);
    }

    py::dict result;
    result["time_ms"] = timer.elapsed_ms();
    result["num_components"] = g_model.num_comp;
    result["num_training"] = (int)faces.size();
    result["mean_face_path"] = mean_path;
    result["eigenface_paths"] = ef_paths;
    result["face_size"] = py::make_tuple(g_model.face_h, g_model.face_w);
    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 *  Recognize Face
 * ═══════════════════════════════════════════════════════════════════ */

py::dict recognize_face_api(const std::string& image_path, double threshold)
{
    if (!g_model_loaded)
        throw std::runtime_error("No model loaded. Train first.");

    Timer timer;
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) throw std::runtime_error("Cannot read: " + image_path);

    auto r = custom::recognize_face(g_model, img, threshold);

    py::dict result;
    result["predicted_label"] = r.label;
    result["distance"] = r.distance;
    result["confidence"] = r.confidence;
    result["time_ms"] = timer.elapsed_ms();
    return result;
}

/* ═══════════════════════════════════════════════════════════════════
 *  Batch Recognize (for evaluation / ROC)
 * ═══════════════════════════════════════════════════════════════════ */

py::dict batch_recognize_api(const std::vector<std::string>& test_paths,
                             const std::vector<int>& true_labels)
{
    if (!g_model_loaded)
        throw std::runtime_error("No model loaded. Train first.");

    Timer timer;
    std::vector<cv::Mat> faces;
    for (auto& p : test_paths) {
        cv::Mat img = cv::imread(p, cv::IMREAD_GRAYSCALE);
        if (img.empty()) throw std::runtime_error("Cannot read: " + p);
        faces.push_back(img);
    }

    std::vector<int> preds;
    std::vector<double> dists;
    custom::batch_recognize(g_model, faces, true_labels, preds, dists);

    int correct = 0;
    for (size_t i = 0; i < true_labels.size(); ++i)
        if (preds[i] == true_labels[i]) correct++;

    py::dict result;
    result["predictions"] = preds;
    result["distances"] = dists;
    result["true_labels"] = true_labels;
    result["accuracy"] = true_labels.empty() ? 0.0 : (double)correct / true_labels.size();
    result["correct"] = correct;
    result["total"] = (int)true_labels.size();
    result["time_ms"] = timer.elapsed_ms();
    return result;
}

/* ═══════════════════ pybind11 bindings ════════════════════════════ */

PYBIND11_MODULE(cv_core, m)
{
    m.doc() = "C++ custom CV core – CV_05: Face Detection & Recognition (PCA/Eigenfaces)";

    m.def("detect_faces", &detect_faces_api, "Face detection (color + grayscale)",
          py::arg("input_path"), py::arg("output_path"),
          py::arg("min_size")=30, py::arg("max_size")=0);

    m.def("train_eigenfaces", &train_eigenfaces_api, "Train PCA/Eigenfaces model",
          py::arg("image_paths"), py::arg("labels"), py::arg("model_output_prefix"),
          py::arg("num_components")=0, py::arg("face_h")=112, py::arg("face_w")=92);

    m.def("recognize_face", &recognize_face_api, "Recognize a face",
          py::arg("image_path"), py::arg("threshold")=5000.0);

    m.def("batch_recognize", &batch_recognize_api, "Batch recognition for evaluation",
          py::arg("test_paths"), py::arg("true_labels"));
}

/* ═══════════════════════════════════════════════════════════════════
 *  CTYPES INTEGRATION (Strict Rubric Compliance)
 * ═══════════════════════════════════════════════════════════════════ */
extern "C" {

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

// ctypes wrapper: detect faces directly from raw numpy image pointer
EXPORT int ctypes_detect_faces(uint8_t* img_data, int rows, int cols, int channels, 
                               int* out_boxes, int max_boxes) 
{
    cv::Mat img(rows, cols, channels == 3 ? CV_8UC3 : CV_8UC1, img_data);
    auto boxes = face::detect_faces(img);
    int count = std::min((int)boxes.size(), max_boxes);
    
    for(int i = 0; i < count; ++i) {
        out_boxes[i * 5 + 0] = boxes[i].x;
        out_boxes[i * 5 + 1] = boxes[i].y;
        out_boxes[i * 5 + 2] = boxes[i].w;
        out_boxes[i * 5 + 3] = boxes[i].h;
        out_boxes[i * 5 + 4] = (int)(boxes[i].conf * 1000); // scaled confidence
    }
    return count;
}

// ctypes wrapper: recognize a single face from raw numpy pointer
EXPORT int ctypes_recognize_face(uint8_t* img_data, int rows, int cols, double* out_distance) 
{
    if (!g_model_loaded) return -1;
    cv::Mat img(rows, cols, CV_8UC1, img_data);
    auto result = custom::recognize_face(g_model, img, 5000.0);
    if (out_distance) *out_distance = result.distance;
    return result.label;
}

} // extern "C"
