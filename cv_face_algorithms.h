#pragma once
#include "cv_custom_algorithms.h"
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>

namespace face {

struct BoundingBox { int x, y, w, h; double conf; };


// ── Face Detection (OpenCV Cascade) ──
inline std::vector<BoundingBox> detect_faces(const cv::Mat& img, int min_sz=30, int max_sz=0) {
    std::vector<BoundingBox> results;
    static cv::CascadeClassifier cascade;
    static bool loaded = false;
    if (!loaded) {
        if(cascade.load("haarcascade_frontalface_default.xml")) {
            loaded = true;
        } else if(cascade.load("datasets/haarcascade_frontalface_default.xml")) {
            loaded = true;
        }
    }

    if (!loaded || img.empty()) return results;

    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = img.clone();
    }
    cv::equalizeHist(gray, gray);

    std::vector<cv::Rect> faces;
    int max_s = max_sz <= 0 ? std::max(img.rows, img.cols) : max_sz;
    cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(min_sz, min_sz), cv::Size(max_s, max_s));

    for (const auto& r : faces) {
        results.push_back({r.x, r.y, r.width, r.height, 1.0}); // Dummy confidence
    }
    return results;
}

// ── Draw Bounding Boxes ──
inline cv::Mat draw_boxes(const cv::Mat& img, const std::vector<BoundingBox>& boxes) {
    cv::Mat canvas;
    if(img.channels()==1){
        canvas=cv::Mat(img.rows,img.cols,CV_8UC3);
        for(int y=0;y<img.rows;++y) for(int x=0;x<img.cols;++x){
            uchar v=img.at<uchar>(y,x); canvas.at<cv::Vec3b>(y,x)=cv::Vec3b(v,v,v);}
    } else canvas=img.clone();
    for(auto& b:boxes) {
        cv::Scalar col(0,255,0);
        custom::draw_line(canvas,{b.x,b.y},{b.x+b.w,b.y},col,2);
        custom::draw_line(canvas,{b.x+b.w,b.y},{b.x+b.w,b.y+b.h},col,2);
        custom::draw_line(canvas,{b.x+b.w,b.y+b.h},{b.x,b.y+b.h},col,2);
        custom::draw_line(canvas,{b.x,b.y+b.h},{b.x,b.y},col,2);
    }
    return canvas;
}

} // namespace face
