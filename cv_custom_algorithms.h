#pragma once
/**
 * cv_custom_algorithms.h — From-scratch CV algorithms for CV_05
 *
 * Face Recognition (PCA / Eigenfaces):
 *   - Histogram Equalization
 *   - Jacobi Eigendecomposition
 *   - PCA Training & Recognition
 *   - Batch Recognition (for ROC evaluation)
 *
 * OpenCV is used ONLY for cv::Mat storage and basic types.
 * All algorithmic logic is implemented from scratch.
 */

#include <opencv2/core.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>

namespace custom {

constexpr double PI = 3.14159265358979323846;

/* ═══════════════════════════════════════════════════════════════════
 *  PIXEL ACCESS HELPERS
 * ═══════════════════════════════════════════════════════════════════ */

inline double get_pix(const cv::Mat& m, int y, int x) {
    if (y < 0) y = 0; if (y >= m.rows) y = m.rows - 1;
    if (x < 0) x = 0; if (x >= m.cols) x = m.cols - 1;
    if (m.type() == CV_8UC1)  return m.at<uchar>(y, x);
    if (m.type() == CV_32FC1) return m.at<float>(y, x);
    if (m.type() == CV_64FC1) return m.at<double>(y, x);
    return 0;
}


inline void set_pixel_color(cv::Mat& img, int x, int y, const cv::Scalar& c) {
    if (x < 0 || x >= img.cols || y < 0 || y >= img.rows) return;
    if (img.channels() == 3)
        img.at<cv::Vec3b>(y, x) = cv::Vec3b((uchar)c[0], (uchar)c[1], (uchar)c[2]);
    else
        img.at<uchar>(y, x) = (uchar)c[0];
}

/* ═══════════════════════════════════════════════════════════════════
 *  GAUSSIAN BLUR (separable, from scratch)
 * ═══════════════════════════════════════════════════════════════════ */

inline std::vector<double> make_gaussian_kernel(int ksize, double sigma) {
    int half = ksize / 2;
    std::vector<double> k(ksize);
    double sum = 0;
    for (int i = 0; i < ksize; ++i) {
        double x = i - half;
        k[i] = std::exp(-(x * x) / (2.0 * sigma * sigma));
        sum += k[i];
    }
    for (auto& v : k) v /= sum;
    return k;
}

inline cv::Mat gaussian_blur(const cv::Mat& src, int ksize, double sigma) {
    auto k = make_gaussian_kernel(ksize, sigma);
    int half = ksize / 2;

    // horizontal pass
    cv::Mat tmp(src.rows, src.cols, CV_64FC1, cv::Scalar(0));
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            double v = 0;
            for (int i = -half; i <= half; ++i)
                v += get_pix(src, y, std::clamp(x + i, 0, src.cols - 1)) * k[i + half];
            tmp.at<double>(y, x) = v;
        }
    // vertical pass
    cv::Mat dst(src.rows, src.cols, CV_64FC1, cv::Scalar(0));
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            double v = 0;
            for (int i = -half; i <= half; ++i)
                v += tmp.at<double>(std::clamp(y + i, 0, src.rows - 1), x) * k[i + half];
            dst.at<double>(y, x) = v;
        }
    return dst;
}


/* ═══════════════════════════════════════════════════════════════════
 *  CONVERT TO GRAYSCALE (from scratch)
 * ═══════════════════════════════════════════════════════════════════ */

inline cv::Mat to_grayscale(const cv::Mat& src) {
    if (src.channels() == 1) return src.clone();
    cv::Mat gray(src.rows, src.cols, CV_8UC1);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x) {
            cv::Vec3b bgr = src.at<cv::Vec3b>(y, x);
            // Standard luminosity formula
            gray.at<uchar>(y, x) = (uchar)(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]);
        }
    return gray;
}

/* ═══════════════════════════════════════════════════════════════════
 *  IMAGE RESIZE (bilinear interpolation, from scratch)
 * ═══════════════════════════════════════════════════════════════════ */

inline cv::Mat resize_image(const cv::Mat& src, int new_rows, int new_cols) {
    cv::Mat dst(new_rows, new_cols, src.type(), cv::Scalar(0));
    double ry = (double)src.rows / new_rows;
    double rx = (double)src.cols / new_cols;

    for (int y = 0; y < new_rows; ++y)
        for (int x = 0; x < new_cols; ++x) {
            double sy = y * ry, sx = x * rx;
            int y0 = (int)sy, x0 = (int)sx;
            int y1 = std::min(y0 + 1, src.rows - 1);
            int x1 = std::min(x0 + 1, src.cols - 1);
            double fy = sy - y0, fx = sx - x0;

            double v = get_pix(src, y0, x0) * (1-fx) * (1-fy)
                     + get_pix(src, y0, x1) * fx * (1-fy)
                     + get_pix(src, y1, x0) * (1-fx) * fy
                     + get_pix(src, y1, x1) * fx * fy;

            if (dst.type() == CV_8UC1)
                dst.at<uchar>(y, x) = (uchar)std::clamp(v, 0.0, 255.0);
            else if (dst.type() == CV_64FC1)
                dst.at<double>(y, x) = v;
            else if (dst.type() == CV_32FC1)
                dst.at<float>(y, x) = (float)v;
        }
    return dst;
}

/* ═══════════════════════════════════════════════════════════════════
 *  DRAWING PRIMITIVES
 * ═══════════════════════════════════════════════════════════════════ */

inline void draw_line(cv::Mat& img, cv::Point p1, cv::Point p2,
                      const cv::Scalar& col, int thick)
{
    int x0=p1.x,y0=p1.y,x1=p2.x,y1=p2.y;
    int dx=std::abs(x1-x0), dy=std::abs(y1-y0);
    int sx=(x0<x1)?1:-1, sy=(y0<y1)?1:-1, err=dx-dy;
    int r=std::max(1,thick/2);
    while (true) {
        for (int iy=-r+1;iy<r;++iy)
            for (int ix=-r+1;ix<r;++ix)
                if (ix*ix+iy*iy<r*r)
                    set_pixel_color(img,x0+ix,y0+iy,col);
        if (x0==x1&&y0==y1) break;
        int e2=2*err;
        if (e2>-dy){err-=dy;x0+=sx;}
        if (e2<dx){err+=dx;y0+=sy;}
    }
}

inline void draw_circle(cv::Mat& img, cv::Point cen, int rad,
                        const cv::Scalar& col, int thick)
{
    if (thick < 0) { // filled
        for (int y=-rad;y<=rad;++y)
            for (int x=-rad;x<=rad;++x)
                if (x*x+y*y<=rad*rad)
                    set_pixel_color(img,cen.x+x,cen.y+y,col);
        return;
    }
    int ri=std::max(0,rad-thick/2), ro=rad+(thick+1)/2;
    int ri2=ri*ri, ro2=ro*ro;
    for (int y=-ro;y<=ro;++y)
        for (int x=-ro;x<=ro;++x) {
            int d2=x*x+y*y;
            if (d2>=ri2&&d2<=ro2)
                set_pixel_color(img,cen.x+x,cen.y+y,col);
        }
}

inline void draw_cross(cv::Mat& img, cv::Point cen, int size,
                       const cv::Scalar& col, int thick)
{
    draw_line(img, {cen.x - size, cen.y}, {cen.x + size, cen.y}, col, thick);
    draw_line(img, {cen.x, cen.y - size}, {cen.x, cen.y + size}, col, thick);
}

// ═══════════════════════════════════════════════════════════
//  EIGENFACES / PCA (from scratch)
// ═══════════════════════════════════════════════════════════

struct EigenfaceModel {
    cv::Mat mean_face;
    cv::Mat eigenfaces;
    cv::Mat projections;
    std::vector<int> labels;
    int face_w, face_h, num_comp;
};

struct RecognitionResult {
    int label; double distance, confidence;
};

// ── Histogram Equalization ──
inline cv::Mat hist_equalize(const cv::Mat& gray) {
    cv::Mat r(gray.rows, gray.cols, CV_8UC1);
    int hist[256]={0};
    for(int y=0;y<gray.rows;++y) for(int x=0;x<gray.cols;++x) hist[gray.at<uchar>(y,x)]++;
    double cdf[256]; cdf[0]=hist[0];
    for(int i=1;i<256;++i) cdf[i]=cdf[i-1]+hist[i];
    double cmin=0; for(int i=0;i<256;++i) if(cdf[i]>0){cmin=cdf[i];break;}
    double tot=gray.rows*gray.cols;
    for(int y=0;y<gray.rows;++y) for(int x=0;x<gray.cols;++x)
        r.at<uchar>(y,x)=(uchar)std::clamp((cdf[gray.at<uchar>(y,x)]-cmin)/(tot-cmin)*255.0,0.0,255.0);
    return r;
}

// ── Jacobi Eigendecomposition (symmetric matrices) ──
inline void jacobi_eigen(const cv::Mat& A, cv::Mat& vals, cv::Mat& vecs, int max_it=300) {
    int n=A.rows;
    cv::Mat D=A.clone();
    vecs=cv::Mat(n,n,CV_64FC1,cv::Scalar(0));
    for(int i=0;i<n;++i) vecs.at<double>(i,i)=1.0;
    for(int it=0;it<max_it;++it) {
        int p=0,q=1; double mx=0;
        for(int i=0;i<n;++i) for(int j=i+1;j<n;++j)
            if(std::abs(D.at<double>(i,j))>mx){mx=std::abs(D.at<double>(i,j));p=i;q=j;}
        if(mx<1e-10) break;
        double app=D.at<double>(p,p),aqq=D.at<double>(q,q),apq=D.at<double>(p,q);
        double theta=(std::abs(app-aqq)<1e-15)?custom::PI/4.0:0.5*std::atan2(2.0*apq,app-aqq);
        double c=std::cos(theta),s=std::sin(theta);
        // Update D
        cv::Mat Dn=D.clone();
        Dn.at<double>(p,p)=c*c*app+2*s*c*apq+s*s*aqq;
        Dn.at<double>(q,q)=s*s*app-2*s*c*apq+c*c*aqq;
        Dn.at<double>(p,q)=0; Dn.at<double>(q,p)=0;
        for(int i=0;i<n;++i){
            if(i==p||i==q) continue;
            double dip=D.at<double>(i,p),diq=D.at<double>(i,q);
            Dn.at<double>(i,p)=c*dip+s*diq; Dn.at<double>(p,i)=Dn.at<double>(i,p);
            Dn.at<double>(i,q)=-s*dip+c*diq; Dn.at<double>(q,i)=Dn.at<double>(i,q);
        }
        D=Dn;
        for(int i=0;i<n;++i){
            double vip=vecs.at<double>(i,p),viq=vecs.at<double>(i,q);
            vecs.at<double>(i,p)=c*vip+s*viq;
            vecs.at<double>(i,q)=-s*vip+c*viq;
        }
    }
    vals=cv::Mat(n,1,CV_64FC1);
    for(int i=0;i<n;++i) vals.at<double>(i,0)=D.at<double>(i,i);
}

// ── Train Eigenfaces ──
inline EigenfaceModel train_eigenfaces(
    const std::vector<cv::Mat>& faces,
    const std::vector<int>& labels,
    int num_comp=0, int face_h=112, int face_w=92)
{
    int M=(int)faces.size(), N=face_h*face_w;
    // Flatten
    cv::Mat X(N,M,CV_64FC1);
    for(int i=0;i<M;++i){
        cv::Mat g=faces[i];
        if(g.channels()>1) g=custom::to_grayscale(g);
        if(g.rows!=face_h||g.cols!=face_w) g=custom::resize_image(g,face_h,face_w);
        g=hist_equalize(g);
        for(int y=0;y<face_h;++y) for(int x=0;x<face_w;++x)
            X.at<double>(y*face_w+x,i)=g.at<uchar>(y,x)/255.0;
    }
    // Mean
    cv::Mat mu(N,1,CV_64FC1,cv::Scalar(0));
    for(int i=0;i<M;++i) for(int j=0;j<N;++j) mu.at<double>(j,0)+=X.at<double>(j,i);
    for(int j=0;j<N;++j) mu.at<double>(j,0)/=M;
    // Center
    cv::Mat A(N,M,CV_64FC1);
    for(int i=0;i<M;++i) for(int j=0;j<N;++j) A.at<double>(j,i)=X.at<double>(j,i)-mu.at<double>(j,0);
    // Cov trick: L=A^T*A
    cv::Mat L(M,M,CV_64FC1,cv::Scalar(0));
    for(int i=0;i<M;++i) for(int j=i;j<M;++j){
        double s=0; for(int k=0;k<N;++k) s+=A.at<double>(k,i)*A.at<double>(k,j);
        L.at<double>(i,j)=s; L.at<double>(j,i)=s;
    }
    // Eigen
    cv::Mat evals,evecs;
    jacobi_eigen(L,evals,evecs,std::max(300,M*3));
    // Sort desc
    std::vector<int> idx(M); std::iota(idx.begin(),idx.end(),0);
    std::sort(idx.begin(),idx.end(),[&](int a,int b){return evals.at<double>(a,0)>evals.at<double>(b,0);});
    // Num components
    if(num_comp<=0||num_comp>=M){
        double tot=0; for(int i=0;i<M;++i) tot+=std::max(0.0,evals.at<double>(i,0));
        double cum=0; num_comp=std::min(M-1,50);
        for(int i=0;i<M;++i){cum+=std::max(0.0,evals.at<double>(idx[i],0));
            if(cum/tot>=0.95){num_comp=i+1;break;}}
        num_comp=std::max(1,std::min(num_comp,M-1));
    }
    // Recover eigenvectors
    cv::Mat ef(num_comp,N,CV_64FC1);
    for(int i=0;i<num_comp;++i){
        int id=idx[i];
        for(int j=0;j<N;++j){double s=0;for(int k=0;k<M;++k) s+=A.at<double>(j,k)*evecs.at<double>(k,id);ef.at<double>(i,j)=s;}
        double nm=0; for(int j=0;j<N;++j) nm+=ef.at<double>(i,j)*ef.at<double>(i,j);
        nm=std::sqrt(nm)+1e-10; for(int j=0;j<N;++j) ef.at<double>(i,j)/=nm;
    }
    // Project
    cv::Mat proj(num_comp,M,CV_64FC1);
    for(int i=0;i<M;++i) for(int j=0;j<num_comp;++j){
        double s=0; for(int k=0;k<N;++k) s+=ef.at<double>(j,k)*A.at<double>(k,i);
        proj.at<double>(j,i)=s;
    }
    return {mu,ef,proj,labels,face_w,face_h,num_comp};
}

// ── Recognize Face ──
inline RecognitionResult recognize_face(const EigenfaceModel& m, const cv::Mat& face, double thresh=5000.0) {
    cv::Mat g=face;
    if(g.channels()>1) g=custom::to_grayscale(g);
    if(g.rows!=m.face_h||g.cols!=m.face_w) g=custom::resize_image(g,m.face_h,m.face_w);
    g=hist_equalize(g);
    int N=m.face_h*m.face_w;
    // Flatten & center
    std::vector<double> v(N);
    for(int y=0;y<m.face_h;++y) for(int x=0;x<m.face_w;++x)
        v[y*m.face_w+x]=g.at<uchar>(y,x)/255.0-m.mean_face.at<double>(y*m.face_w+x,0);
    // Project
    std::vector<double> w(m.num_comp);
    for(int j=0;j<m.num_comp;++j){double s=0;for(int k=0;k<N;++k) s+=m.eigenfaces.at<double>(j,k)*v[k];w[j]=s;}
    // Find nearest
    int best=-1; double best_d=1e30;
    int M=m.projections.cols;
    for(int i=0;i<M;++i){
        double d=0; for(int j=0;j<m.num_comp;++j){double dd=w[j]-m.projections.at<double>(j,i);d+=dd*dd;}
        d=std::sqrt(d); if(d<best_d){best_d=d;best=i;}
    }
    int lbl=(best>=0&&best_d<thresh)?m.labels[best]:-1;
    return {lbl,best_d,1.0/(1.0+best_d)};
}

// ── Batch recognition (returns distances + labels for ROC) ──
inline void batch_recognize(const EigenfaceModel& m, const std::vector<cv::Mat>& faces,
    const std::vector<int>& true_labels,
    std::vector<int>& pred_labels, std::vector<double>& distances)
{
    pred_labels.clear(); distances.clear();
    for(size_t i=0;i<faces.size();++i){
        auto r=recognize_face(m,faces[i],1e30);
        pred_labels.push_back(r.label); distances.push_back(r.distance);
    }
}

// ── Visualize mean face / eigenfaces ──
inline cv::Mat eigenface_to_image(const cv::Mat& ef_row, int h, int w) {
    cv::Mat img(h,w,CV_8UC1);
    double mn=1e30,mx=-1e30;
    for(int i=0;i<h*w;++i){double v=ef_row.at<double>(0,i);if(v<mn)mn=v;if(v>mx)mx=v;}
    double range=mx-mn+1e-10;
    for(int y=0;y<h;++y) for(int x=0;x<w;++x)
        img.at<uchar>(y,x)=(uchar)((ef_row.at<double>(0,y*w+x)-mn)/range*255.0);
    return img;
}

inline cv::Mat mean_face_to_image(const cv::Mat& mean, int h, int w) {
    cv::Mat img(h,w,CV_8UC1);
    for(int y=0;y<h;++y) for(int x=0;x<w;++x)
        img.at<uchar>(y,x)=(uchar)std::clamp(mean.at<double>(y*w+x,0)*255.0,0.0,255.0);
    return img;
}

} // namespace custom
