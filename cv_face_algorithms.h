#pragma once
#include "cv_custom_algorithms.h"
#include <map>
#include <numeric>
#include <algorithm>

namespace face {

struct BoundingBox { int x, y, w, h; double conf; };

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

// ── Skin Detection (YCbCr) ──
inline cv::Mat detect_skin(const cv::Mat& bgr) {
    cv::Mat skin(bgr.rows, bgr.cols, CV_8UC1, cv::Scalar(0));
    for(int y=0;y<bgr.rows;++y) for(int x=0;x<bgr.cols;++x) {
        cv::Vec3b p=bgr.at<cv::Vec3b>(y,x);
        double R=p[2],G=p[1],B=p[0];
        double Cb=-0.169*R-0.331*G+0.500*B+128.0;
        double Cr= 0.500*R-0.419*G-0.081*B+128.0;
        double Y = 0.299*R+0.587*G+0.114*B;
        if(Cb>=77&&Cb<=127&&Cr>=133&&Cr<=173&&Y>30) skin.at<uchar>(y,x)=255;
    }
    return skin;
}

// ── Morphological Ops ──
inline cv::Mat erode_bin(const cv::Mat& b, int k=3) {
    int h=k/2; cv::Mat r(b.rows,b.cols,CV_8UC1,cv::Scalar(0));
    for(int y=h;y<b.rows-h;++y) for(int x=h;x<b.cols-h;++x) {
        bool ok=true;
        for(int ky=-h;ky<=h&&ok;++ky) for(int kx=-h;kx<=h&&ok;++kx)
            if(b.at<uchar>(y+ky,x+kx)==0) ok=false;
        if(ok) r.at<uchar>(y,x)=255;
    } return r;
}
inline cv::Mat dilate_bin(const cv::Mat& b, int k=3) {
    int h=k/2; cv::Mat r(b.rows,b.cols,CV_8UC1,cv::Scalar(0));
    for(int y=h;y<b.rows-h;++y) for(int x=h;x<b.cols-h;++x) {
        bool ok=false;
        for(int ky=-h;ky<=h&&!ok;++ky) for(int kx=-h;kx<=h&&!ok;++kx)
            if(b.at<uchar>(y+ky,x+kx)!=0) ok=true;
        if(ok) r.at<uchar>(y,x)=255;
    } return r;
}
inline cv::Mat morph_open(const cv::Mat& b, int k=5){return dilate_bin(erode_bin(b,k),k);}
inline cv::Mat morph_close(const cv::Mat& b, int k=5){return erode_bin(dilate_bin(b,k),k);}

// ── Union-Find ──
class UF {
    std::vector<int> p,r;
public:
    UF(int n):p(n),r(n,0){for(int i=0;i<n;++i)p[i]=i;}
    int find(int x){while(p[x]!=x){p[x]=p[p[x]];x=p[x];}return x;}
    void unite(int a,int b){a=find(a);b=find(b);if(a==b)return;if(r[a]<r[b])std::swap(a,b);p[b]=a;if(r[a]==r[b])r[a]++;}
};

// ── Connected Components → Bounding Boxes ──
inline std::vector<BoundingBox> cc_boxes(const cv::Mat& bin, int min_area=400) {
    int R=bin.rows, C=bin.cols;
    cv::Mat lbl(R,C,CV_32SC1,cv::Scalar(0));
    UF uf(R*C+1); int nl=1;
    for(int y=0;y<R;++y) for(int x=0;x<C;++x) {
        if(bin.at<uchar>(y,x)==0) continue;
        int left=(x>0&&bin.at<uchar>(y,x-1))?lbl.at<int>(y,x-1):0;
        int above=(y>0&&bin.at<uchar>(y-1,x))?lbl.at<int>(y-1,x):0;
        if(!left&&!above) lbl.at<int>(y,x)=nl++;
        else if(left&&!above) lbl.at<int>(y,x)=left;
        else if(!left&&above) lbl.at<int>(y,x)=above;
        else {lbl.at<int>(y,x)=std::min(left,above);uf.unite(left,above);}
    }
    for(int y=0;y<R;++y) for(int x=0;x<C;++x)
        if(lbl.at<int>(y,x)>0) lbl.at<int>(y,x)=uf.find(lbl.at<int>(y,x));
    std::map<int,int> x1,y1,x2,y2,area;
    for(int y=0;y<R;++y) for(int x=0;x<C;++x) {
        int l=lbl.at<int>(y,x); if(!l) continue;
        if(!area.count(l)){x1[l]=x;y1[l]=y;x2[l]=x;y2[l]=y;area[l]=0;}
        if(x<x1[l])x1[l]=x; if(x>x2[l])x2[l]=x;
        if(y<y1[l])y1[l]=y; if(y>y2[l])y2[l]=y; area[l]++;
    }
    std::vector<BoundingBox> boxes;
    int img_area=R*C;
    for(auto&[l,a]:area) {
        if(a<min_area||a>img_area*0.4) continue;
        int w=x2[l]-x1[l]+1, h=y2[l]-y1[l]+1;
        double ar=(double)h/w;
        if(ar<0.8||ar>3.0) continue;
        boxes.push_back({x1[l],y1[l],w,h,(double)a/(w*h)});
    }
    return boxes;
}

// ── IoU + NMS ──
inline double iou(const BoundingBox& a, const BoundingBox& b) {
    int x1=std::max(a.x,b.x), y1=std::max(a.y,b.y);
    int x2=std::min(a.x+a.w,b.x+b.w), y2=std::min(a.y+a.h,b.y+b.h);
    if(x2<=x1||y2<=y1) return 0;
    double inter=(x2-x1)*(y2-y1);
    return inter/(a.w*a.h+b.w*b.h-inter);
}
inline std::vector<BoundingBox> nms(std::vector<BoundingBox>& boxes, double thresh=0.3) {
    std::sort(boxes.begin(),boxes.end(),[](auto&a,auto&b){return a.conf>b.conf;});
    std::vector<BoundingBox> keep;
    std::vector<bool> supp(boxes.size(),false);
    for(int i=0;i<(int)boxes.size();++i) {
        if(supp[i]) continue; keep.push_back(boxes[i]);
        for(int j=i+1;j<(int)boxes.size();++j)
            if(iou(boxes[i],boxes[j])>thresh) supp[j]=true;
    }
    return keep;
}

// ── Face Detection (Color) ──
inline std::vector<BoundingBox> detect_faces_color(const cv::Mat& bgr, int min_sz=30, int max_sz=0) {
    if(max_sz<=0) max_sz=std::min(bgr.rows,bgr.cols);
    cv::Mat skin=detect_skin(bgr);
    skin=morph_open(skin,5);
    skin=morph_close(skin,7);
    skin=dilate_bin(skin,5);
    auto boxes=cc_boxes(skin, min_sz*min_sz/4);
    // Filter by size
    std::vector<BoundingBox> filtered;
    for(auto& b:boxes) {
        if(b.w<min_sz||b.h<min_sz||b.w>max_sz||b.h>max_sz) continue;
        filtered.push_back(b);
    }
    return nms(filtered, 0.3);
}

// ── Face Detection (Grayscale) — edge density + symmetry sliding window ──
inline std::vector<BoundingBox> detect_faces_gray(const cv::Mat& gray, int min_sz=30, int max_sz=0) {
    if(max_sz<=0) max_sz=std::min(gray.rows,gray.cols)/2;
    cv::Mat blur=custom::gaussian_blur(gray,5,1.0);
    cv::Mat Ix=custom::sobel(blur,1,0), Iy=custom::sobel(blur,0,1);
    cv::Mat edge(gray.rows,gray.cols,CV_64FC1);
    for(int y=0;y<gray.rows;++y) for(int x=0;x<gray.cols;++x)
        edge.at<double>(y,x)=std::sqrt(Ix.at<double>(y,x)*Ix.at<double>(y,x)+Iy.at<double>(y,x)*Iy.at<double>(y,x));
    // Integral image of edges
    cv::Mat integ(gray.rows+1,gray.cols+1,CV_64FC1,cv::Scalar(0));
    for(int y=0;y<gray.rows;++y) for(int x=0;x<gray.cols;++x)
        integ.at<double>(y+1,x+1)=edge.at<double>(y,x)+integ.at<double>(y,x+1)+integ.at<double>(y+1,x)-integ.at<double>(y,x);

    auto rect_sum=[&](int y1,int x1,int y2,int x2)->double{
        y1=std::max(0,y1);x1=std::max(0,x1);y2=std::min(gray.rows,y2);x2=std::min(gray.cols,x2);
        return integ.at<double>(y2,x2)-integ.at<double>(y1,x2)-integ.at<double>(y2,x1)+integ.at<double>(y1,x1);
    };

    std::vector<BoundingBox> cands;
    for(int sz=min_sz;sz<=max_sz;sz=(int)(sz*1.3)) {
        int ww=sz, hh=(int)(sz*1.3);
        int step=std::max(1,sz/4);
        for(int y=0;y+hh<gray.rows;y+=step) for(int x=0;x+ww<gray.cols;x+=step) {
            double area=ww*hh;
            double ed=rect_sum(y,x,y+hh,x+ww)/area;
            if(ed<5||ed>80) continue;
            // Symmetry: compare left and right halves
            int mid=x+ww/2;
            double left_e=rect_sum(y,x,y+hh,mid);
            double right_e=rect_sum(y,mid,y+hh,x+ww);
            double sym=1.0-std::abs(left_e-right_e)/(left_e+right_e+1e-10);
            if(sym<0.6) continue;
            // Vertical structure: middle band (eye region) has more edges
            double top_e=rect_sum(y,x,y+hh/3,x+ww)/(ww*hh/3.0);
            double mid_e=rect_sum(y+hh/3,x,y+2*hh/3,x+ww)/(ww*hh/3.0);
            double bot_e=rect_sum(y+2*hh/3,x,y+hh,x+ww)/(ww*hh/3.0);
            // Face heuristic: middle band (eyes/nose) should have highest edge density
            double vert_score=(mid_e>top_e&&mid_e>bot_e)?1.0:0.5;
            double score=sym*0.35+std::min(1.0,ed/30.0)*0.35+vert_score*0.3;
            if(score>0.5) cands.push_back({x,y,ww,hh,score});
        }
    }
    return nms(cands, 0.3);
}

// ── Unified Face Detection ──
inline std::vector<BoundingBox> detect_faces(const cv::Mat& img, int min_sz=30, int max_sz=0) {
    if(img.channels()==3) return detect_faces_color(img, min_sz, max_sz);
    return detect_faces_gray(img, min_sz, max_sz);
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

// ═══════════════════════════════════════════════════════════
//  EIGENFACES / PCA
// ═══════════════════════════════════════════════════════════

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

} // namespace face
