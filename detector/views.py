"""Views for the detector app — CV_03."""

import os
import uuid

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# ── Make sure OpenCV DLLs are findable ──
_opencv_candidates = [
    r"C:\Program Files\opencv\build\bin",
    r"C:\Program Files\opencv\build\x64\vc16\bin",
    r"C:\Program Files\opencv\build\x64\vc17\bin",
    r"C:\opencv\build\bin",
    r"C:\opencv\build\x64\vc16\bin",
    r"C:\opencv\build\x64\vc17\bin",
]
for _opencv_bin in _opencv_candidates:
    if os.path.isdir(_opencv_bin):
        os.add_dll_directory(_opencv_bin)
        if _opencv_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _opencv_bin + ";" + os.environ.get("PATH", "")

import cv_core


# ── Helpers ────────────────────────────────────────────────────────

def _save_upload(f):
    """Save uploaded file, return absolute path."""
    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(f.name)[1].lower() or ".png"
    name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(upload_dir, name)
    with open(path, "wb") as dest:
        for chunk in f.chunks():
            dest.write(chunk)
    return path


def _media_url(abs_path):
    """Absolute path → media URL."""
    rel = os.path.relpath(abs_path, settings.MEDIA_ROOT)
    return settings.MEDIA_URL + rel.replace("\\", "/")


def _results_dir():
    d = os.path.join(settings.MEDIA_ROOT, "results")
    os.makedirs(d, exist_ok=True)
    return d


# ── Single page ───────────────────────────────────────────────────

def home(request):
    return render(request, "detector/home.html")


# ── API: upload image ─────────────────────────────────────────────

@csrf_exempt
def api_upload(request):
    if request.method != "POST" or "image" not in request.FILES:
        return JsonResponse({"error": "POST with image required"}, status=400)
    path = _save_upload(request.FILES["image"])
    return JsonResponse({"path": path, "url": _media_url(path)})


# ── API: Harris Corner Detection ──────────────────────────────────

@csrf_exempt
def api_harris(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    img_path = request.POST.get("image_path", "")
    if not img_path or not os.path.isfile(img_path):
        return JsonResponse({"error": "Invalid image path"}, status=400)

    out_path = os.path.join(_results_dir(), f"{uuid.uuid4().hex}_harris.png")

    result = cv_core.harris_detect(
        img_path, out_path,
        float(request.POST.get("k", 0.04)),
        float(request.POST.get("threshold", 1e6)),
        int(request.POST.get("nms_radius", 5)),
    )

    return JsonResponse({
        "output": _media_url(result["output"]),
        "corner_count": result["corner_count"],
        "time_ms": round(result["time_ms"], 2),
    })


# ── API: Lambda-Minus (λ⁻) Corner Detection ──────────────────────

@csrf_exempt
def api_lambda_minus(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    img_path = request.POST.get("image_path", "")
    if not img_path or not os.path.isfile(img_path):
        return JsonResponse({"error": "Invalid image path"}, status=400)

    out_path = os.path.join(_results_dir(), f"{uuid.uuid4().hex}_lambda.png")

    result = cv_core.lambda_minus_detect(
        img_path, out_path,
        float(request.POST.get("threshold", 1e4)),
        int(request.POST.get("nms_radius", 5)),
    )

    return JsonResponse({
        "output": _media_url(result["output"]),
        "corner_count": result["corner_count"],
        "time_ms": round(result["time_ms"], 2),
    })


# ── API: SIFT Descriptor Generation ──────────────────────────────

@csrf_exempt
def api_sift(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    img_path = request.POST.get("image_path", "")
    if not img_path or not os.path.isfile(img_path):
        return JsonResponse({"error": "Invalid image path"}, status=400)

    out_path = os.path.join(_results_dir(), f"{uuid.uuid4().hex}_sift.png")

    result = cv_core.sift_detect(
        img_path, out_path,
        int(request.POST.get("num_octaves", 4)),
        int(request.POST.get("scales_per_octave", 3)),
        float(request.POST.get("contrast_threshold", 0.04)),
        float(request.POST.get("edge_threshold", 10.0)),
    )

    return JsonResponse({
        "output": _media_url(result["output"]),
        "keypoint_count": result["keypoint_count"],
        "descriptor_dim": result["descriptor_dim"],
        "time_ms": round(result["time_ms"], 2),
    })


# ── API: SSD Feature Matching ────────────────────────────────────

@csrf_exempt
def api_match_ssd(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    path1 = request.POST.get("image_path_1", "")
    path2 = request.POST.get("image_path_2", "")
    if not path1 or not os.path.isfile(path1):
        return JsonResponse({"error": "Invalid image 1 path"}, status=400)
    if not path2 or not os.path.isfile(path2):
        return JsonResponse({"error": "Invalid image 2 path"}, status=400)

    out_path = os.path.join(_results_dir(), f"{uuid.uuid4().hex}_ssd.png")

    result = cv_core.match_features_ssd(
        path1, path2, out_path,
        float(request.POST.get("ratio_threshold", 0.75)),
        int(request.POST.get("num_octaves", 4)),
        float(request.POST.get("contrast_threshold", 0.04)),
    )

    return JsonResponse({
        "output": _media_url(result["output"]),
        "kp_count_1": result["kp_count_1"],
        "kp_count_2": result["kp_count_2"],
        "match_count": result["match_count"],
        "sift1_time_ms": round(result["sift1_time_ms"], 2),
        "sift2_time_ms": round(result["sift2_time_ms"], 2),
        "match_time_ms": round(result["match_time_ms"], 2),
        "total_time_ms": round(result["total_time_ms"], 2),
    })


# ── API: NCC Feature Matching ────────────────────────────────────

@csrf_exempt
def api_match_ncc(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    path1 = request.POST.get("image_path_1", "")
    path2 = request.POST.get("image_path_2", "")
    if not path1 or not os.path.isfile(path1):
        return JsonResponse({"error": "Invalid image 1 path"}, status=400)
    if not path2 or not os.path.isfile(path2):
        return JsonResponse({"error": "Invalid image 2 path"}, status=400)

    out_path = os.path.join(_results_dir(), f"{uuid.uuid4().hex}_ncc.png")

    result = cv_core.match_features_ncc(
        path1, path2, out_path,
        float(request.POST.get("ncc_threshold", 0.7)),
        int(request.POST.get("num_octaves", 4)),
        float(request.POST.get("contrast_threshold", 0.04)),
    )

    return JsonResponse({
        "output": _media_url(result["output"]),
        "kp_count_1": result["kp_count_1"],
        "kp_count_2": result["kp_count_2"],
        "match_count": result["match_count"],
        "sift1_time_ms": round(result["sift1_time_ms"], 2),
        "sift2_time_ms": round(result["sift2_time_ms"], 2),
        "match_time_ms": round(result["match_time_ms"], 2),
        "total_time_ms": round(result["total_time_ms"], 2),
    })
