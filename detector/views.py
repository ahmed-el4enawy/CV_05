"""Views for the detector app — CV_05: Face Detection & Recognition."""

import os
import sys
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

# Add project root to path for face_utils
_project_root = str(settings.BASE_DIR)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import face_utils


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


# ═══════════════════════════════════════════════════════════════════
#  CV_05 ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@csrf_exempt
def api_detect_faces(request):
    """Detect faces in an uploaded image."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    img_path = request.POST.get("image_path", "")
    if not img_path or not os.path.isfile(img_path):
        return JsonResponse({"error": "Invalid image path"}, status=400)

    out_path = os.path.join(_results_dir(), f"{uuid.uuid4().hex}_faces.png")
    min_size = int(request.POST.get("min_size", 30))
    max_size = int(request.POST.get("max_size", 0))

    try:
        result = cv_core.detect_faces(img_path, out_path, min_size, max_size)
        return JsonResponse({
            "output": _media_url(result["output"]),
            "face_count": result["face_count"],
            "time_ms": round(result["time_ms"], 2),
            "faces": list(result["faces"]),
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_train_model(request):
    """Train eigenface model on a dataset directory."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    dataset_name = request.POST.get("dataset", "orl_faces")
    dataset_dir = os.path.join(settings.DATASET_ROOT, dataset_name)
    if not os.path.isdir(dataset_dir):
        return JsonResponse({"error": f"Dataset not found: {dataset_name}"}, status=400)

    num_components = int(request.POST.get("num_components", 0))
    train_ratio = float(request.POST.get("train_ratio", 0.7))

    try:
        # Load and split dataset
        data = face_utils.load_orl_dataset(dataset_dir)
        if not data:
            data = face_utils.load_generic_dataset(dataset_dir)
        if not data:
            return JsonResponse({"error": "No images found"}, status=400)

        train_paths, train_labels, test_paths, test_labels = \
            face_utils.stratified_split(data, train_ratio)

        results_path = os.path.join(settings.MEDIA_ROOT, "results")
        model_prefix = os.path.join(results_path, "eigenface_model")

        result = cv_core.train_eigenfaces(
            train_paths, train_labels, model_prefix, num_components
        )

        # Convert eigenface paths to URLs
        ef_urls = [_media_url(p) for p in result["eigenface_paths"]]
        mean_url = _media_url(result["mean_face_path"])

        return JsonResponse({
            "num_components": result["num_components"],
            "num_training": result["num_training"],
            "num_subjects": len(set(train_labels)),
            "num_test": len(test_paths),
            "mean_face": mean_url,
            "eigenfaces": ef_urls,
            "face_size": list(result["face_size"]),
            "time_ms": round(result["time_ms"], 2),
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_recognize(request):
    """Recognize a face using the trained model."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    img_path = request.POST.get("image_path", "")
    if not img_path or not os.path.isfile(img_path):
        return JsonResponse({"error": "Invalid image path"}, status=400)

    threshold = float(request.POST.get("threshold", 5000.0))

    try:
        result = cv_core.recognize_face(img_path, threshold)
        return JsonResponse({
            "predicted_label": result["predicted_label"],
            "distance": round(result["distance"], 4),
            "confidence": round(result["confidence"], 6),
            "time_ms": round(result["time_ms"], 2),
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_evaluate(request):
    """Run full evaluation pipeline with ROC curves."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    dataset_name = request.POST.get("dataset", "orl_faces")
    dataset_dir = os.path.join(settings.DATASET_ROOT, dataset_name)
    if not os.path.isdir(dataset_dir):
        return JsonResponse({"error": f"Dataset not found: {dataset_name}"}, status=400)

    num_components = int(request.POST.get("num_components", 0))
    train_ratio = float(request.POST.get("train_ratio", 0.7))

    try:
        results_path = os.path.join(settings.MEDIA_ROOT, "results")
        result = face_utils.run_full_evaluation(
            cv_core, dataset_dir, results_path, train_ratio, num_components
        )

        if "error" in result:
            return JsonResponse({"error": result["error"]}, status=400)

        # Convert paths to URLs
        response = {
            "accuracy": round(result["accuracy"] * 100, 2),
            "correct": result["correct"],
            "total": result["total"],
            "num_components": result["num_components"],
            "num_training": result["num_training"],
            "num_subjects": result["num_subjects"],
            "auc": round(result["auc"], 4),
            "eer": round(result["eer"], 4),
            "train_time_ms": round(result["train_time_ms"], 2),
            "eval_time_ms": round(result["eval_time_ms"], 2),
        }

        if result.get("mean_face_path"):
            response["mean_face"] = _media_url(result["mean_face_path"])
        if result.get("eigenface_paths"):
            response["eigenfaces"] = [_media_url(p) for p in result["eigenface_paths"]]
        if result.get("roc_path") and os.path.isfile(result["roc_path"]):
            response["roc_curve"] = _media_url(result["roc_path"])
        if result.get("cm_path") and os.path.isfile(result["cm_path"]):
            response["confusion_matrix"] = _media_url(result["cm_path"])

        return JsonResponse(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_list_datasets(request):
    """List available datasets in the datasets directory."""
    dataset_root = str(settings.DATASET_ROOT)
    datasets = []
    if os.path.isdir(dataset_root):
        for name in sorted(os.listdir(dataset_root)):
            path = os.path.join(dataset_root, name)
            if os.path.isdir(path):
                # Count subjects and images
                subjects = 0
                images = 0
                for sub in os.listdir(path):
                    subpath = os.path.join(path, sub)
                    if os.path.isdir(subpath):
                        subjects += 1
                        images += len([f for f in os.listdir(subpath)
                                      if os.path.splitext(f)[1].lower()
                                      in (".pgm", ".png", ".jpg", ".jpeg", ".bmp")])
                datasets.append({
                    "name": name,
                    "subjects": subjects,
                    "images": images,
                })
    return JsonResponse({"datasets": datasets})
