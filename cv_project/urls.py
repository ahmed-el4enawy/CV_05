"""CV_05 URL configuration."""

from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from detector import views

urlpatterns = [
    path("",                  views.home,              name="home"),
    path("api/upload/",       views.api_upload,        name="api_upload"),
    # CV_05 endpoints
    path("api/detect-faces/", views.api_detect_faces,  name="api_detect_faces"),
    path("api/train-model/",  views.api_train_model,   name="api_train_model"),
    path("api/recognize/",    views.api_recognize,     name="api_recognize"),
    path("api/evaluate/",     views.api_evaluate,      name="api_evaluate"),
    path("api/dataset-info/",views.api_dataset_info, name="api_dataset_info"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
