"""CV_03 URL configuration."""

from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from detector import views

urlpatterns = [
    path("",              views.home,              name="home"),
    path("api/upload/",   views.api_upload,        name="api_upload"),
    path("api/harris/",   views.api_harris,        name="api_harris"),
    path("api/lambda/",   views.api_lambda_minus,  name="api_lambda"),
    path("api/sift/",     views.api_sift,          name="api_sift"),
    path("api/match-ssd/", views.api_match_ssd,    name="api_match_ssd"),
    path("api/match-ncc/", views.api_match_ncc,    name="api_match_ncc"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
