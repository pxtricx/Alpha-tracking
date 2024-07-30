from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_video, name='upload_video'),
    path('process_video/<str:video_path>/', views.process_video, name='process_video'),
]