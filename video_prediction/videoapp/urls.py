from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_video, name='upload_video'),
    path('process/<int:video_id>/', views.process_video, name='process_video'),
]