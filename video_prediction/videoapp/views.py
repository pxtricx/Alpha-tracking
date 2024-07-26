from django.shortcuts import render, redirect
from .forms import VideoForm
from .models import Video
import os
import subprocess

def upload_video(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('process_video', video_id=form.instance.id)
    else:
        form = VideoForm()
    return render(request, 'upload.html', {'form': form})

def process_video(request, video_id):
    video = Video.objects.get(id=video_id)
    video_path = video.video_file.path

    # Ensure the path is correctly formatted
    video_path = video_path.replace("\\", "/")

    # Correct path to the script directory
    script_directory = os.path.abspath("D:/VScode/a project-thesis/Alpha-tracking/ultralytics/yolo/v8/detect")
    os.chdir(script_directory)

    # Run the predict.py script on the uploaded video
    command = f"python predict.py model=yolov8l.pt source=\"{video_path}\" show=True"
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        output = result.stdout
        error = result.stderr
    except subprocess.CalledProcessError as e:
        output = e.stdout
        error = e.stderr

    return render(request, 'process.html', {'video': video, 'output': output, 'error': error})
