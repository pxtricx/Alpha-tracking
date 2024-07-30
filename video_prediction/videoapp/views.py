from django.shortcuts import render, redirect
from .forms import VideoForm
from .models import Video, VideoResult
import os
import subprocess
import pandas as pd

def upload_video(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            model = request.POST.get('model')
            form.save()
            return redirect('process_video', video_id=form.instance.id, model=model)
    else:
        form = VideoForm()
    return render(request, 'upload.html', {'form': form})

def process_video(request, video_id, model):
    video = Video.objects.get(id=video_id)
    video_path = video.video_file.path

    # Ensure the path is correctly formatted
    video_path = video_path.replace("\\", "/")

    # Correct path to the script directory
    script_directory = os.path.abspath("D:/VScode/a project-thesis/Alpha-tracking/ultralytics/yolo/v8/detect")
    os.chdir(script_directory)

    # Run the predict.py script on the uploaded video
    command = f"python predict.py model={model} source=\"{video_path}\" show=True"
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        output = result.stdout
        error = result.stderr
    except subprocess.CalledProcessError as e:
        output = e.stdout
        error = e.stderr

    results_dir = 'D:/VScode/a project-thesis/Alpha-tracking/video_prediction/media/results'
    excel_path = os.path.join(results_dir, 'detection_and_analysis_results_yolov8x.xlsx')

    if not os.path.exists(excel_path):
        # Call the predict function here
        subprocess.run(['python', 'path/to/your/predict_script.py', '--source', video_path])

    # Read the Excel file
    data_frames = pd.read_excel(excel_path, sheet_name=None)

        # Save results to the database
    video_result, created = VideoResult.objects.update_or_create(
        video=video,
        defaults={
            'image_analysis': data_frames['Image Analysis'].to_html(),
            'detection_metrics': data_frames['Detection Metrics'].to_html(),
            'confusion_matrix': data_frames['Confusion Matrix'].to_html(),
            'person_count': data_frames['Person Count per Frame'].to_html(),
            'processing_time': data_frames['Processing Time'].to_html(),
            'person_data' : data_frames['Person Count mean'].to_html()
        }
    )
    
    # Pass the data frames to the template
    context = {
        'image_analysis': video_result.image_analysis,
        'detection_metrics': video_result.detection_metrics,
        'confusion_matrix': video_result.confusion_matrix,
        'person_count': video_result.person_count,
        'processing_time': video_result.processing_time,
        'person_data' : video_result.person_data
    }
    
    return render(request, 'results.html', context)

