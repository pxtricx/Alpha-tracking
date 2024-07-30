# views.py
import os
import sys
current_file_path = os.path.abspath(os.path.dirname(__file__))

# Navigate up to the project root (Alpha-tracking)
project_root = os.path.abspath(os.path.join(current_file_path, '..', '..', '..', '..', '..'))
deep_sort_path = os.path.join(project_root, 'ultralytics', 'yolo', 'v8', 'detect', 'deep_sort_pytorch')
# Add the project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if deep_sort_path not in sys.path:
    sys.path.insert(0, deep_sort_path)

print("Project root:", project_root)
print("sys.path:", sys.path)

from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from ultralytics.yolo.v8.detect.predict import run_prediction, init_tracker
#from deep_sort_pytorch.utils.parser import get_config
from ultralytics.yolo.v8.detect.deep_sort_pytorch.utils.parser import get_config
from ultralytics.yolo.utils import DEFAULT_CONFIG


def upload_video(request):
    if request.method == 'POST' and request.FILES['video']:
        video = request.FILES['video']
        fs = FileSystemStorage()
        filename = fs.save(video.name, video)
        video_path = fs.path(filename)
        return redirect('process_video', video_path=video_path)
    return render(request, 'upload.html')

def process_video(request, video_path):
    cfg = get_config()
    cfg.source = video_path
    # You don't need to set cfg.config_file here, as it's handled in init_tracker()
    print(f"Processing video: {video_path}")
    results = run_prediction(cfg)

    context = {
        'excel_path': results['excel_path'],
        'plot_path': results['plot_path'],
        'metrics_df': results['metrics_df'].to_html(),
        'cm_df': results['cm_df'].to_html(),
        'person_count_df': results['person_count_df'].to_html(),
        'time_df': results['time_df'].to_html(),
        'df_images': results['df_images'].to_html()
    }

    return render(request, 'results.html', context)
