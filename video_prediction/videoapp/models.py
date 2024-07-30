from django.db import models

# Create your models here.
class Video(models.Model):
    title = models.CharField(max_length=100)
    video_file = models.FileField(upload_to='videos/')

    def __str__(self):
        return self.title

class VideoResult(models.Model):
    video = models.OneToOneField(Video, on_delete=models.CASCADE, related_name='result')
    image_analysis = models.TextField()
    detection_metrics = models.TextField()
    confusion_matrix = models.TextField()
    person_count = models.TextField()
    processing_time = models.TextField()
    person_data = models.TextField()
    processed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Results for {self.video.title}"