# Generated by Django 4.1 on 2024-07-28 22:56

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("videoapp", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="VideoResult",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("image_analysis", models.TextField()),
                ("detection_metrics", models.TextField()),
                ("confusion_matrix", models.TextField()),
                ("person_count", models.TextField()),
                ("processing_time", models.TextField()),
                ("processed_at", models.DateTimeField(auto_now_add=True)),
                (
                    "video",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="result",
                        to="videoapp.video",
                    ),
                ),
            ],
        ),
    ]
