# Generated by Django 4.1 on 2024-07-29 17:48

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("videoapp", "0002_videoresult"),
    ]

    operations = [
        migrations.AddField(
            model_name="videoresult",
            name="person_data",
            field=models.TextField(default=1),
            preserve_default=False,
        ),
    ]