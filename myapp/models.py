from django.db import models
from django.contrib.auth.models import User
from django.http import HttpRequest
from django.contrib.auth import get_user_model

User=get_user_model()
# Create your models here.
# models.py
class PDFFile(models.Model):
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE, default=True)
    file = models.FileField(upload_to='pdf_files/')
    prediction = models.CharField(max_length=50, blank=True,default=True)
    num_pages = models.IntegerField(default=True)
    key=models.CharField(max_length=200,default=True)


    def __str__(self):
        return self.file.name
