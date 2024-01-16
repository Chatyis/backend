from django.db import models

class Test(models.Model):
    name = models.CharField(max_length=30)
    description = models.CharField(max_length=300)

class UploadedImage(models.Model):
    applyRag = models.BooleanField()
    areCustomParametersApplied = models.BooleanField()
    clusterAmount = models.IntegerField(max_length=1000)
    file = models.ImageField(upload_to='uploaded_images/')
    maximumFileSize = models.IntegerField(max_length=1000)
    ragThreshold = models.IntegerField(max_length=32)
    
