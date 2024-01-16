from rest_framework import serializers
from django import forms
from .models import Test, UploadedImage

class TestSerializer(serializers.ModelSerializer):
    class Meta:
        model = Test
        fields = ['id', 'name', 'description']

class ImageUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedImage
        fields = ['file', 'applyRag', 'areCustomParametersApplied', 'clusterAmount', 'maximumFileSize', 'ragThreshold']
