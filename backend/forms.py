from django import forms


class UploadImageForm(forms.Form):
    class Meta:
        title = forms.CharField(max_length=50)
        file = forms.FileField()