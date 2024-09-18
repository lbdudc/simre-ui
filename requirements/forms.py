# -*- coding: utf-8 -*-

from django import forms
from django.core.exceptions import ValidationError
import os

def validate_file_extension(value):
    ext = os.path.splitext(value.name)[1]  
    valid_extensions = ['.xml', '.uvl']  
    if ext.lower() not in valid_extensions:
        raise ValidationError('Unsupported file extension.')


class CustomCheckboxSelectMultiple(forms.CheckboxSelectMultiple):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.option_help_texts = kwargs.pop('option_help_texts', {})

    def create_option(self, *args, **kwargs):
        option = super().create_option(*args, **kwargs)
        option['help_text'] = self.option_help_texts.get(option['value'], '')
        return option
        
OPCIONES = [
        ('1', 'Multilingual MiniLM-L12-v2'),
        ('2', 'Multilingual distiluse-cased-v2'),
        ('3', 'Multilingual mpnet-base-v2'),
        ('4', 'Word2vec'),
        ('5', 'FastText'),
    ]

OPTION_HELP_TEXTS = {
        '1': 'This is help text for Option 1.',
        '2': 'This is help text for Option 2.',
        '3': 'This is help text for Option 3.',
        '4': 'This is help text for Option 3.',
        '5': 'This is help text for Option 3.'
    }
  
langs = [
         ('es', 'Spanish'),
         ('en', 'English'),
     ]
   
class DesgargaReq(forms.Form):
    submit_button = forms.CharField(widget=forms.HiddenInput(), initial="Enviar")
    #submit_button = forms.SubmitField(label='Ver requisitos')
       
class AnalisisForm(forms.Form):
    req = forms.CharField(label="Enter the new requirements",max_length=500,widget=forms.Textarea(attrs={'rows': 5, 'cols': 50, 'style': 'resize: vertical;'}),required=False,help_text = "Requirements can be entered manually, separated by semicolons (;), or you can upload a csv file containing a single column with the list of requirements.")
    docfile = forms.FileField(label='',required=False)
    docfileComp = forms.FileField(label='Enter the existing feature models and their requirements',help_text = 'Select the file with the features (.xml or .uvl format) and the file with the description of the requirements associated with the features.',required=True,validators=[validate_file_extension])
    docfileComp2 = forms.FileField(required=True)
    threshold = forms.CharField(label='Enter the similarity score threshold',max_length=5,required=True,help_text = "The threshold is the percentage of similarity that the model will use to filter the results. In case more than one model is chosen, this value will be the average of the models' results.",initial=0.7)
    modelos = forms.MultipleChoiceField(required=True,
        label='Select the model(s)',
        widget=forms.CheckboxSelectMultiple(), initial=['1'],
        choices=OPCIONES,
        help_text = "More than one model can be chosen, but the processing time will increase as each model is included.")
    langs = forms.ChoiceField(choices=langs, label='Select a language')
    preprocess = forms.BooleanField(label='Performing a pre-process step', required=False, initial=True)
    
    
class AnalisisForm1(forms.Form):
    req = forms.CharField(label="Ingrese los requisitos",max_length=500,widget=forms.Textarea(attrs={'rows': 5, 'cols': 50, 'style': 'resize: vertical;'}),required=False,help_text = "Los requisitos pueden ingresarse de manera manual, separados por punto y coma (;), o se puede subir un archivo de tipo csv que contenga una única columna con el listado de requisitos")
    docfile = forms.FileField(label='',required=False)
    docfileComp = forms.FileField(label='Selecciona el archivo para comparar (.xml y .json)',required=True)
    docfileComp2 = forms.FileField(required=True)
    threshold = forms.CharField(label='Ingrese el umbral de la similitud',max_length=5,required=True,help_text = "El umbral es el porcentaje de similaridad que usará el modelo para filtrar los resultados. En caso se escoja más de un modelo, éste valor será el promedio de los resultados de los modelos.",initial=0.7)
    modelos = forms.MultipleChoiceField(required=True,
        widget=forms.CheckboxSelectMultiple(), initial=['1'],
        choices=OPCIONES,
        help_text = "Se pueden escoger más de un modelo, pero el tiempo de espera se incrementará conforme se incluyan cada modelo.")
    langs = forms.ChoiceField(choices=langs, label='Selecciona un idioma')

    