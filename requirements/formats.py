# -*- coding: utf-8 -*-

import nltk
nltk.download('stopwords')
nltk.download('punkt')
import json
import os
from bs4 import BeautifulSoup
import pandas as pd
from uvl.UVLCustomLexer import UVLCustomLexer
from uvl.UVLPythonParser import UVLPythonParser
from antlr4 import CommonTokenStream, FileStream
from django.core.exceptions import ValidationError
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
   
def get_req_feature(fileFeature, fileJson):
    if fileFeature.name.endswith('xml') :
         df_list_features = get_req_feature_xml(fileFeature, fileJson)  
    elif fileFeature.name.endswith('uvl'):
        df_list_features = get_req_feature_uvl(fileFeature, fileJson)    
    else:
        print('error')
        raise ValidationError('Unsupported file extension.')
        
    if not fileJson.name.endswith('json'):
        print('error')
        raise ValidationError('Unsupported file extension.')
    
    return df_list_features

def get_req_feature_xml(fileFeature, fileJson):
    """Obtiene los requisitos y los features de dos archivos
    Parámetros:
        fileFeature -- archivo en formato xml donde están los features
        fileJson -- archivo en formato json donde se encuentran las descripciones de los requisitos
    Salida:
        dataframe con los datos
    """
    soup = BeautifulSoup(fileFeature.read(), features="lxml")
    data2 = json.load(fileJson)    
    rows = []
    
    for feature in soup.find_all(attrs={'name': True}):
        dict={}
        nombre = feature.get('name')
        descripcion = ''
        nombre_sp = ''
        if nombre in data2.keys():
            nombre_sp = data2[nombre]['label']
            if 'desc' in data2[nombre]:
                descripcion = data2[nombre]['desc']
        dict['name'] = nombre
        dict['hidden'] = feature.get('hidden')
        dict['disabled'] = feature.get('disabled')
        dict['mandatory'] = feature.get('mandatory')
        dict['nombre'] = nombre_sp
        dict['descripcion'] = descripcion
        rows.append(dict)
        
    df_list_features = pd.DataFrame(rows)
    return df_list_features

def get_context_type(group):
    tipo = ""    
    if isinstance(group, UVLPythonParser.OrGroupContext):
        tipo = "or"
    elif isinstance(group, UVLPythonParser.AlternativeGroupContext):
        tipo = "alt"
    elif isinstance(group, UVLPythonParser.OptionalGroupContext):
        tipo = "optional"
    elif isinstance(group, UVLPythonParser.MandatoryGroupContext):
        tipo = "mandatory"
    elif isinstance(group, UVLPythonParser.CardinalityGroupContext):
        tipo = "or"
    else:
        print("Tipo de variable desconocido")
    return tipo

def getUVLFeatureChildren(group, rows, fileJson):
    """Obtiene los features hijos del archivo uvl, función recursiva
    Parámetros:
        group -- grupo de feature
        rows -- arreglo donde se están guardando la información de los features
        fileJson -- archivo en formato json donde se encuentran las descripciones de los requisitos
    """
    for f1 in group:
      typeChild = get_context_type(f1);
      for j in range(f1.getChildCount()):
        group_spec = f1.getChild(j)
        if isinstance(group_spec, UVLPythonParser.GroupSpecContext):
            for k in range(group_spec.getChildCount()):
                feature_context = group_spec.getChild(k)
                if isinstance(feature_context, UVLPythonParser.FeatureContext):
                    reference_context = feature_context.getChild(0)
                    if reference_context.getChildCount() == 1:
                        _group = feature_context.group()
                        nombre = reference_context.getText()
                        description = ''
                        if nombre in fileJson.keys():
                            if 'desc' in fileJson[nombre]:
                                description = fileJson[nombre]['desc']
                        new_feature = {
                            'disabled': "false",
                            'mandatory': "true" if typeChild == "mandatory" else "false",
                            'name': nombre,
                            'descripcion': description,
                        }
                        rows.append(new_feature)
                        getUVLFeatureChildren(_group, rows,fileJson)                    


def get_req_feature_uvl(fileFeature, fileJson):
    """Obtiene los requisitos y los features de dos archivos
    Parámetros:
        fileFeature -- archivo en formato uvl donde están los features
        fileJson -- archivo en formato json donde se encuentran las descripciones de los requisitos
    Salida:
        dataframe con los datos
    """
    rows = []        
    path = default_storage.save('tmp/somename.uvl', ContentFile(fileFeature.read()))
    tmp_file = os.path.join(settings.MEDIA_ROOT, path)
    input_stream = FileStream(tmp_file);
    lexer = UVLCustomLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = UVLPythonParser(stream)
    tree = parser.featureModel();
    listFeatures = tree.features();
    feature = listFeatures.feature();
    grupo = feature.group()
    data2 = json.load(fileJson)
    getUVLFeatureChildren(grupo, rows,data2)        
    df_list_features = pd.DataFrame(rows)
    default_storage.delete(path)
    
    return df_list_features
  