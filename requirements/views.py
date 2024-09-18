from django.shortcuts import render
from django.contrib import messages
from django.http import HttpResponse
from .forms import AnalisisForm
from .process import get_listadoAnalizar,get_similar_varios
from .formats import get_req_feature
import mimetypes
import os

def index(request):
    return render(request, "index_new.html")

def analisis_req(request):
    if request.method == 'POST':
        if 'btn' in request.POST:
            form = AnalisisForm(request.POST, request.FILES)            
            if form.is_valid():
                requisitos = form.cleaned_data['req']
                langs = form.cleaned_data['langs']
                threshold = form.cleaned_data['threshold']
                lista_requisitos = form.cleaned_data['docfile']
                requisitos_comp = form.cleaned_data['docfileComp']
                requisitos_comp2 = form.cleaned_data['docfileComp2']
                newdoc = request.FILES["docfile"] if lista_requisitos else ''              
                opciones_seleccionadas = form.cleaned_data['modelos']
                pre_process = form.cleaned_data['preprocess']
                df_repo = get_req_feature(requisitos_comp, requisitos_comp2)                
                df = get_listadoAnalizar(requisitos, newdoc)  
                df_final = get_similar_varios(df, df_repo, opciones_seleccionadas,langs,float(threshold),pre_process) 
                
                if df_final is not None and not df_final.empty:         
                    lista1 = df_final.to_html(na_rep='',index=False)
                    return render(request, "analisis_req.html", {'form': form,'listado': lista1})
                else:
                    messages.warning(request, 'No similar requirements found.')
                    return render(request, "analisis_req.html", {'form': form})
            else:
                return render(request, "analisis_req.html", {'form': form})
        elif 'btn_descarga' in request.POST:
            response = descargar_requisitos()
            return response
    else:
        form = AnalisisForm(initial={'req_input': '',})
        return render(request, "analisis_req.html", {'form': form})
      
        
#################################################################################################

def descargar_requisitos():  
    filename = 'Lista_resultados_analisis.csv'
    response = descargar_general(filename) 
    return response

def descargar_general(filename):  
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
    filepath = BASE_DIR + '/' + filename 
    path = open(filepath, 'r', encoding="utf8") 
 
    mime_type, _ = mimetypes.guess_type(filepath)    
    response = HttpResponse(path, content_type = mime_type) 
    response['Content-Disposition'] = f"attachment; filename={filename}" 
    return response








