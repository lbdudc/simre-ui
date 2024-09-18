# -*- coding: utf-8 -*-
import sys,os
import subprocess
import fasttext.util
import requests
import gdown
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR_ = os.path.join(BASE_DIR, '')
FILES_DIR = os.path.abspath(os.path.join(BASE_DIR, './fileserver'))
Path(FILES_DIR).mkdir(parents=True, exist_ok=True)
    
def main():
    print('comienza la descarga de modelos')
    subprocess.run([sys.executable, '-m', 'spacy', 'download', 'es_core_news_sm'])
    subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])    
    fasttext.util.download_model('en', if_exists='ignore') 
    fasttext.util.download_model('es', if_exists='ignore')
    os.rename(BASE_DIR_+'RE_SPL_Similarity/'+'cc.en.300.bin', FILES_DIR+'/'+'cc.en.300.bin')
    os.rename(BASE_DIR_+'RE_SPL_Similarity/'+'cc.es.300.bin', FILES_DIR+'/'+'cc.es.300.bin')
    os.remove(BASE_DIR_+'RE_SPL_Similarity/'+'cc.en.300.bin.gz')
    os.remove(BASE_DIR_+'RE_SPL_Similarity/'+'cc.es.300.bin.gz')
    
    #https://crscardellino.github.io/SBWCE/    
    url = "https://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.bin.gz"
    response = requests.get(url)
    if response.status_code == 200:
        with open(FILES_DIR+'/'+"SBW-vectors-300-min5.bin.gz", "wb") as file:
            file.write(response.content)
            print("File downloaded successfully!")
    else:
        print("Failed to download the file.")    
        
    #https://code.google.com/archive/p/word2vec/
    url = 'https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM'
    output = FILES_DIR+'/'+'GoogleNews-vectors-negative300.bin.gz'
    gdown.download(url, output, quiet=False)

if __name__ == '__main__':
    main()
    