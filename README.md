# SIMRE

*SimRE is a Python-based tool for automatically identifying requirements' similarity oriented to SPL projects.

## Installation

* Its necessary to have installed al least Python 3.8.10 and pip 23.1.2
* Install the required libraries. All necessary libraries are listed in the requirements.txt file. To install them, execute the following command `pip install -r ./requirements.txt`
* It is necessary to download several pre-trained models. This can be done automatically or manually. To download the models automatically, execute the following command: `python init_models.py`. It necesary to perform this task only one time.
* To download the models manually, follow these steps:

    * a. Download the models of spacy for Spanish and English: `python -m spacy download es_core_news_sm` and `python -m spacy download en_core_web_sm`
    * b. Download the models of fastText for Spanish and English: `cc.es.300.bin` and `cc.es.300.bin` from https://fasttext.cc/docs/en/crawl-vectors.html
    * c. Download the word2vec-based models for Spanish and English: `SBW-vectors-300-min5.bin.gz` from https://crscardellino.github.io/SBWCE/  and `GoogleNews-vectors-negative300.bin.gz` from https://code.google.com/archive/p/word2vec/  
    * d. At the same directory level as the main folder, create a new folder named fileserver. Place all the pre-trained models into this folder.

## Usage

* To execute the tool, you have to do the following command: `python manage.py runserver 0.0.0.0:8000`. Then you can go to the url: `http://127.0.0.1:8000/analisis_req/`
* The parameters of the tool are:
  * CSV file that contains the list of new requirements. 
  * XML or UVL file that contains the existing requirements. 
  * JSON file that contains the requirements description. 
  * language: 'en' for English and 'es' for Spanish
  * listModels: optional. List of the models. optional. The default model is 1: MiniLM-L12-v2
  * threshold: optional. The default value is 0.7.
  * preprocess: optional. The values are: True to allow the pre-processing (default value), and False for without pre-processing

* Models: 1:Model multilingual MiniLM-L12-v2
          2:Model multilingual distiluse-cased-v2
          3:Model multilingual mpnet-base-v2
          4:'Model word2vec
          5:'Model fastText  
            
## Formats files

*XML file:

```
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<featureModel chosenLayoutAlgorithm="4">
  <struct>
    <and mandatory="true" name="GEMA_SPL">      
      <and name="UserManagement">
        <or name="UM_Registration">
          <feature mandatory="true" name="UM_R_ByAdmin"/>
          <feature mandatory="true" name="UM_R_Anonymous"/>
        </or>              
      </and>      
    </and>
  </struct>
  <featureOrder userDefined="false"/>
</featureModel>
```

*UVL file: 

```
features
	UserManagement 
		optional
			UM_Registration 
				or
					UM_R_ByAdmin 
					UM_R_Anonymous
```

*JSON file:

```
{
  "UserManagement": { 
      "label": "User Management",
      "desc": "User Management" 
  },
  "UM_Registration": { 
      "label": "User Registration",
      "desc": "User Registration" 
  },
  
}
```
