# rag-webservice


## Getting started
**Setupn**
* Run `poetry update`
* Import nltk if necessary: 
```
import nltk
nltk.download('averaged_perceptron_tagger_eng')
```
* Edit the .env file
* Start the webservice `poetry run python3.11 src/ws-basic.py`
* Test RAG `poetry run python3.11 src/ws/RAG_Manager.py`


**Testing Endpoints**
See the list of all endpoints: http://localhost:5000/apidocs/ 
* Get list of installed LLM models `curl -X POST http://localhost:5000/llm/models/list` 
