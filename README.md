# rag-webservice
RAG-Webservice is a middleware for an Ollama LLM server and a Moodle plugin called Openchat.

## Features
* Create an index for a provided document.
* Stores a document index in a vector store including metadata about the requestion system, course ID, content type, and instance ID
* Answers querys concering all documents within a given context defined by the provided metadata (system, course, content type, instance ID)

## Getting started

**Docker setup**
* Build Docker container: `docker build --no-cache -t rag-webservice .`
* `docker compose up`
* note: `docker run --name rag-webservice -p 5000:5000 rag-webservice`

**Local setup without using docker**
* Run `poetry install`
* Import nltk if necessary: 
```
import nltk
nltk.download('averaged_perceptron_tagger_eng')
```
* Create and edit an .env file, use .env.example as a template
* Start the webservice `poetry run python3.11 src/ws-basic.py`
* Test RAG `poetry run python3.11 src/RAG_Manager.py`


**Testing Endpoints**
See the list of all endpoints in the browser: http://localhost:5000/apidocs/ 
* Get list of installed LLM models `curl -X POST http://localhost:5000/llm/models/list` 
* Test RAG inside docker: `docker exec -it rag-webservice python3.11 src/RAG_Manager.py`
* Test Ollama inside docker: `docker exec -it rag-webservice curl http://host.docker.internal:11434`
* `curl -X POST http://localhost:5000/ -H "Authorization: Bearer hello"`

* curl -X POST http://localhost:5000/documents/get_index -H "Content-Type: application/json" -d '{"system":"aple-demo-moodle", "course_id":0, "activity_id":7, "activity_type":"activity_longpage"}'



## License

This project is licensed under MIT License. For more details, please refer to the [LICENSE](https://choosealicense.com/licenses/mit/).

## Citation

```Seidel, N. (2025). RAGWEED: Retrieval Augmented Generation Webservice for Education. https://github.com/CATALPAresearch/rag-webservice ```