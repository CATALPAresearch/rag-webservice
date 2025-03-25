# rag-webservice
RAG-Webservice is a middleware for an Ollama LLM server and a Moodle plugin called Openchat.

## Features
* ...


## Getting started

**Docker setup**
* Build Docker container: `docker build --no-cache -t rag-webservice .`
* Optional: Check for vulnerabilities: `docker scout quickview `
* Start the container: `docker run --name rag-webservice -p 5000:5000 rag-webservice`

* `docker compose up --build`
* `docker compose up -d`

**Local setup**
* Run `poetry install`
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



## License

This project is licensed under the GPL License. For more details, please refer to the [LICENSE](./LICENSE) file.

## Citation

```Seidel, N. (2025). RAGOUT: Retrieval Augmented Generation for Online Universal Testing in Online Learning```