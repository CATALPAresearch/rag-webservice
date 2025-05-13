import os
import asyncio
import json
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from flasgger import Swagger, swag_from
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from RAG_Manager import RAG_Manager
from LLM_Manager import LLM_Manager

app = Flask(__name__)
swagger = Swagger(app)
cors = CORS(app)

UPLOAD_FOLDER = 'data/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# init
rag_manager = RAG_Manager()


@app.route('/', methods=['GET'])
@cross_origin()
@swag_from('specs/root.yml')
def root():
    """
    Say hello in the browser
    """
    return jsonify({
        'message': 'Hello, your RAG-Webservice is operating will but will not be accessible as a webpage (;'
        }), 200


@app.route('/documents/create_index', methods=['POST'])
@cross_origin()
@swag_from('specs/documents_create_index.yml')
def create_document_index():
    """
    Test: curl -X POST -F file=@"/Users/nise/Downloads/INFO_ZLB___98_Didactics_of_Computer_Science_EN_20250212.pdf" http://localhost:5000/process
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part '+str(len(request.files))}), 400
        
    file = request.files['file']
    system = request.form.get('system', 'unknown')
    course_id = request.form.get('course_id', 'unknown')
    document_type = request.form.get('document_type', 'unknown')
    document_id = request.form.get('document_id', 'unknown')

    if file and file.filename.endswith('.pdf'):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        documents = [
            {
                "system": system,
                "course_id": course_id,
                "file": filepath,
            }
        ]
        documents[0][document_type] = document_id
        document_index = asyncio.run(rag_manager.rt.add_documents(documents))
        
        return jsonify({
            'message': 'File received successfully', 
            'document_index': document_index,
            'file': file.filename
            }), 200
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/documents/delete_index', methods=['POST'])
@cross_origin()
@swag_from('specs/documents_delete_index.yml')
def delete_index():
    """..."""
    pass



@app.route('/documents/documents_by_course', methods=['POST'])
@cross_origin()
@swag_from('specs/documents_by_course.yml')
def get_documents_by_course():
    try:
        data = request.get_json()
        system = data.get('system')
        course_id = data.get('course_id')

        if not system or not course_id:
            return jsonify({"success": False, "message": "Missing parameters"}), 400

        documents = rag_manager.rt.get_documents_by_course(system=system, course_id=course_id)

        return jsonify({
            "success": True,
            "documents": documents
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route('/documents/get_index', methods=['POST'])
@cross_origin()
@swag_from('specs/documents_get_index.yml')
def get_index():
    """..."""
    pass

@app.route('/documents/update_index', methods=['POST'])
@cross_origin()
@swag_from('specs/documents_update_index.yml')
def update_index():
    """..."""
    pass




@app.route('/llm/models/list', methods=['POST'])
@cross_origin()
@swag_from('specs/llm_models_list.yml')
def get_models():
    llmm = LLM_Manager()
    response = llmm.get_model_names()
    return jsonify({
        'message': '', 
        'success': True,
        'data': response
        }), 200


@app.route('/llm/query', methods=['POST'])
@cross_origin()
@swag_from('specs/llm_query.yml')
def query_llm():
    """
    Query a large language model (LLM) with a prompt.
    """
    prompt = request.form.get('prompt', 'unknown')
    
    response = rag_manager.process_simple_question(prompt)
        
    return jsonify({
        'message': '', 
        'response': response
        }), 200


@app.route('/llm/query_documents', methods=['POST'])
@cross_origin()
@swag_from('specs/llm_query_documents.yml')
def query_rag():
    data = request.get_json()
    #document_index = request.form.get('document_index', None)
    model = data.get('model', 'unknown') # 'phi3:latest'
    prompt = data.get('prompt', 'unknown')
    filter = data.get('filter', {})
    
    """
    filter = {
            'system': ['aple-demo-moodle'],
            'courses': [0],
            'activity_longpage': [1,7],
        }
    """
    logger.info('prompt: '+prompt)
    #matched_docs, response = asyncio.run(rag_manager.process_rag_question(prompt, document_index))
    matched_docs, response = asyncio.run(rag_manager.process_rag_question(prompt, model=model, filter=filter))
    
    logger.info("response")
    logger.info(response)
    return jsonify({
        'response': response
        }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    

