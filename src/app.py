import os
import asyncio
import json
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from flasgger import Swagger, swag_from
import logging
from utils.require_salt import require_salt
from utils.logging import setup_logging
from utils.document_model import RAG_Document

from RAG_Manager import RAG_Manager
from LLM_Manager import LLM_Manager

setup_logging()


UPLOAD_FOLDER = 'data/uploads/'
API_TOKEN = os.getenv("API_TOKEN", "")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app = Flask(__name__)
swagger = Swagger(app)
cors = CORS(app)

# init
rag_manager = RAG_Manager()


# ROUTES

@app.route('/', methods=['GET'])
@cross_origin()
#@require_salt(API_TOKEN)
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
    activity_type = request.form.get('activity_type', 'unknown')
    activity_name = request.form.get('activity_name', 'unknown')
    activity_id = request.form.get('activity_id', 'unknown')

    if file and file.filename.endswith('.pdf'):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        doc = RAG_Document(
            system=system,
            course_id=course_id,
            activity_type=activity_type,
            activity_name=activity_name,
            file=filepath,
        )
        documents = [doc.dict(exclude_none=True)]
        documents[0][activity_type] = activity_id
        document_index = asyncio.run(rag_manager.rt.add_documents(documents))
        
        return jsonify({
            'message': 'File received successfully', 
            'document_index': document_index,
            'file': file.filename
            }), 200
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/documents/delete_index', methods=['POST'])
@cross_origin()
@require_salt(API_TOKEN)
@swag_from('specs/documents_delete_index.yml')
def delete_index():
    """..."""
    try:
        data = request.get_json()
        system = data.get('system', None)
        course_id = data.get('course_id', -1)
        activity_type = data.get('activity_type', None)
        activity_id = data.get('activity_id', None)

        if system == None  or course_id == -1 or activity_type == None or activity_id == None:
            return jsonify({"success": False, "message": "Missing parameters"}), 400

        rag_manager.rt.delete_document(
            system=system, 
            course_id=course_id,
            activity_type=activity_type,
            activity_id=activity_id
            )

        return jsonify({
            "success": True
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500



@app.route('/documents/documents_by_course', methods=['POST'])
@cross_origin()
@swag_from('specs/documents_by_course.yml')
def get_documents_by_course():
    try:
        data = request.get_json()
        system = data.get('system', "")
        course_id = data.get('course_id', -1)

        if system == ""  or course_id == -1:
            return jsonify({"success": False, "message": "Missing parameters"}), 400

        documents = rag_manager.rt.get_documents_by_course(system=system, course_id=course_id)

        return jsonify({
            "success": True,
            "documents": documents
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500
    

@app.route('/documents/list', methods=['POST'])
@cross_origin()
@swag_from('specs/documents_list.yml')
def get_documents():
    try:
        documents = rag_manager.rt.get_documents_by_course(system=None, course_id=None)

        return jsonify({
            "success": True,
            "documents": documents
        }), 200

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
    try:
        data = request.get_json()
        system = data.get('system', None)
        course_id = data.get('course_id', -1)
        activity_type = data.get('activity_type', None)
        activity_id = data.get('activity_id', None)

        if system == None  or course_id == -1 or activity_type == None or activity_id == None:
            return jsonify({"success": False, "message": "Missing parameters"}), 400

        document = rag_manager.rt.get_index(
            system=system, 
            course_id=course_id,
            activity_type=activity_type,
            activity_id=activity_id
            )

        return jsonify({
            "success": True,
            "document": document
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route('/documents/update_index', methods=['POST'])
@cross_origin()
@swag_from('specs/documents_update_index.yml')
def update_index():
    """
    Updates a document from the document store. Currently not implemented.
    """
    return jsonify({'msg': 'none'}), 200




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
    

