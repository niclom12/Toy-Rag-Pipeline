# Has two api endpoints, upload doc-> which takes in the document and then chunks and emmbeds it and then Query-> which takes in prompt and returns response 
from flask import Flask, request, jsonify
from Rag.Gen import Gen
from Rag.VectorDatabase import VectorDatabase
from Rag.Converters import Converter
from werkzeug.utils import secure_filename
import os


# Initialise database, embedding model and Groq client
vector = VectorDatabase("docs", 384, "./database")
convert = Converter()
gen = Gen()


app = Flask(__name__)

DOCUMENTS_FOLDER = 'documents'
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'md'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_doc', methods=['POST'])
def upload_doc():
    """
    Upload a document to the system, save it to the 'documents' folder, 
    and then process and insert its contents into the vector database.
    
    Expects a file and 'doc_name' field in the request.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    doc_name = request.form.get('doc_name')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not doc_name or doc_name.strip() == '':
        return jsonify({"error": "'doc_name' is required and cannot be empty"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format. Allowed formats: txt, pdf, md."}), 400

    if not file.filename:
        return jsonify({"error": "No filename provided"}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(DOCUMENTS_FOLDER, filename)

    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500
    
    try:
        chunks = convert.convert_to_chunks(file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to read or parse document: {str(e)}"}), 500
            
    try:
        vector.insert(chunks)
    except Exception as e:
        return jsonify({"error": f"Failed to insert into database: {str(e)}"}), 500

    return jsonify({"message": "Document uploaded and processed successfully!"}), 200

    

@app.route('/query', methods=['POST'])
def query():
    """
    Query the RAG system with a prompt by retrieving context from the vector database 
    and using the Groq LLM to generate a response.
    
    Expects a JSON payload with 'prompt'.
    """
    try:
        data = request.get_json()
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "'prompt' is required"}), 400
        query_vector = convert.model.encode(prompt)
        top_k_chunks = vector.similarity_search(query_vector, top_k=5)
        if not top_k_chunks or top_k_chunks[0].get("doc_name") is None:
            return jsonify({"response": "No relevant documents found."}), 200

        context = "\n".join([chunk["chunk_text"] for chunk in top_k_chunks])
        response = gen.process_and_respond(context, prompt)
        return jsonify({"response": response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
if __name__ == '__main__':
    app.run(debug=False)