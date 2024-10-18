from flask import Flask, render_template, request, jsonify
import os
from model.model import LLM
from RAG.vectordb import RAG
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.environ.get("DATA_PATH") # Change DATA_PATH in .env
CHROMA_PATH = os.environ.get("CHROMA_PATH") # Change CHROMA_PATH in .env

app = Flask(__name__)

# Instantiate LLM and vectordb
llm = LLM()
db = RAG()

# Delete collection for testing
# db.chroma_client.delete_collection(name="my_documents")

# Choose vectordb collection
collection = db.chroma_client.get_or_create_collection(name="my_documents")

# Add the document(s) to the ChromaDB collection
# pdf_files = [os.path.join(file_path, file) for file in os.listdir(file_path) if file.endswith('.pdf')]
# file_loader.add_documents(pdf_files=pdf_files, collection=collection)

# Run html template for UI
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['POST'])
def send():
    # Get user input from the request
    user_input = request.json.get('query', '')

    # Employing Query Expansion to get clearer query. Helpful when user's questions can be vague
    query_expansion_prompt = f"""Expand this query to include additional relevant terms and synonyms: {user_input}
    """
    
    expanded_answer = llm.model.invoke(
        model=llm.model.model,
        input=query_expansion_prompt
    )
    
    # Concatenate original input and expanded input
    combined_query = f"{user_input} {expanded_answer.content}"
    
    # RAG retrieval: retrieve relevant context for the query
    # Use embedded hypothetical answer as query
    retriever = db.retrieve(collection=collection, query=combined_query)
    retrieved_data = "\n\n".join(retriever)

    # Create a new prompt using the retrieved context and user input
    new_prompt = f"""Using this data: {retrieved_data}. Respond to this message: {user_input}.
    You are a QA assistant. You answers user's question with given data.
    Don't mention the data for user, only give them the answer with some context.
    If the data does not make sense, you are free to inteprete truthfully."""

    # LLM invocation: pass the new prompt to the model
    output = llm.model.invoke(
        model=llm.model.model,
        input=new_prompt
    )

    # Extract the AI response
    ai_response = output.content

    # Prepare the response data to send back to the front-end
    return jsonify({
        'user_message': f"You: {user_input}",
        'response': ai_response
    })

# Serve the FAQ data dynamically
@app.route('/faq', methods=['GET'])
def get_faq():
    faq_data = {
        'questions': [
            'How to get money in Monopoly?',
            'How jail works?',
            'How auction works?',
            'What is the winning goal?',
            'How to trade with another player?'
        ]
    }
    return jsonify(faq_data)

# Start app
if __name__ == '__main__':
    # host = 0.0.0.0 for LAN access
    app.run(host='0.0.0.0', port=5000, debug=True)