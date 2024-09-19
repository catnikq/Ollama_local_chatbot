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
# db.add_documents(pdf_files=[file_path], collection=collection)

# Run html template for UI
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['POST'])
def send():
    # Get user input from the request
    user_input = request.json.get('query', '')

    # RAG retrieval: retrieve relevant context for the query
    retriever = db.retrieve(collection=collection, query=user_input)
    retrieved_data = "\n\n".join(retriever)

    # Create a new prompt using the retrieved context and user input
    new_prompt = f"""Using this data: {retrieved_data}. Respond to this message: {user_input}.
    These data are related to some documents. Your task is to answer user's question about the content.
    Don't mention the data for user, only give them the answer.
    You are free to interpret the information as long as it is truthful."""

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
        'response': f"AI: {ai_response}"
    })

# Start app
if __name__ == '__main__':
    app.run(debug=True)