# Ollama_local_chatbot
 An application using Llama3.1 8B Instruct model with RAG for QA chatbot.
 Scope:
 - Use Ollama Llama 3.1 8B Instruct for local LLM model. You can use other proprietery models by their API.
 - For embedding model use Ollama mxbai-embed-large. You can use openai embedding or any huggingface embedding model.
 - Create RAG with Chroma to store and retrieve PDF files.
 - Finally publish as webapp for demonstration purposes.

First create .env file and specify DATA_PATH and CHROMA_PATH for your documents files location and where you want to save chromadb collections respectively.
Then run application via app.py and use localhost.
