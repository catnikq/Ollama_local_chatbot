from langchain_ollama import OllamaEmbeddings

class EmbeddingModel:
    def __init__(self) -> None:
        self.embedding_model = OllamaEmbeddings(model='mxbai-embedding-large')
        
    def generate_embedding(self, document):
        embedding = self.embedding_model.embed_documents(document)
        return embedding