import lmstudio as lms
from langchain_core.embeddings import Embeddings

class LMStudioEmbeddings(Embeddings):
    """
    Special class to allow for using LM Studio embedding models with LangChain.
    """
    def __init__(self, model_name="text-embedding-nomic-embed-text-v1.5@q4_k_s"):
        self.model = lms.embedding_model(model_name)

    def embed_documents(self, texts):
        return [self.model.embed(text) for text in texts]

    def embed_query(self, text):
        return self.model.embed(text)