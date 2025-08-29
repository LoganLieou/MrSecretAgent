from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
import lmstudio as lms

# TODO I think I can just use this for embeddings in create_index too
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


store = FAISS.load_local("faiss_index", LMStudioEmbeddings(), allow_dangerous_deserialization=True)

llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm-studio"
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=store.as_retriever()
)

if __name__ == "__main__":
    print("Type 'exit' or 'quit' to end the chat.\n")
    chat_history = []
    while True:
        query = input("You: ")
        if query.strip().lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break
        context = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])
        full_query = f"{context}\nQ: {query}" if chat_history else query
        response = chain.run(full_query)
        print(f"Agent: {response}\n")
        chat_history.append((query, response))