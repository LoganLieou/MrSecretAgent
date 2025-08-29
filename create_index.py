from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import argparse
import lmstudio as lms
from chat import LMStudioEmbeddings

store = None

embeddings = LMStudioEmbeddings() # lms.embedding_model("text-embedding-nomic-embed-text-v1.5@q4_k_s")

"""
def get_embeddings(texts):
    return [embeddings.embed(text) for text in texts]
"""

def split_paragraphs(rawText):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_text(rawText)

def process_pdf(file_path):
    global store
    pdf = PdfReader(file_path)
    raw_text = ''
    for page in pdf.pages:
        raw_text += page.extract_text() + '\n'
    
    paragraphs = split_paragraphs(raw_text)

    # vectors = get_embeddings(paragraphs)
    # docs = [Document(page_content=para) for para in paragraphs]
    
    if store is None:
        # store = FAISS.from_embeddings(zip(paragraphs, vectors), docs)
        store = FAISS.from_texts(paragraphs, embeddings)
    else:
        # new_store = FAISS.from_embeddings(zip(paragraphs, vectors), docs)
        new_store = FAISS.from_texts(paragraphs, embeddings)
        store.merge_from(new_store)

    store.save_local("faiss_index") # save the index locally

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Process a PDF and create/update a FAISS index.")
    argparser.add_argument("pdf_path", type=str, help="Path to the PDF file to process.")
    args = argparser.parse_args()
    process_pdf(args.pdf_path)