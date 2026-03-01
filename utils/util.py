import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from configs.config import DOCS_PATH, get_embedding_model, PERSIST_DIRECTORY


def directory_loader(doc_path = None, glob="test.txt"):
    path = doc_path or DOCS_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory '{path}' does not exist.")

    return DirectoryLoader(
        path=doc_path or DOCS_PATH,
        glob=glob,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

def create_vector_store(chunks):
    """Create and persist chromaDB vector store"""
    print("Creating vector store...")

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        persist_directory=PERSIST_DIRECTORY,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print("Vector store created successfully.")