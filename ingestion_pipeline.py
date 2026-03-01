import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.util import directory_loader, create_vector_store
from configs.config import DOCS_PATH


def load_documents():
    """Load all text files from the docs directory"""
    print(f"Loading documents from {DOCS_PATH}...")

    # check if directory exists or not
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"{DOCS_PATH} directory not found.")

    # load the directory
    loader = directory_loader()

    # load() provides the list of langchain documents [Document(page_content="", metadata={'source': 'docs/file.txt'}), Document(...)]
    documents = loader.load()

    if not documents:
        raise ValueError("No documents found in docs folder.")

    print(f"Loaded {len(documents)} documents.")
    return documents

# def split_documents(documents):
#     """Split documents into smaller chunks with overlapping"""
#     splitter = CharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=100,
#     )
#
#     # create chunks
#     chunks = splitter.split_documents(documents)
#     return chunks

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller, semantically coherent chunks.

    Uses RecursiveCharacterTextSplitter for smarter splitting:
    - Tries to split on paragraphs, sentences, then characters
    - Keeps context with chunk_overlap
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, # counts characters
        separators=["\n\n", "\n", ".", "!", "?", " ", ""], # split priority
    )

    chunks = splitter.split_documents(documents)
    return chunks

def main():
    print("Starting ingestion pipeline...")

    # 1. Load files
    docs = load_documents()
    # 2. Chunk files
    chunks = split_documents(docs)
    # 3. Embeddings and Storing in vector DB
    create_vector_store(chunks)

    print("Ingestion completed.")

if __name__ == "__main__":
    main()