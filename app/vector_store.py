from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model="nomic-embed-text"
    )


def build_vector_store(documents: List[Document]) -> FAISS:
    """
    Create a FAISS vector store from document chunks
    using local Ollama embeddings.
    """
    if not documents:
        raise ValueError("No documents provided to build the vector store.")

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    return vector_store


def search_documents(
    vector_store: FAISS,
    query: str,
    k: int = 4,
) -> List[Document]:
    """
    Retrieve the most relevant document chunks for a query.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    return vector_store.similarity_search(
        query=query,
        k=k,
    )


def save_vector_store(
    vector_store: FAISS,
    directory: str = "data/faiss_index",
) -> None:
    """
    Save the FAISS index locally.
    """
    vector_store.save_local(directory)


def load_vector_store(
    directory: str = "data/faiss_index",
) -> FAISS:
    """
    Load an existing FAISS vector store.
    """
    embeddings = get_embeddings()

    return FAISS.load_local(
        directory,
        embeddings,
        allow_dangerous_deserialization=True,
    )
