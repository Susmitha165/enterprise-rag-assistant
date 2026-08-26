from typing import List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.ingestion import ingest_document
from app.vector_store import (
    build_vector_store,
    search_documents,
)


class RAGPipeline:
    def __init__(self):
        self.vector_store = None
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )

    def ingest(self, file_path: str) -> int:
        """
        Load a document, chunk it, and build the vector store.
        Returns the number of chunks created.
        """
        chunks = ingest_document(file_path)
        self.vector_store = build_vector_store(chunks)
        return len(chunks)

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant chunks from the vector store.
        """
        if self.vector_store is None:
            raise ValueError(
                "No documents have been ingested yet."
            )

        return search_documents(
            vector_store=self.vector_store,
            query=query,
            k=k,
        )

    def answer(self, query: str) -> dict:
        """
        Retrieve context and generate a grounded answer.
        """
        documents = self.retrieve(query)

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""
You are an enterprise document assistant.

Answer the question using only the provided context.

If the answer is not supported by the context,
say that the information is not available.

Context:
{context}

Question:
{query}
"""

        response = self.llm.invoke(prompt)

        sources = [
            document.metadata.get("source", "unknown")
            for document in documents
        ]

        return {
            "answer": response.content,
            "sources": list(dict.fromkeys(sources)),
        }
