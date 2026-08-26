from typing import List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.config import OPENAI_API_KEY
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
            api_key=OPENAI_API_KEY,
        )

    def ingest(self, file_path: str) -> int:
        """
        Load a document, chunk it, generate embeddings,
        and build the FAISS vector store.
        """
        chunks = ingest_document(file_path)

        self.vector_store = build_vector_store(chunks)

        return len(chunks)

    def retrieve(
        self,
        query: str,
        k: int = 4,
    ) -> List[Document]:
        """
        Retrieve the most relevant document chunks.
        """
        if self.vector_store is None:
            raise ValueError(
                "No documents have been ingested yet."
            )

        documents = search_documents(
            vector_store=self.vector_store,
            query=query,
            k=k,
        )

        return documents

    def answer(self, query: str) -> dict:
        """
        Retrieve relevant context and generate
        a grounded response using the LLM.
        """
        if not query.strip():
            raise ValueError("Question cannot be empty.")

        documents = self.retrieve(query)

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""
You are an enterprise document question-answering assistant.

Use only the context provided below to answer the question.

Do not invent information.

If the answer cannot be found in the provided context,
respond with:

"The information is not available in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(prompt)

        sources = []

        for document in documents:
            source = document.metadata.get(
                "source",
                "unknown",
            )

            if source not in sources:
                sources.append(source)

        return {
            "answer": response.content,
            "sources": sources,
        }
