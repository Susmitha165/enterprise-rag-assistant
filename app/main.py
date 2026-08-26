from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.rag_pipeline import RAGPipeline

app = FastAPI(
    title="Enterprise RAG Assistant",
    description="Production-style RAG API for enterprise document question answering.",
    version="0.2.0",
)

rag_pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    file_path: str


@app.get("/")
def root():
    return {
        "service": "Enterprise RAG Assistant",
        "status": "running",
        "version": "0.2.0",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ingest")
def ingest_document(request: IngestRequest):
    try:
        chunks = rag_pipeline.ingest(request.file_path)

        return {
            "status": "success",
            "chunks_created": chunks,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@app.post("/query")
def query_rag(request: QueryRequest):
    try:
        result = rag_pipeline.answer(request.question)

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )
