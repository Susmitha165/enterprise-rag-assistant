from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Enterprise RAG Assistant",
    description="Production-style RAG API for enterprise document question answering.",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "service": "Enterprise RAG Assistant",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/query")
def query_rag(request: QueryRequest):
    return {
        "question": request.question,
        "answer": "RAG pipeline will be connected in the next implementation step.",
        "sources": [],
    }
