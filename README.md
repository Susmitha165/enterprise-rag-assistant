# Enterprise RAG Assistant

A production-style Retrieval-Augmented Generation (RAG) application designed for enterprise document question answering.

## Features

- Document ingestion and chunking
- Embedding generation and vector search
- Hybrid retrieval using semantic and lexical search
- Reranking for better retrieval quality
- Grounded answers with source citations
- FastAPI-based backend
- LLM evaluation using RAGAS
- Basic guardrails and validation
- Dockerized deployment
- Automated testing

## Tech Stack

- Python
- FastAPI
- LangChain
- FAISS / pgvector
- OpenAI or compatible LLM provider
- RAGAS
- Docker
- Pytest

## Architecture

User Query  
→ FastAPI  
→ Query Processing  
→ Hybrid Retrieval  
→ Reranking  
→ Context Assembly  
→ LLM Generation  
→ Validation  
→ Response with Citations

## Project Status

Currently under active development.

## Planned Improvements

- Add metadata filtering
- Add hybrid BM25 + semantic retrieval
- Add PII redaction
- Add evaluation dashboard
- Add production monitoring
- Add CI/CD
