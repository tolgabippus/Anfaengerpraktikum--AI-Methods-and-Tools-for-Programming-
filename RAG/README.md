Minimal RAG System 
A lightweight Retrieval-Augmented Generation (RAG) pipeline built with pure Python. No LangChain, no vector database, no embedding API.
Documents → Chunking → TF-IDF Index
                            ↓
Query → Retrieval (cosine similarity) → Top-K chunks
                            ↓
          Prompt = chunks + query → LLM → Answer

How it Works: 
Chunking: splits documents into overlapping word-level windows
TF-IDF indexing: builds a term-frequency/inverse-document-frequency index from scratch
Retrieval: ranks chunks by cosine similarity to the query vector
Generation: passes retrieved context to Claude for a grounded answer

Why RAG?
Standard LLMs hallucinate when asked about private or recent information. RAG grounds the model's answer in actual retrieved documents, making responses more factual and verifiable.
