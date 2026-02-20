# Day 5: RAG — Retrieval-Augmented Generation

Chunk text → retrieve from pgvector → inject into prompt → generate grounded answer.

## Why RAG?
- **Prevents hallucination**: model answers from YOUR data, not training data
- **No fine-tuning needed**: just put relevant context in the prompt
- **Updatable**: add new docs to Postgres anytime, no retraining
- **Auditable**: you can see exactly which sources the answer came from

## Prerequisites

Day 4's pgvector must be running with documents seeded:
```bash
cd ../day4-embeddings-pgvector
docker compose up -d
python3 embeddings_pgvector.py --mode seed
```

## Quick Start

```bash
pip3 install -r requirements.txt

# Full RAG pipeline
python3 rag.py --query "how do caches improve performance"

# Show retrieved context before the answer
python3 rag.py --query "explain circuit breakers" --show-context

# Adjust retrieval
python3 rag.py --query "message queues" --top-k 3 --min-similarity 0.5

# Demo chunking (no API call)
python3 rag.py --mode chunk

# Ingest a text file (chunk → embed → store)
python3 rag.py --mode ingest --file ./my_notes.txt
```

## RAG Pipeline

```
                  ┌──────────────┐
User Question ──→ │ Embed Query  │
                  └──────┬───────┘
                         ↓
                  ┌──────────────┐
                  │  pgvector    │ ← cosine similarity search
                  │  top-k docs  │
                  └──────┬───────┘
                         ↓
                  ┌──────────────┐
                  │  Rerank      │ ← filter by min similarity
                  └──────┬───────┘
                         ↓
                  ┌──────────────────────────────────┐
                  │  Build Prompt                     │
                  │  System: "answer from context"    │
                  │  User: [context] + [question]     │
                  └──────┬───────────────────────────┘
                         ↓
                  ┌──────────────┐
                  │  GPT-4o-mini │ → Grounded Answer
                  └──────────────┘
```

## Key Concepts

| Concept | Detail |
|---------|--------|
| **Chunking** | Split long docs into token-sized pieces with overlap |
| **Overlap** | ~50 tokens shared between chunks to preserve context at boundaries |
| **Top-k retrieval** | Get the k most similar chunks from pgvector |
| **Reranking** | Filter/sort by relevance score (here: cosine similarity cutoff) |
| **Context injection** | Place retrieved text in the prompt so the model stays grounded |
| **Source citation** | Tell the model to reference [Source N] for traceability |

## What would make this production-grade?

1. **Cross-encoder reranking** (Cohere Rerank, sentence-transformers)
2. **Hybrid search** (combine vector + keyword/BM25)
3. **Metadata filtering** (WHERE topic = 'caching' AND level = 'beginner')
4. **Streaming responses** for lower TTFB
5. **Evaluation** (RAGAS, human annotation)
