# Day 4: Embeddings + pgvector

Generate embeddings with OpenAI, store in PostgreSQL using pgvector, run similarity search.

## Why pgvector over FAISS?
- **Production-ready**: runs inside Postgres you already know
- **No extra infra**: no separate vector DB to manage
- **SQL-native**: filter by metadata with normal WHERE clauses
- **HNSW indexing**: fast approximate nearest neighbor at scale

## Quick Start

```bash
# 1. Start pgvector (Postgres 16 + vector extension)
docker compose up -d

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Seed sample documents (generates embeddings + inserts)
python3 embeddings_pgvector.py --mode seed

# 4. Search!
python3 embeddings_pgvector.py --mode search --query "how do caches work"
python3 embeddings_pgvector.py --mode search --query "kubernetes autoscaling" --top-k 3

# 5. Inspect stored docs
python3 embeddings_pgvector.py --mode inspect
```

## Key Concepts

| Concept | What it means |
|---------|---------------|
| **Embedding** | A dense vector (1536 floats) that captures semantic meaning |
| **text-embedding-3-small** | OpenAI's cheapest embedding model ($0.02/1M tokens) |
| **Cosine distance (`<=>`)** | How "far apart" two vectors are (0 = identical) |
| **HNSW index** | Graph-based approximate nearest neighbor (fast, good recall) |
| **Batch embedding** | Embed multiple texts in one API call (cheaper + faster) |

## pgvector Distance Operators

```sql
<=>   -- cosine distance (most common for text)
<->   -- L2 / euclidean distance
<#>   -- inner product (negative, for max inner product search)
```

## Architecture

```
User Query
    ↓
OpenAI text-embedding-3-small  →  [1536-dim vector]
    ↓
PostgreSQL + pgvector
    ↓  ORDER BY embedding <=> query_vector
Top-K similar documents
```
