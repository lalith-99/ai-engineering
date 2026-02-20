"""
Day 5: RAG (Retrieval-Augmented Generation)
=============================================
Chunk text -> embed -> retrieve top-k from pgvector -> inject into prompt -> generate answer.

Covers: text chunking (sentence-aware with overlap), vector retrieval,
        similarity reranking, RAG prompt construction, file ingestion.

Prerequisites:
  - Day 4's pgvector container running (docker compose up -d in day4 folder)
  - Documents seeded (python3 ../day4-embeddings-pgvector/embeddings_pgvector.py --mode seed)

Usage:
  python3 rag.py --query "how do caches improve performance"
  python3 rag.py --query "explain circuit breakers" --top-k 3
  python3 rag.py --query "what is a message queue" --show-context
  python3 rag.py --mode chunk --text "Your long document text here..."
  python3 rag.py --mode ingest --file ./sample_doc.txt

Production improvements covered in guide but not here:
  - Hybrid search (vector + BM25 keyword) for exact-match terms
  - Cross-encoder reranking (Cohere Rerank, BGE Reranker v2)
  - Agentic RAG (agent decides when/what to retrieve)
  - Semantic chunking (embedding-based topic detection)
  - RAGAS/DeepEval evaluation in CI/CD
"""

import os
import json
import argparse
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("OpenAI SDK not installed. Run: pip3 install -r requirements.txt")

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    raise SystemExit("psycopg2 not installed. Run: pip3 install -r requirements.txt")

try:
    from pgvector.psycopg2 import register_vector
except ImportError:
    raise SystemExit("pgvector Python package not installed. Run: pip3 install -r requirements.txt")

try:
    import tiktoken
except ImportError:
    raise SystemExit("tiktoken not installed. Run: pip3 install -r requirements.txt")

import numpy as np


# ========== CONFIG ==========

DB_CONFIG = {
    "host": os.getenv("PGVECTOR_HOST", "localhost"),
    "port": int(os.getenv("PGVECTOR_PORT", "5433")),
    "dbname": os.getenv("PGVECTOR_DB", "embeddings"),
    "user": os.getenv("PGVECTOR_USER", "llm"),
    "password": os.getenv("PGVECTOR_PASSWORD", "llm_pass"),
}

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 300          # tokens per chunk
CHUNK_OVERLAP = 50        # overlapping tokens between chunks
MAX_CONTEXT_TOKENS = 2000 # max tokens for retrieved context in prompt


# ========== CLIENT HELPERS ==========


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY. Export it, then 'source ~/.zshrc'.")
    return OpenAI(api_key=api_key)


def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        register_vector(conn)
        return conn
    except psycopg2.OperationalError as e:
        raise SystemExit(
            f"Cannot connect to Postgres: {e}\n"
            "Make sure pgvector is running (Day 4): docker compose up -d"
        )


# ========== TEXT CHUNKING ==========


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens using tiktoken (accurate for OpenAI models)."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks by token count.

    Why overlap?
      Sentences that straddle chunk boundaries would lose context.
      Overlap ensures each chunk has neighboring context.

    Strategy: sentence-aware splitting to avoid cutting mid-sentence.
    """
    # Split into sentences first
    sentences = []
    for line in text.replace("\n", " ").split(". "):
        s = line.strip()
        if s:
            if not s.endswith("."):
                s += "."
            sentences.append(s)

    chunks = []
    current_chunk: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(" ".join(current_chunk))

            # Keep overlap: walk backwards until we have ~overlap tokens
            overlap_chunk: List[str] = []
            overlap_tokens = 0
            for s in reversed(current_chunk):
                t = count_tokens(s)
                if overlap_tokens + t > chunk_overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_tokens += t

            current_chunk = overlap_chunk
            current_tokens = overlap_tokens

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ========== RETRIEVAL ==========


def embed_text(client: OpenAI, text: str) -> List[float]:
    """Generate embedding for a single text."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def retrieve_similar(
    conn, query_embedding: List[float], top_k: int = 5
) -> List[Dict[str, Any]]:
    """Retrieve top-k similar documents from pgvector with cosine similarity."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                id,
                content,
                metadata,
                1 - (embedding <=> %s) AS similarity
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (np.array(query_embedding), np.array(query_embedding), top_k),
        )
        return cur.fetchall()


def rerank_by_score(
    results: List[Dict[str, Any]], min_similarity: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Simple reranking: filter out low-similarity results.

    In production you'd use a cross-encoder reranker (e.g., Cohere rerank,
    sentence-transformers cross-encoder), but score-based filtering is
    a practical first step.
    """
    filtered = [r for r in results if r["similarity"] >= min_similarity]
    # Already sorted by similarity from SQL, but be explicit
    filtered.sort(key=lambda x: x["similarity"], reverse=True)
    return filtered


# ========== RAG PROMPT CONSTRUCTION ==========


def build_rag_prompt(query: str, context_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Build the chat messages for RAG:
      1. System: defines role + rules (stay grounded, cite sources)
      2. User: context block + question

    Key principle: the model should ONLY use the provided context.
    This prevents hallucination and keeps answers verifiable.
    """
    # Build context block
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        sim = doc["similarity"]
        meta = doc["metadata"] if isinstance(doc["metadata"], dict) else json.loads(doc["metadata"])
        topic = meta.get("topic", "unknown")
        context_parts.append(f"[Source {i} | topic={topic} | relevance={sim:.3f}]\n{doc['content']}")

    context_block = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable technical assistant. "
                "Answer the user's question using ONLY the provided context. "
                "If the context doesn't contain enough information, say so clearly. "
                "Cite which source(s) you used (e.g., [Source 1]). "
                "Be concise and precise."
            ),
        },
        {
            "role": "user",
            "content": (
                f"### Retrieved Context\n\n{context_block}\n\n"
                f"---\n\n"
                f"### Question\n{query}\n\n"
                f"Answer based on the context above:"
            ),
        },
    ]

    return messages


# ========== RAG PIPELINE ==========


def run_rag(
    client: OpenAI,
    conn,
    query: str,
    top_k: int = 5,
    show_context: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 500,
    min_similarity: float = 0.3,
) -> str:
    """
    Full RAG pipeline:
      1. Embed the query
      2. Retrieve top-k documents from pgvector
      3. Rerank / filter by similarity score
      4. Build prompt with context
      5. Generate answer with GPT
    """
    print(f"\n{'='*70}")
    print(f"RAG Query: \"{query}\"")
    print(f"{'='*70}")

    # Step 1: Embed query
    t0 = time.time()
    query_embedding = embed_text(client, query)
    embed_ms = (time.time() - t0) * 1000
    print(f"\n[Step 1] Query embedded in {embed_ms:.0f}ms ({EMBEDDING_MODEL})")

    # Step 2: Retrieve
    t0 = time.time()
    raw_results = retrieve_similar(conn, query_embedding, top_k)
    retrieve_ms = (time.time() - t0) * 1000
    print(f"[Step 2] Retrieved {len(raw_results)} candidates in {retrieve_ms:.0f}ms")

    if not raw_results:
        return "No documents found. Did you seed the database? (Day 4: --mode seed)"

    # Step 3: Rerank (score-based filter; production would use cross-encoder reranker)
    ranked = rerank_by_score(raw_results, min_similarity)
    dropped = len(raw_results) - len(ranked)
    print(f"[Step 3] After reranking: {len(ranked)} docs (dropped {dropped} below {min_similarity} similarity)")

    if not ranked:
        return "No sufficiently relevant documents found for this query."

    # Show context if requested
    if show_context:
        print(f"\nRetrieved Context:")
        for i, doc in enumerate(ranked, 1):
            sim = doc["similarity"]
            level = "HIGH" if sim >= 0.7 else "MED" if sim >= 0.5 else "LOW"
            print(f"  [{level:>4}] [Source {i}] sim={sim:.4f}: {doc['content'][:100]}...")

    # Step 4: Build prompt
    messages = build_rag_prompt(query, ranked)
    context_tokens = count_tokens(messages[1]["content"])
    print(f"[Step 4] Prompt built ({context_tokens} context tokens)")

    if context_tokens > MAX_CONTEXT_TOKENS:
        print(f"   Warning: Context exceeds {MAX_CONTEXT_TOKENS} tokens, truncating to fit")
        # Trim docs from the bottom (least relevant)
        while context_tokens > MAX_CONTEXT_TOKENS and len(ranked) > 1:
            ranked.pop()
            messages = build_rag_prompt(query, ranked)
            context_tokens = count_tokens(messages[1]["content"])

    # Step 5: Generate
    t0 = time.time()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    gen_ms = (time.time() - t0) * 1000
    answer = response.choices[0].message.content
    usage = response.usage

    print(f"[Step 5] Answer generated in {gen_ms:.0f}ms")
    print(f"   Tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}")

    print(f"\n{'='*70}")
    print(f"ANSWER:\n")
    print(answer)
    print(f"\n{'='*70}")

    return answer


# ========== CHUNK COMMAND ==========


def cmd_chunk(text: str):
    """Demonstrate text chunking (no API call needed)."""
    chunks = chunk_text(text)
    print(f"\nInput: {count_tokens(text)} tokens")
    print(f"Chunked into {len(chunks)} pieces (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}):\n")
    for i, chunk in enumerate(chunks, 1):
        tokens = count_tokens(chunk)
        print(f"  Chunk {i} ({tokens} tokens):")
        print(f"    {chunk[:120]}...")
        print()


def cmd_ingest(client: OpenAI, conn, file_path: str):
    """Read a text file, chunk it, embed chunks, store in pgvector."""
    if not os.path.exists(file_path):
        raise SystemExit(f"File not found: {file_path}")

    with open(file_path, "r") as f:
        text = f.read()

    print(f"\nFile: {file_path} ({count_tokens(text)} tokens)")

    # Chunk
    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks")

    # Embed all chunks in batch
    print(f"Generating embeddings...")
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=chunks)
    sorted_data = sorted(resp.data, key=lambda x: x.index)
    embeddings = [item.embedding for item in sorted_data]

    # Insert
    filename = os.path.basename(file_path)
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        metadata = {"source": filename, "chunk_index": i, "total_chunks": len(chunks)}
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (content, metadata, embedding) VALUES (%s, %s, %s)",
                (chunk, json.dumps(metadata), np.array(emb)),
            )
    conn.commit()
    print(f"Ingested {len(chunks)} chunks from {filename}")


# ========== MAIN ==========


def main():
    parser = argparse.ArgumentParser(description="Day 5: RAG with pgvector")
    parser.add_argument(
        "--mode",
        choices=["rag", "chunk", "ingest"],
        default="rag",
        help="rag: full pipeline | chunk: demo chunking | ingest: load a file",
    )
    parser.add_argument("--query", default="how do caches improve performance", help="Question to ask")
    parser.add_argument("--top-k", type=int, default=5, help="Number of docs to retrieve")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved context before answering")
    parser.add_argument("--temperature", type=float, default=0.2, help="Model temperature")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens for answer")
    parser.add_argument("--min-similarity", type=float, default=0.3, help="Minimum similarity score for reranking")
    parser.add_argument("--text", default="", help="Text to chunk (for --mode chunk)")
    parser.add_argument("--file", default="", help="File path to ingest (for --mode ingest)")
    args = parser.parse_args()

    if args.mode == "chunk":
        if not args.text:
            # Use a sample text for demo
            sample = (
                "Caching is a technique for storing frequently accessed data in fast memory. "
                "Redis and Memcached are popular in-memory stores. Write-through caches write "
                "to both cache and database simultaneously. Write-behind caches write to cache "
                "first and asynchronously sync to the database. Cache-aside lets the application "
                "manage cache reads and writes explicitly. TTL (time-to-live) prevents stale data. "
                "Cache invalidation is one of the hardest problems in computer science. "
                "Distributed caches like Redis Cluster handle scaling across multiple nodes."
            )
            args.text = sample
        cmd_chunk(args.text)
        return

    conn = get_db_connection()
    client = get_openai_client()

    if args.mode == "ingest":
        if not args.file:
            raise SystemExit("Provide --file path for ingest mode")
        cmd_ingest(client, conn, args.file)
    else:
        run_rag(
            client, conn, args.query,
            top_k=args.top_k,
            show_context=args.show_context,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            min_similarity=args.min_similarity,
        )

    conn.close()


if __name__ == "__main__":
    main()
