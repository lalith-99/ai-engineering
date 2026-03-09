"""
Day 4: Embeddings + pgvector
=============================
Generate embeddings with OpenAI -> store in PostgreSQL (pgvector) -> similarity search.

Covers: embedding generation, batch embeddings, Matryoshka (dimension reduction),
        cosine/L2/inner product distance, HNSW indexing, similarity search.

Setup:
  1. docker compose up -d            # starts pgvector on port 5433
  2. pip3 install -r requirements.txt
  3. export OPENAI_API_KEY=sk-...

Usage:
  python3 embeddings_pgvector.py --mode seed      # generate embeddings & insert sample docs
  python3 embeddings_pgvector.py --mode search --query "how do caches work"
  python3 embeddings_pgvector.py --mode search --query "kubernetes scaling" --top-k 3
  python3 embeddings_pgvector.py --mode inspect    # show all stored documents + vector preview
  python3 embeddings_pgvector.py --mode matryoshka  # demo dimension reduction
  python3 embeddings_pgvector.py --mode distances   # compare distance functions
  python3 embeddings_pgvector.py --mode index       # create HNSW index for fast search
"""

import os
import json
import argparse
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

import numpy as np


# ========== CONFIG ==========

DB_CONFIG = {
    "host": os.getenv("PGVECTOR_HOST", "localhost"),
    "port": int(os.getenv("PGVECTOR_PORT", "5433")),
    "dbname": os.getenv("PGVECTOR_DB", "embeddings"),
    "user": os.getenv("PGVECTOR_USER", "llm"),
    "password": os.getenv("PGVECTOR_PASSWORD", "llm_pass"),
}

EMBEDDING_MODEL = "text-embedding-3-small"   # 1536 dimensions, cheap ($0.02 / 1M tokens)
EMBEDDING_DIM = 1536


# ========== SAMPLE DOCUMENTS ==========

SAMPLE_DOCS = [
    {
        "content": (
            "Caching stores frequently accessed data in fast memory like Redis or Memcached. "
            "It reduces database queries and improves latency. Common strategies include "
            "write-through, write-behind, and cache-aside patterns."
        ),
        "metadata": {"topic": "caching", "level": "intermediate"},
    },
    {
        "content": (
            "Kubernetes horizontally scales applications using pod autoscaling. The HPA "
            "monitors CPU or custom metrics and adjusts replicas. Cluster Autoscaler "
            "adds or removes nodes based on pending pod demands."
        ),
        "metadata": {"topic": "kubernetes", "level": "intermediate"},
    },
    {
        "content": (
            "Database indexing uses B-tree or hash structures to speed up queries. "
            "Indexes trade write performance for faster reads. Composite indexes "
            "cover multiple columns but order matters for query optimization."
        ),
        "metadata": {"topic": "databases", "level": "beginner"},
    },
    {
        "content": (
            "Circuit breakers prevent cascading failures in microservices. When a "
            "downstream service fails repeatedly, the breaker opens and returns a "
            "fallback response. After a cooldown, it enters half-open state to test recovery."
        ),
        "metadata": {"topic": "resilience", "level": "intermediate"},
    },
    {
        "content": (
            "Message queues like SQS or RabbitMQ decouple producers from consumers. "
            "They enable asynchronous processing, retry logic, and dead-letter queues "
            "for failed messages. This improves reliability and throughput."
        ),
        "metadata": {"topic": "messaging", "level": "beginner"},
    },
    {
        "content": (
            "API rate limiting protects services from abuse. Common algorithms include "
            "token bucket, sliding window, and fixed window counters. Rate limits are "
            "typically enforced at the API gateway or load balancer level."
        ),
        "metadata": {"topic": "api-design", "level": "beginner"},
    },
    {
        "content": (
            "Observability combines metrics, logs, and traces. Prometheus scrapes metrics, "
            "Grafana visualizes them, and Jaeger or OpenTelemetry traces requests across "
            "microservices. This helps pinpoint latency bottlenecks and failures."
        ),
        "metadata": {"topic": "observability", "level": "intermediate"},
    },
    {
        "content": (
            "Vector databases store high-dimensional embeddings for similarity search. "
            "pgvector adds vector support to PostgreSQL. HNSW and IVFFlat indexes enable "
            "fast approximate nearest neighbor queries without external dependencies."
        ),
        "metadata": {"topic": "vector-databases", "level": "advanced"},
    },
]


# ========== CLIENT HELPERS ==========


def get_openai_client() -> OpenAI:
    """Create an OpenAI client from environment config."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY. Export it, then 'source ~/.zshrc'.")
    return OpenAI(api_key=api_key)


def get_db_connection():
    """Connect to pgvector Postgres and register the vector type."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        register_vector(conn)
        return conn
    except psycopg2.OperationalError as e:
        raise SystemExit(
            f"Cannot connect to Postgres: {e}\n"
            "Make sure pgvector is running: docker compose up -d"
        )


# ========== EMBEDDING GENERATION ==========


def generate_embedding(client: OpenAI, text: str) -> List[float]:
    """Generate a single embedding vector using OpenAI."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def generate_embeddings_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts in one API call (cheaper + faster)."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    # Sort by index to guarantee order
    sorted_data = sorted(resp.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


# ========== DATABASE OPERATIONS ==========


def insert_document(conn, content: str, metadata: Dict[str, Any], embedding: List[float]) -> int:
    """Insert a document with its embedding into pgvector."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (content, metadata, embedding)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (content, json.dumps(metadata), np.array(embedding)),
        )
        doc_id = cur.fetchone()[0]
    conn.commit()
    return doc_id


def similarity_search(
    conn, query_embedding: List[float], top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Find the most similar documents using cosine distance.

    pgvector operators:
      <=>  cosine distance      (1 - cosine_similarity)
      <->  L2 (euclidean) distance
      <#>  inner product (negative)

    We use cosine distance and convert to similarity for readability.
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                id,
                content,
                metadata,
                1 - (embedding <=> %s) AS similarity,
                created_at
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (np.array(query_embedding), np.array(query_embedding), top_k),
        )
        return cur.fetchall()


def get_all_documents(conn) -> List[Dict[str, Any]]:
    """Retrieve all documents (for inspection)."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, content, metadata,
                   LEFT(embedding::text, 60) AS embedding_preview,
                   created_at
            FROM documents
            ORDER BY id
            """
        )
        return cur.fetchall()


def get_document_count(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents")
        return cur.fetchone()[0]


# ========== COMMANDS ==========


def cmd_seed(client: OpenAI, conn):
    """Seed the database with sample documents and their embeddings."""
    existing = get_document_count(conn)
    if existing > 0:
        print(f"\nDatabase already has {existing} documents. Skipping seed.")
        print("   To re-seed, run: docker compose down -v && docker compose up -d")
        return

    print(f"\nSeeding {len(SAMPLE_DOCS)} documents...")

    # Batch-embed all docs in one API call
    texts = [doc["content"] for doc in SAMPLE_DOCS]
    print(f"Generating embeddings for {len(texts)} texts (batch call)...")
    embeddings = generate_embeddings_batch(client, texts)

    for i, (doc, emb) in enumerate(zip(SAMPLE_DOCS, embeddings)):
        doc_id = insert_document(conn, doc["content"], doc["metadata"], emb)
        print(f"   [{i+1}/{len(SAMPLE_DOCS)}] id={doc_id} topic={doc['metadata']['topic']}")

    print(f"\nSeeded {len(SAMPLE_DOCS)} documents with {EMBEDDING_DIM}-dim embeddings.")
    print(f"   Model: {EMBEDDING_MODEL}")


def cmd_search(client: OpenAI, conn, query: str, top_k: int):
    """Embed the query, then find the top-k most similar documents."""
    print(f"\nQuery: \"{query}\"")
    print(f"   Generating query embedding ({EMBEDDING_MODEL})...")

    query_embedding = generate_embedding(client, query)

    print(f"   Searching top-{top_k} similar documents...\n")
    results = similarity_search(conn, query_embedding, top_k)

    if not results:
        print("   No documents found. Did you run --mode seed first?")
        return

    print(f"{'='*70}")
    for i, row in enumerate(results, 1):
        similarity = row["similarity"]
        if similarity >= 0.7:
            badge = "HIGH"
        elif similarity >= 0.5:
            badge = "MED "
        else:
            badge = "LOW "

        meta = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"])
        print(f"\n  [{badge}] #{i}  similarity={similarity:.4f}  id={row['id']}  topic={meta.get('topic', '?')}")
        print(f"     {row['content'][:120]}...")
    print(f"\n{'='*70}")


def cmd_inspect(conn):
    """Show all stored documents for debugging."""
    docs = get_all_documents(conn)
    if not docs:
        print("\nNo documents stored yet. Run --mode seed first.")
        return

    print(f"\n{len(docs)} documents in database:\n")
    for doc in docs:
        meta = doc["metadata"] if isinstance(doc["metadata"], dict) else json.loads(doc["metadata"])
        print(f"  id={doc['id']}  topic={meta.get('topic', '?'):15}  created={doc['created_at']}")
        print(f"    content: {doc['content'][:80]}...")
        print(f"    vector:  {doc['embedding_preview']}...")
        print()


# ========== MATRYOSHKA EMBEDDINGS (interview buzzword!) ==========


def cmd_matryoshka(client: OpenAI):
    """
    Demo Matryoshka Representation Learning — same model, fewer dimensions.

    text-embedding-3 models support a `dimensions` parameter that truncates
    the vector while keeping most quality. Useful for:
      - Reducing storage costs (512 dims = 3x less storage than 1536)
      - Faster search (shorter vectors = faster distance calculations)
      - Prototyping vs production (256 for dev, 1536 for prod)
    """
    text = "Caching stores frequently accessed data in fast memory like Redis."

    dimensions_to_test = [256, 512, 1024, 1536]
    embeddings = {}

    print(f"\nMatryoshka Embeddings Demo")
    print(f"Text: \"{text}\"\n")

    for dim in dimensions_to_test:
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=dim,  # <-- the Matryoshka parameter
        )
        vec = resp.data[0].embedding
        embeddings[dim] = vec
        print(f"  {dim:>5} dimensions: vector[:5] = [{', '.join(f'{v:.4f}' for v in vec[:5])}...]")

    # Show that truncated vectors preserve relative similarities
    text2 = "Redis and Memcached are popular in-memory stores for caching."
    text3 = "Kubernetes horizontally scales applications using pod autoscaling."

    print(f"\n  Similarity preservation test:")
    print(f"    Text A: \"{text[:60]}...\"")
    print(f"    Text B (similar): \"{text2[:60]}...\"")
    print(f"    Text C (different): \"{text3[:60]}...\"")
    print()

    for dim in [256, 1536]:
        resp_b = client.embeddings.create(model=EMBEDDING_MODEL, input=text2, dimensions=dim)
        resp_c = client.embeddings.create(model=EMBEDDING_MODEL, input=text3, dimensions=dim)
        vec_a = np.array(embeddings[dim])
        vec_b = np.array(resp_b.data[0].embedding)
        vec_c = np.array(resp_c.data[0].embedding)

        # Cosine similarity
        sim_ab = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        sim_ac = np.dot(vec_a, vec_c) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_c))
        print(f"    {dim} dims:  sim(A,B)={sim_ab:.4f}  sim(A,C)={sim_ac:.4f}  (B should be >> C)")

    print(f"\n  Takeaway: 256 dims preserves ranking with 6x less storage.")


# ========== DISTANCE FUNCTION COMPARISON ==========


def cmd_distances(client: OpenAI):
    """
    Compare the three pgvector distance functions.

    <=>  cosine distance   (1 - cosine_similarity) — best for text
    <->  L2 (euclidean)    — straight-line distance
    <#>  inner product     — fastest when vectors are normalized
    """
    texts = [
        "Caching stores data in fast memory like Redis.",
        "Redis and Memcached are in-memory caching stores.",   # similar to [0]
        "Kubernetes scales pods using horizontal autoscaling.", # different
    ]

    print(f"\nDistance Function Comparison\n")
    for i, t in enumerate(texts):
        print(f"  [{i}] {t}")

    # Get embeddings
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vecs = [np.array(d.embedding) for d in sorted(resp.data, key=lambda x: x.index)]

    # Compute all three distance types
    pairs = [(0, 1, "similar"), (0, 2, "different")]
    print(f"\n  {'Pair':<20} {'Cosine Sim':>12} {'Cosine Dist':>12} {'L2 Dist':>12} {'Inner Prod':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for i, j, label in pairs:
        a, b = vecs[i], vecs[j]
        cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        cosine_dist = 1 - cosine_sim
        l2_dist = np.linalg.norm(a - b)
        inner_prod = np.dot(a, b)
        print(f"  [{i}] vs [{j}] ({label:>9}) {cosine_sim:>12.4f} {cosine_dist:>12.4f} {l2_dist:>12.4f} {inner_prod:>12.4f}")

    print(f"\n  When to use which:")
    print(f"    <=> Cosine distance  — default for text (ignores magnitude)")
    print(f"    <-> L2 distance      — when magnitude matters (rare for text)")
    print(f"    <#> Inner product    — fastest, use when vectors are normalized")
    print(f"    Note: OpenAI vectors are normalized, so cosine and inner product give same ranking.")


# ========== HNSW INDEX ==========


def cmd_create_index(conn):
    """
    Create an HNSW index for fast approximate nearest neighbor search.

    HNSW (Hierarchical Navigable Small World):
      - Default choice for pgvector
      - Slower to build, much faster to query
      - ~99% recall (finds 99 of the true 100 nearest)

    IVFFlat alternative:
      - Faster to build, slower to query
      - Only use when memory is very constrained
    """
    print(f"\nCreating HNSW index on documents.embedding...")
    with conn.cursor() as cur:
        # Check if index already exists
        cur.execute("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'documents' AND indexname = 'documents_embedding_hnsw_idx'
        """)
        if cur.fetchone():
            print("   HNSW index already exists. Skipping.")
            return

        # Create HNSW index for cosine distance
        cur.execute("""
            CREATE INDEX documents_embedding_hnsw_idx
            ON documents
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)
        # m = max connections per node (default 16, higher = more accurate, more memory)
        # ef_construction = build-time search width (higher = slower build, better recall)
    conn.commit()
    print("   HNSW index created successfully.")
    print("   Parameters: m=16, ef_construction=64 (cosine distance)")
    print("   Effect: vector searches are now ~100x faster at >99% recall.")


# ========== MAIN ==========


def main():
    parser = argparse.ArgumentParser(description="Day 4: Embeddings + pgvector")
    parser.add_argument(
        "--mode",
        choices=["seed", "search", "inspect", "matryoshka", "distances", "index"],
        default="search",
        help="seed | search | inspect | matryoshka (dim reduction) | distances (compare funcs) | index (create HNSW)",
    )
    parser.add_argument("--query", default="how do caches improve performance", help="Search query text")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()

    # Modes that only need OpenAI (no DB)
    if args.mode in ("matryoshka", "distances"):
        client = get_openai_client()
        if args.mode == "matryoshka":
            cmd_matryoshka(client)
        else:
            cmd_distances(client)
        return

    # All other modes need DB
    conn = get_db_connection()

    if args.mode == "seed":
        client = get_openai_client()
        cmd_seed(client, conn)
    elif args.mode == "search":
        client = get_openai_client()
        cmd_search(client, conn, args.query, args.top_k)
    elif args.mode == "inspect":
        cmd_inspect(conn)
    elif args.mode == "index":
        cmd_create_index(conn)

    conn.close()


if __name__ == "__main__":
    main()
