# Days 4–6: Embeddings, RAG & LLM Observability

> Interview prep guide for SDE2/AI roles — Feb 2026

---

## Table of Contents
1. [Day 4 — Embeddings & Vector Storage](#day-4--embeddings--vector-storage)
2. [Day 5 — RAG (Retrieval-Augmented Generation)](#day-5--rag-retrieval-augmented-generation)
3. [Day 6 — LLM Observability & Cost Control](#day-6--llm-observability--cost-control)
4. [Interview Cheat Sheet — Days 4–6 Buzzwords](#-interview-cheat-sheet--days-46-buzzwords)
5. [System Design Patterns for RAG Interviews](#-system-design-patterns-for-rag-interviews)
6. [Glossary](#glossary)

---

# Day 4 — Embeddings & Vector Storage

## 1. What Are Embeddings? (Plain English)

Imagine every sentence you write has a **hidden address** — a point in a giant map of meaning. Sentences about "dogs playing fetch" and "puppies catching a ball" end up very close together, while "quantum physics equations" is far away.

An **embedding** is that address — a list of numbers (a vector) that captures what text *means*.

```
"How do caches work?"  →  [0.012, -0.034, 0.056, ... 1536 numbers]
"Explain caching"      →  [0.011, -0.033, 0.057, ... 1536 numbers]  ← very similar!
"Best pizza in NYC"    →  [0.891, 0.234, -0.567, ... 1536 numbers]  ← very different!
```

**Distance between vectors = semantic similarity.** That's the whole idea.

### Interview Buzzword: "Embeddings / Semantic Similarity"
Embeddings convert text to vectors where semantic similarity = vector proximity. Use cosine similarity for semantic search and RAG.

### Resume Keyword
"Vector embeddings for semantic search & retrieval"

---

## 2. Embedding Models — The Current Landscape (Feb 2026)

### OpenAI Models
| Model | Dimensions | Max Tokens | Price (per 1M tokens) | Best For |
|-------|-----------|------------|----------------------|---------|
| `text-embedding-3-small` | 1536 (configurable) | 8192 | $0.02 | Budget-friendly, most projects ✅ |
| `text-embedding-3-large` | 3072 (configurable) | 8192 | $0.13 | Higher accuracy when needed |
| `text-embedding-ada-002` | 1536 | 8192 | $0.10 | Legacy, avoid for new projects |

**Key feature of v3 models:** **Matryoshka Representation Learning** — you can *reduce* dimensions with the `dimensions` parameter (e.g., 512 instead of 1536) to save storage while keeping most quality.
Interview Buzzword: "Matryoshka Embeddings"
Matryoshka embeddings let you reduce dimensions (256 instead of 1536) while keeping quality. Same model, less storage.
Matryoshka embeddings allow variable-dimension outputs from the same model. I can use 256 dimensions for prototyping (75% storage savings) and scale to full 1536 for production — same model, just truncate the vector.

### Google Vertex AI
| Model | Dimensions | Notes |
|-------|-----------|-------|
| `gemini-embedding-001` | up to 3072 | leading, 100+ languages |
| `text-embedding-005` | up to 768 | English & code specialized |
| `text-multilingual-embedding-002` | up to 768 | 100+ languages |

**Unique feature:** `task_type` parameter — tell the model HOW you'll use the embedding (retrieval, classification, clustering) and it optimizes accordingly.

### Other Notable Providers
| Provider | Model | Why Notable | Notes |
|----------|-------|-------------|----------------------|
| **Cohere** | `embed-v4` | Excellent multilingual + built-in compression | "Best for multilingual RAG" |
| **Voyage AI** (Anthropic) | `voyage-3` | Tops retrieval benchmarks | "Acquired by Anthropic — integrated into Claude ecosystem" |
| **Jina AI** | `jina-embeddings-v3` | Open-source, 8K context | "Great open alternative" |
| **BGE (BAAI)** | `bge-m3` | Best free/open-source, multilingual | "Zero cost, competitive quality" |
| **Nomic** | `nomic-embed-text-v2-moe` | MoE architecture, efficient | "MoE approach to embeddings — novel" |

### When to Use Which? (Interview Answer)
For most projects, I start with `text-embedding-3-small` ($0.02/M) — it's cheap and good enough for 90% of use cases. For multilingual, I'd use Cohere `embed-v4`. For budget-zero, BGE `bge-m3` is the best open-source option. For maximum quality, Voyage AI or `text-embedding-3-large`.

---

## 3. Vector Databases — Where Embeddings Live

### The Landscape (Feb 2026)

```
                    ┌──────────────────────────────────────────┐
                    │          Vector Storage Options            │
                    ├────────────┬─────────────┬────────────────┤
                    │  Dedicated │  DB + Vector│   In-Memory    │
                    │  Vector DBs│  Extensions │   Libraries    │
                    ├────────────┼─────────────┼────────────────┤
                    │ Pinecone   │ pgvector    │ FAISS          │
                    │ Weaviate   │ (Postgres)  │ Annoy          │
                    │ Qdrant     │             │ HNSWLib        │
                    │ Chroma     │ MongoDB     │                │
                    │ Milvus     │ Atlas Search│                │
                    └────────────┴─────────────┴────────────────┘
```

### Comparison Table

| Solution | Type | GitHub Stars | Free Tier | Best For | Notes |
|----------|---| Postgres extension | 19.9K ⭐ | Yes (self-host) | You already know Postgres | Boring is good — one less thing to learn
| **pgvector** | Postgres extension | 19.9K ⭐ | Yes (self-host) | You already know Postgres | "Boring technology — and that's a compliment" |
| **Pinecone** | Managed cloud | N/A | Yes (limited) | Zero-ops, auto-scale | "Serverless vector DB, no infra management" |
| **Chroma** | Embedded | 18K+ ⭐ | Yes (OSS) | Quick prototyping | "SQLite of vector databases" |
| **Weaviate** | Dedicated | 14K+ ⭐ | Yes | Multi-modal (text + images) | "Native multi-modal search" |
| **Qdrant** | Dedicated | 24K+ ⭐ | Yes | Rust-based, very fast | "Highest performance per dollar" |
| **Milvus** | Dedicated | 34K+ ⭐ | Yes | Billions of vectors | "Billion-scale vector search" |
| **FAISS** | Library | 33K+ ⭐ | Yes (OSS) | Research, in-process | "Meta's research library, not a DB" |
Why pgvector?
Boring tech choice — vector search as a Postgres extension. Keep your SQL, ACID, joins, backups, replication. No new infra. Good for < 10M vectors.
pgvector is the 'boring technology' choice — I get vector search AS a Postgres extension, so I keep all the benefits: SQL, ACID, JOINs, backups, replication. No new infrastructure to manage. For 90% of projects with < 10M vectors, pgvector is the right answer.

### When to NOT Use pgvector
- **Billions** of vectors → Milvus or Pinecone
- **Sub-millisecond** latency at massive scale → Qdrant
- Built-in **multi-modal** (image + text) search → Weaviate

### Resume Keyword: "Vector database design (pgvector, Pinecone, FAISS)"

---

## 4. pgvector Key Concepts (Your Code!)

### Distance Operators
```sql
-- The three distance operators you need to know:
<==>   -- Cosine distance    (most common for text, 0-2, lower = more similar)
<->    -- L2/Euclidean       (geometric distance)
<#>    -- Inner product      (for normalized vectors, fastest)

-- Create a table with vectors (from your embeddings_pgvector.py)
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding vector(1536)    -- 1536 dims for OpenAI small
);

-- Similarity search
SELECT content, embedding <=> query_vector AS distance
FROM documents
ORDER BY embedding <=> query_vector
LIMIT 5;
```

### HNSW vs IVFFlat — Index Types

| | **HNSW** ✅ | **IVFFlat** |
|---|------|---------|
| **How it works** | Multi-layer graph (skip list for vectors) | Clusters vectors, searches nearby buckets |
| **Build speed** | Slower | Faster |
| **Query speed** | **Faster** ✅ | Slower |
| **Memory** | More | Less |
| **Recall** | **Better** ✅ | Good |
| **Recommendation** | **Default choice** | Memory-constrained only |

### Interview Tip
I use HNSW indexing by default — it's slower to build but much faster to query. IVFFlat only makes sense when memory is very constrained. For OpenAI embeddings, cosine distance and inner product give identical rankings because vectors are normalized.

---

## 5. Distance Functions — Which One?

```
Cosine Distance  ← USE THIS for text (most common)
   Measures: angle between vectors
   Range: 0 (identical) to 2 (opposite)
   Why: works regardless of magnitude

Euclidean (L2) Distance
   Measures: straight-line distance
   Range: 0 to ∞
   Why: when magnitude matters (rare for text)

Inner Product (Dot Product)
   Measures: projection of one vector onto another
   Range: -∞ to ∞
   Why: fastest when vectors are normalized (OpenAI vectors ARE)
```

### Interview Buzzword: "Cosine Similarity / Approximate Nearest Neighbor (ANN)"
I use cosine similarity for text embeddings. For production scale, I apply HNSW-based ANN indexing — it's not exact, but gives ~99% recall at 100x speed.

---

# Day 5 — RAG (Retrieval-Augmented Generation)

## 1. The Problem RAG Solves

```
Without RAG:
  User: "What's our company's refund policy?"
  LLM:  "Generally, companies offer 30-day refund..." ← HALLUCINATED/GENERIC

With RAG:
  User: "What's our company's refund policy?"
  System: [searches your docs, finds policy document]
  LLM:  "Per your policy (v2.3), refunds are available
         within 14 days for unused items..." ← GROUNDED IN YOUR DATA
```

### Interview Buzzword: "RAG (Retrieval-Augmented Generation)"
RAG grounds LLM responses in external data by retrieving relevant documents before generation. It eliminates hallucination, requires no retraining, and updates instantly when docs change.

### Resume Keyword: "RAG pipeline design with vector search & LLM integration"

### Why RAG > Fine-Tuning (Interview Answer!)

| Approach | Cost | Update Speed | Accuracy | When to Use |
|----------|------|-------------|----------|-------------|
| **RAG** ✅ | Low ($) | Instant | High (cites sources) | Add knowledge to LLM |
| **Fine-tuning** | High ($$$) | Slow (retrain) | Medium (no citations) | Change model's behavior/style |
| **Context stuffing** | Medium ($$) | Instant | High but limited | Small, fixed context |

RAG is the default for 90% of knowledge-augmentation use cases. Fine-tuning is for changing the model's *behavior/style*, not its *knowledge*.

---

## 2. The RAG Pipeline — Step by Step

```
┌─────────────────────────────────────────────────────────┐
│                    OFFLINE (Ingestion)                    │
│                                                          │
│  Raw Documents → Clean → Chunk → Embed → Store in DB    │
│  (PDFs, docs,    (strip   (split   (OpenAI   (pgvector) │
│   wikis, etc.)   noise)   text)    API)                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    ONLINE (Query Time)                    │
│                                                          │
│  User Question                                           │
│       ↓                                                  │
│  Embed the question (same model as ingestion!)           │
│       ↓                                                  │
│  Vector search → top-k similar chunks from DB            │
│       ↓                                                  │
│  (Optional) Rerank results for better relevance          │
│       ↓                                                  │
│  Build prompt: "Answer using ONLY this context: ..."     │
│       ↓                                                  │
│  LLM generates grounded answer with citations            │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Chunking — The Most Underrated Part

Chunking = how you split documents into pieces before embedding. **This is where most RAG pipelines fail or succeed.**

### Why It Matters
```
Too big  (10,000 tokens) → embedding is too vague, retrieval is poor
Too small (50 tokens)    → loses context, fragments don't make sense
Sweet spot:               200–500 tokens per chunk, 50–100 token overlap
```

### Chunking Strategies

| Strategy | How | When to Use | Notes |
|----------|-----|-------------|----------------------|
| **Fixed-size** | Split every N tokens | Simple default | "Good baseline but ignores semantics" |
| **Sentence-based** | Split on sentence boundaries | Better for QA | "Preserves sentence-level meaning" |
| **Paragraph-based** | Split on `\n\n` | Well-structured docs | "Natural document boundaries" |
| **Recursive** | paragraph → sentence → word | Versatile | "LangChain default — tries biggest first" |
| **Semantic** | Embedding-based break points | Best quality | "Detects topic shifts — highest quality" |
| **Document-specific** | Markdown headers, HTML tags | Structured docs | "Schema-aware chunking" |

### Interview Buzzword: "Semantic Chunking"
Semantic chunking uses embeddings to detect natural topic boundaries in text, producing chunks that are semantically coherent. It outperforms fixed-size chunking but is more expensive.

### The Overlap Trick (From Your Code!)
```
Chunk 1: "...Redis stores data in memory for fast access. It supports
          multiple data structures including strings, lists, and sets."

Chunk 2: "...including strings, lists, and sets. Redis also provides
          persistence options like RDB snapshots and AOF logging..."
          ↑___ 50-token overlap preserves context across boundaries
```

Your `rag.py` uses `CHUNK_SIZE = 300` and `CHUNK_OVERLAP = 50` — a solid default.

---

## 4. Retrieval Strategies — Beyond Basic Vector Search

### Level 1: Semantic Search Only (Your Day 5 Code)
```python
# What you're doing in rag.py
results = vector_search(query_embedding, top_k=5)
```

### Level 2: Hybrid Search (Vector + Keyword)
```
Combine:
  - Vector search: finds semantically similar ("automobile" → "car")
  - Keyword search (BM25): finds exact matches ("error code XJ-4521")

Why both? Vector misses specific terms; keywords miss meaning.
```

### Interview Buzzword: "Hybrid Search"
I combine vector similarity search with BM25 keyword search. Vector captures semantic meaning ('automobile' matches 'car'), while BM25 catches exact terms like error codes or product IDs that embeddings might miss.

### Level 3: Reranking
```
Step 1: Retrieve 20 candidates (fast, broad search)
Step 2: Rerank with a cross-encoder model (slow, accurate scoring)
Step 3: Return top 5

Popular rerankers:
  - Cohere Rerank ($1/1K searches)
  - BGE Reranker v2 (open-source, free)
  - Jina Reranker v2 (open-source)
```

### Interview Buzzword: "Retrieval Reranking / Cross-Encoder"
I use a two-stage retrieval pipeline: a fast bi-encoder for candidate generation, then a cross-encoder reranker for precise scoring. This gives me the speed of vector search with the accuracy of full pair-wise comparison.

### Cutting Edge Techniques (2025–2026)

| Technique | What It Does | Maturity | Notes |
|-----------|-------------|----------|----------------------|
| **Agentic RAG** | LLM decides what/when to retrieve, iterates | Production ✅ | "Agent-driven retrieval for complex questions" |
| **GraphRAG** | Knowledge graphs from docs, retrieve subgraphs | Experimental | "Microsoft's approach — builds entity graphs" |
| **RAPTOR** | Hierarchical doc summaries at multiple levels | Research | "Multi-level abstraction tree" |
| **Corrective RAG (CRAG)** | LLM evaluates retrieval, re-searches if poor | Emerging | "Self-correcting retrieval loop" |
| **Self-RAG** | Model decides when to retrieve, self-critiques | Research | "Retrieval-aware generation" |
| **Contextual Retrieval** | Adds context to chunks before embedding | Production ✅ | "Anthropic's approach — prepend context per chunk" |
| **Late Chunking** | Embeds full doc, then chunks the embeddings | Emerging | "Better global context preservation" |

### Resume Keyword: "Advanced RAG — hybrid search, reranking, agentic retrieval"

---

## 5. RAG Evaluation — How Do You Know It's Working?

### Key Metrics (Interview Must-Know!)

| Metric | What It Measures | Score | Interview Answer |
|--------|-----------------|-------|-----------------|
| **Faithfulness** | Does the answer stick to context? (no hallucination) | 0–1 | "Tests if LLM made stuff up" |
| **Answer Relevancy** | Does the answer address the question? | 0–1 | "Tests if LLM answered the right thing" |
| **Context Precision** | Are retrieved chunks relevant? | 0–1 | "Tests retrieval quality" |
| **Context Recall** | Did we retrieve ALL needed info? | 0–1 | "Tests retrieval completeness" |

### Interview Answer: "How do you evaluate a RAG pipeline?"
I measure four things: (1) Faithfulness — is the answer grounded in retrieved context? (2) Answer relevancy — does it address the question? (3) Context precision — are the retrieved chunks actually relevant? (4) Context recall — did we miss any needed information? I use RAGAS or DeepEval for automated evaluation in CI/CD.

### Tools for RAG Evaluation

| Tool | Type | Stars | Best For |
|------|------|-------|---------|
| **RAGAS** | Open-source | 8K+ ⭐ | Industry standard for RAG eval |
| **DeepEval** | Open-source | 13.7K ⭐ | More metrics, nice UI, pytest integration |
| **Langfuse Evals** | Platform | 22K ⭐ | Integrated with observability |
| **Arize Phoenix** | Open-source | 12K+ ⭐ | Embedding visualization |
| **Braintrust** | Commercial | N/A | End-to-end eval platform |

### Resume Keyword: "RAG evaluation pipelines (RAGAS, DeepEval, LLM-as-judge)"

---

## 6. RAG Frameworks & Tools

| Tool | Learning Curve | Best For | Notes |
|------|---------------|---------|----------------------|
| **Raw OpenAI + pgvector** ✅ | Low | Understanding fundamentals (your Day 5!) | "I built RAG from scratch to understand every step" |
| **LangChain** | Medium | Most popular, huge ecosystem | "De facto standard, massive community" |
| **LlamaIndex** | Medium | Purpose-built for RAG, great for docs | "Best document ingestion and indexing" |
| **Haystack** | Medium | Production-grade, by deepset | "Enterprise-ready, modular pipeline" |
| **Vercel AI SDK** | Low | Web apps, streaming | "Great for Next.js frontend integration" |

### When to Use Which? (Interview Answer)
I start raw (like my Day 5 code) to understand every step. For production, LangChain for general-purpose, LlamaIndex for document-heavy RAG, and Haystack for enterprise compliance. The framework doesn't matter as much as understanding the underlying pipeline.

---

# Day 6 — LLM Observability & Cost Control

## 1. Why Observability Is Critical

```
Real horror stories (actual industry incidents):
  A recursive agent loop burned $2,400 in 3 hours
  A chatbot using GPT-4 instead of GPT-4o-mini cost 66x more
  A prompt that grew with conversation history hit $0.50/request
  A bug caused 10,000 retry loops against the API in 1 minute
```

### Interview Buzzword: "LLMOps / LLM Observability"
LLMOps is DevOps for AI — tracing every LLM call for tokens, cost, latency, and quality. I use Langfuse for open-source observability and implement budget guards to prevent cost runaway.

### Resume Keyword: "LLMOps — observability, cost control, production monitoring"

---

## 2. Current OpenAI Pricing (Feb 2026) — Updated!

### Latest Models (Complete Reference)

| Model | Input ($/1M) | Output ($/1M) | Best For |
|-------|-------------|---------------|---------|
| **GPT-5.2** | $1.75 | $14.00 | Frontier, complex reasoning |
| **GPT-5.2 pro** | $21.00 | $168.00 | Maximum intelligence |
| **GPT-5.1** | $1.25 | $10.00 | Strong general purpose |
| **GPT-5** | $1.25 | $10.00 | Flagship |
| **GPT-5 mini** | $0.25 | $2.00 | Cost-effective smart |
| **GPT-5 nano** | $0.05 | $0.40 | Ultra-cheap, simple |
| **GPT-4.1** | $2.00 | $8.00 | Great for coding |
| **GPT-4.1 mini** | $0.40 | $1.60 | Budget workhorse |
| **GPT-4.1 nano** | $0.10 | $0.40 | High-volume, simple |
| **o3** | $2.00 | $8.00 | Deep reasoning |
| **o3-pro** | $20.00 | $80.00 | Maximum reasoning |
| **o4-mini** | $1.10 | $4.40 | Budget reasoning |
| **gpt-4o-mini** *(legacy)* | $0.15 | $0.60 | Legacy, still popular |
| **text-embedding-3-small** | $0.02 | — | Embeddings |

### New Pricing Tiers (2025–2026)

| Tier | Latency | Price | Use Case |
|------|---------|-------|---------|
| **Flex** | High (async) | **75% cheaper** | Batch jobs, non-urgent |
| **Standard** (default) | Normal | Full price | Real-time apps |
| **Priority** | Lowest | Full price | Latency-critical |
| **Batch API** | Hours | **50% cheaper** | Bulk processing |

### Interview Tip: "Flex tier is new and gives 75% off for async requests. For batch processing like embeddings or evals, I'd use Flex + Batch API for maximum savings."

### Cost Comparison — Real Example

Answering 10,000 customer questions (avg 500 input + 200 output tokens each):

| Model | Total Cost | Relative | When to Use |
|-------|-----------|----------|-------------|
| GPT-5.2 | $36.75 | 735x | Only if nano/mini fail |
| GPT-5 mini | $5.25 | 105x | Balanced quality/cost |
| GPT-4.1 mini | $5.20 | 104x | Coding tasks |
| GPT-4.1 nano | $1.30 | 26x | Classification, extraction |
| **GPT-5 nano** | **$0.05** | **1x** | **Start here!** |

### Interview Answer: "How do you optimize LLM costs?"
Model selection is the #1 lever — GPT-5 nano is 735x cheaper than GPT-5.2. I start with the cheapest model that works and only upgrade when quality demands it. Beyond model selection: (1) prompt caching for repeated prefixes, (2) Flex/Batch API for non-urgent work, (3) max_tokens guard to prevent runaway generation, (4) semantic caching for repeated queries.

---

## 3. Observability Platforms — The Market (Feb 2026)

### The Big Players

| Platform | Open Source | Stars | Pricing | Key Strength | Notes |
|----------|-----------|-------|---------|-------------|----------------------|
| **Langfuse** | ✅ Yes | 22K+ ⭐ | Free → $29+/mo | Traces, evals, prompt mgmt | "Acquired by ClickHouse, market leader" |
| **LangSmith** | ❌ No | N/A | Free → $39+/mo | Tight LangChain integration | "Best if you're deep in LangChain" |
| **Helicone** | ✅ Yes | 5K+ ⭐ | Free → paid | Simplest setup (proxy) | "One-line integration, change base_url" |
| **Arize Phoenix** | ✅ Yes | 12K+ ⭐ | Free (self-host) | Embedding viz, spans | "Best for debugging embeddings" |
| **Braintrust** | ❌ No | N/A | Free → paid | Evals + logging combined | "Combined eval & observability" |
| **Weights & Biases** | ❌ No | 9K+ ⭐ | Free → paid | ML experiment tracking | "Standard for ML, expanding to LLM" |
| **OpenLLMetry** | ✅ Yes | 3K+ ⭐ | Free | OpenTelemetry-based | "Standards-based, vendor-neutral" |

### What Gets Tracked

```
Every LLM call captures:
  ├── Input (prompt, messages, system prompt)
  ├── Output (completion, tool calls)
  ├── Tokens (prompt_tokens, completion_tokens, total)
  ├── Cost ($X.XX calculated from model pricing)
  ├── Latency (TTFB, total time, p50/p95/p99)
  ├── Model (which model, which version)
  ├── Status (success, error, rate_limited)
  ├── Metadata (user_id, session_id, tags)
  └── Trace (parent-child relationships for chains/agents)
```

### Langfuse — Market Leader (3 Lines to Integrate!)

```python
# Drop-in replacement — zero code changes!
from langfuse.openai import openai  # ← this replaces `from openai import OpenAI`

# All calls automatically traced
response = openai.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
# Now check Langfuse dashboard: tokens, cost, latency, full trace ✅
```

### Interview Tip: "Langfuse was acquired by ClickHouse in 2026 — it's the most popular open-source LLM observability platform, used by Khan Academy, Canva, Twilio. I can self-host it or use the cloud version."

### Helicone — Simplest Option (Change One Line!)

```python
client = OpenAI(
    base_url="https://oai.helicone.ai/v1",  # ← only change
    default_headers={"Helicone-Auth": "Bearer sk-..."}
)
# All calls automatically logged — zero code changes ✅
```

### Resume Keyword: "LLM observability (Langfuse, Helicone, OpenTelemetry)"

---

## 4. The Four Pillars of LLM Cost Control

### Pillar 1: Model Selection (Biggest Impact — 700x difference!)
```
Decision tree:
  Simple extraction/classification? → GPT-5 nano ($0.05/M)
  General chat, summarization?      → GPT-5 mini ($0.25/M)
  Complex reasoning, coding?        → GPT-4.1 or GPT-5 ($1.25-2/M)
  Mission-critical accuracy?        → GPT-5.2 ($1.75/M)
  Last resort?                      → GPT-5.2 pro ($21/M)
```

### Pillar 2: Token Management (Your Day 6 Code!)
```python
# From your observability.py — pre-flight estimation
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("your text here")
estimated_cost = len(tokens) / 1_000_000 * price_per_million

# Guard: cap max_tokens to prevent runaway generation
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=messages,
    max_tokens=500  # ← prevents $$$$ output
)

# Track actual usage
print(f"Input: {response.usage.prompt_tokens} tokens")
print(f"Output: {response.usage.completion_tokens} tokens")
```

### Pillar 3: Caching
```
Prompt Caching (OpenAI built-in):
  - Repeated system prompts auto-cached → 90% cheaper on input
  - No code changes needed!

Semantic Caching (DIY):
  - If a similar question was asked → return cached answer
  - Tools: GPTCache, Redis + embeddings
  - Impact: 50-90% cost reduction for repetitive queries
```

### Pillar 4: Budget Enforcement (Your Code!)
```python
# From your observability.py — budget tracking
class BudgetTracker:
    def __init__(self, daily_budget_usd=5.0):
        self.daily_budget = daily_budget_usd
        self.spent_today = 0.0

    def can_spend(self, estimated_cost):
        return (self.spent_today + estimated_cost) <= self.daily_budget

    def record(self, actual_cost):
        self.spent_today += actual_cost
        if self.spent_today > self.daily_budget * 0.8:
            alert("⚠️ 80% of daily budget consumed!")
```

### Interview Answer: "How would you prevent LLM cost overruns?"
Four layers: (1) Model routing — cheapest model that works, (2) max_tokens guard on every call, (3) prompt caching for repeated prefixes (90% input savings), (4) budget enforcement with alerts at 80% threshold. I'd also use Flex/Batch API for non-urgent workloads at 75% discount.

---

## 5. Retry & Error Handling (Production Patterns)

### Exponential Backoff (From Your Code!)

```
Attempt 1: wait 1s   → retry
Attempt 2: wait 2s   → retry
Attempt 3: wait 4s   → retry
Attempt 4: wait 8s   → give up

Rate limit (429): double wait times + jitter
Server error (500): standard backoff
Auth error (401):   DON'T retry (it won't help)
```

### Circuit Breaker Pattern

```
CLOSED (normal)  →  5 errors in 60s  →  OPEN (reject all for 30s)
                                              ↓
                                        HALF-OPEN (try 1 call)
                                              ↓
                                   success? → CLOSED ✅
                                   failure? → OPEN 🔴
```

### Interview Buzzword: "Circuit Breaker Pattern"
I implement circuit breakers for LLM APIs — if we see 5 consecutive failures, we open the circuit for 30 seconds to prevent cascading failures and allow the API to recover. After 30s, we try one call (half-open) to test if recovery happened.

---

## 6. Production Monitoring Checklist

```
□ Token usage per call (prompt + completion)
□ Cost per call (calculated from model pricing)
□ Latency (p50, p95, p99)
□ Error rate (% of failed calls)
□ Rate limit hits (429 responses)
□ Daily/weekly/monthly spend with alerts
□ Per-user or per-feature cost breakdown
□ Model version tracking
□ Prompt version tracking (Langfuse)
□ Quality metrics (faithfulness, relevancy) — nightly evals
□ Alert on: cost spike > 2x, error rate > 5%, p99 > 10s
```

### Resume Keyword: "Production LLM monitoring — cost tracking, latency SLAs, quality metrics"

---

# Interview Cheat Sheet — Days 4–6 Buzzwords

## Resume Section: "RAG & Observability Skills"

### For SDE2 / AI Engineer
```
Embeddings: OpenAI text-embedding-3, Voyage AI, Cohere embed-v4,
            Matryoshka Representation Learning, Cosine Similarity

Vector Databases: pgvector (HNSW indexing), Pinecone, Chroma,
                  Qdrant, FAISS, Hybrid Search (vector + BM25)

RAG: Retrieval-Augmented Generation, Chunking Strategies,
     Semantic Chunking, Reranking (Cross-Encoder), Agentic RAG,
     GraphRAG, Contextual Retrieval

Evaluation: RAGAS, DeepEval, LLM-as-Judge, Faithfulness Metrics,
            Automated RAG Quality Testing in CI/CD

Observability: Langfuse, Helicone, OpenTelemetry, Token Budget
               Enforcement, Cost Optimization, Circuit Breaker
```

---

## Top 20 Interview Buzzwords — Days 4–6

### Tier 1 — Asked in Every RAG Interview
| # | Buzzword | One-Line Definition |
|---|----------|-------------------|
| 1 | **RAG** | Retrieval-Augmented Generation — search docs, feed to LLM |
| 2 | **Embeddings** | Convert text to vectors for semantic similarity |
| 3 | **Vector Database** | DB optimized for storing/searching embeddings |
| 4 | **Chunking** | Splitting documents into pieces for embedding |
| 5 | **Hallucination** | LLM confidently generates false information |
| 6 | **Cosine Similarity** | Measures angular distance between vectors (0–1) |
| 7 | **Context Window** | Max tokens a model can process at once |
| 8 | **Observability / LLMOps** | Monitoring, tracing, debugging AI in production |

### Tier 2 — Asked in 50%+ of Interviews
| # | Buzzword | One-Line Definition |
|---|----------|-------------------|
| 9 | **Hybrid Search** | Combining vector search + keyword search (BM25) |
| 10 | **Reranking** | Second-pass scoring of retrieved docs for accuracy |
| 11 | **HNSW Index** | Graph-based approximate nearest neighbor algorithm |
| 12 | **Faithfulness** | Does the answer stick to the retrieved context? |
| 13 | **Semantic Chunking** | Splitting by meaning, not fixed size |
| 14 | **pgvector** | Postgres extension for vector search |
| 15 | **Prompt Caching** | Cache repeated prompt prefixes for cost savings |

### Tier 3 — Differentiators
| # | Buzzword | One-Line Definition |
|---|----------|-------------------|
| 16 | **Agentic RAG** | Agent decides when/what/where to retrieve |
| 17 | **GraphRAG** | RAG using knowledge graphs instead of just vectors |
| 18 | **Matryoshka Embeddings** | Variable-dimension outputs from one model |
| 19 | **Cross-Encoder** | Pair-wise model for precise relevance scoring |
| 20 | **Circuit Breaker** | Fail-fast pattern for external API resilience |

---

# System Design Patterns for RAG Interviews

## Common Interview Question: "Design a RAG-powered knowledge base"

```
┌──────────────────────────────────────────────────────────────────┐
│                    RAG KNOWLEDGE BASE                              │
│                                                                   │
│  INGESTION PIPELINE (offline)                                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  S3 / GDrive / Confluence → Document Loader                │  │
│  │       ↓                                                     │  │
│  │  Clean & Parse (unstructured, markdownify)                  │  │
│  │       ↓                                                     │  │
│  │  Chunk (recursive, 300 tokens, 50 overlap)                  │  │
│  │       ↓                                                     │  │
│  │  Embed (text-embedding-3-small, batch API, Flex tier)       │  │
│  │       ↓                                                     │  │
│  │  Store (pgvector + HNSW index + metadata)                   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  QUERY PIPELINE (online)                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  User Query → Embed → Hybrid Search (vector + BM25)        │  │
│  │       ↓                                                     │  │
│  │  Retrieve top-20 → Rerank (Cohere) → top-5                 │  │
│  │       ↓                                                     │  │
│  │  Build prompt: system + context + question                  │  │
│  │       ↓                                                     │  │
│  │  Model Router: simple → GPT-5 nano, complex → GPT-4.1      │  │
│  │       ↓                                                     │  │
│  │  Generate answer with source citations                      │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  OBSERVABILITY (always-on)                                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Langfuse: trace every call (tokens, cost, latency)        │  │
│  │  Budget: max $5/day, alert at 80%                           │  │
│  │  Quality: nightly DeepEval runs (faithfulness > 0.85)       │  │
│  │  Alerts: cost spike > 2x, error rate > 5%, p99 > 10s       │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Key Talking Points:
1. **Chunking strategy**: Recursive at 300 tokens with 50 overlap — preserves context boundaries
2. **Hybrid search**: Vector for semantic, BM25 for exact matches (product codes, error IDs)
3. **Reranking**: Two-stage retrieval — fast recall, then precise ranking
4. **Model routing**: Nano for simple lookups, GPT-4.1 for complex synthesis
5. **Cost control**: Prompt caching, Flex tier for embeddings, budget guards
6. **Quality assurance**: Nightly evals with DeepEval, faithfulness > 0.85 SLA
7. **Observability**: Langfuse traces on every call, Grafana dashboards for SLAs

---

## Common Interview Question: "How would you improve a RAG pipeline that's returning bad answers?"

```
Diagnostic Framework:

1. CHECK RETRIEVAL QUALITY
   └── Are the right chunks being retrieved?
       ├── No → Fix chunking (too big/small), add hybrid search
       └── Yes but wrong order → Add reranking

2. CHECK CONTEXT QUALITY
   └── Is the retrieved context sufficient?
       ├── No → Increase top-k, reduce chunk size
       └── Yes but noisy → Add metadata filtering, improve chunking

3. CHECK GENERATION QUALITY
   └── Is the LLM using the context correctly?
       ├── Hallucinating → Strengthen "ONLY use this context" instruction
       ├── Ignoring context → Restructure prompt, add few-shot examples
       └── Wrong format → Add output constraints, JSON mode

4. CHECK EVALUATION
   └── Do you even know what "bad" means?
       ├── No metrics → Add RAGAS/DeepEval pipeline
       └── Has metrics → Look at which metric is low, fix that stage
```

---

# Glossary

| Term | Plain English Definition |
|------|------------------------|
| **ANN** | Approximate Nearest Neighbor — finding "close enough" vectors fast |
| **Bi-encoder** | Model that independently embeds query and documents (fast) |
| **BM25** | Classic keyword-matching algorithm (like old-school search) |
| **Chunking** | Splitting documents into pieces for embedding |
| **Circuit breaker** | Fail-fast pattern for external service failures |
| **Cosine similarity** | How similar two vectors are (1 = identical, 0 = unrelated) |
| **Cross-encoder** | Model that scores query-document pairs jointly (accurate) |
| **Dimensions** | How many numbers in a vector (1536 for OpenAI small) |
| **Embedding** | A list of numbers representing the meaning of text |
| **Exponential backoff** | Doubling wait time between retries |
| **Faithfulness** | Whether an answer sticks to the given context |
| **Grounding** | Giving the LLM factual context to prevent hallucination |
| **Hallucination** | LLM confidently generating false information |
| **HNSW** | Graph-based algorithm for fast approximate vector search |
| **Hybrid search** | Combining vector search with keyword search |
| **IVFFlat** | Clustering-based algorithm for approximate vector search |
| **Latency** | Time from request to response (measured in ms) |
| **LLMOps** | DevOps for LLM applications — observability, monitoring |
| **Matryoshka** | Variable-dimension embedding technique |
| **Prompt caching** | Caching repeated prompt prefixes for cost savings |
| **RAGAS** | Framework for evaluating RAG pipeline quality |
| **RAG** | Retrieval-Augmented Generation |
| **Rate limiting** | API caps on requests/tokens per time period |
| **Reranking** | Scoring and reordering search results for relevance |
| **Semantic chunking** | Splitting text by meaning, not fixed size |
| **Token** | Subword piece (~¾ of a word) |
| **Top-k** | Return the k most similar results |
| **Trace** | Record of everything that happened during one LLM interaction |
| **TTFB** | Time To First Byte — latency to start receiving response |
| **Vector** | A list of numbers (an embedding is a type of vector) |

---

## 🗺️ How Days 4–6 Fit Into the Full Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR AI ENGINEERING JOURNEY                │
│                                                              │
│  Days 1-3: APIs → PROMPTS → FUNCTION CALLING                │
│  "I can talk to LLMs and make them use tools"                │
│           ↓                                                  │
│  Day 4: EMBEDDINGS + VECTOR STORAGE                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Documents → Embedding Model → pgvector (Postgres)    │    │
│  │ "I can convert text to searchable vectors"            │    │
│  └─────────────────────────────────────────────────────┘    │
│           ↓                                                  │
│  Day 5: RAG PIPELINE                                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Question → Embed → Search → Context → LLM → Answer   │    │
│  │ "I can build knowledge-grounded AI applications"      │    │
│  └─────────────────────────────────────────────────────┘    │
│           ↓                                                  │
│  Day 6: OBSERVABILITY + COST CONTROL                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Trace → Monitor → Budget → Alert → Optimize          │    │
│  │ "I can run AI in production without going broke"      │    │
│  └─────────────────────────────────────────────────────┘    │
│           ↓                                                  │
│  Days 7-12: AGENTS → MCP → MULTI-AGENT → EVAL → PROD      │
│  "I can build autonomous AI systems at scale"                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

> 💡 **Pro Tip for Interviews:** Don't just memorize definitions — for each concept, have a **story**: "I built a RAG pipeline using pgvector that reduced customer support response time by 60%." That's what interviewers remember.

> **Next Steps:** Run your Day 4-6 code to build muscle memory. Then try: (1) different embedding models, (2) hybrid search, (3) adding reranking, (4) setting up Langfuse observability. Each improvement is a resume-worthy bullet point.

---

*Generated with current market research as of February 2026. Pricing and model availability evolve frequently — check provider docs for latest.*
