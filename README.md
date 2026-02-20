# AI Engineering

Notes, guides, and code from learning AI/ML engineering. Covers LLM APIs, RAG, agents, classical ML, and production patterns.

## Structure

```
Guides (reference material):
  DAY1-3   — LLM APIs, prompt engineering, function calling
  DAY4-6   — Embeddings, RAG, observability
  DAY7-12  — Agents, MCP, multi-agent, guardrails, eval, production
  DAY13-15 — Classical ML, deep learning, MLOps

Code (runnable examples):
  day1-2-apis-prompting/   — OpenAI API basics, prompt modes
  day3-function-calling/   — Tool use, SES integration
  day4-embeddings-pgvector/ — pgvector setup, similarity search
  day5-rag/                — RAG pipeline with chunking
  day6-observability-cost/ — Cost tracking, call logging
  day7-agents/             — ReAct agent loop with tools
  day8-mcp/                — MCP server (project tracker)
  day9-multi-agent/        — Multi-agent pipeline (researcher → analyst → writer)
  day10-guardrails/        — Pydantic structured output, content moderation
  day11-evaluation/        — LLM eval: contains-check + LLM-as-judge
  day12-production/        — Retry, caching, rate limiting, budget tracking
  day13-classical-ml/      — Linear/logistic regression, RF, XGBoost, K-Means
  day14-deep-learning/     — Neural net from scratch, PyTorch, self-attention
  day15-mlops/             — Model versioning, drift detection, FastAPI serving
```

## Running the Code

Each day folder has its own `requirements.txt`. To run:

```bash
cd day7-agents
pip install -r requirements.txt
python agent.py "What's the weather in Austin?"
```

Most examples need `OPENAI_API_KEY` set. See `.env.example` for all env vars.

## What the Guides Cover

**Days 1–3:** API landscape (OpenAI, Anthropic, Google), tokenization, cost optimization, prompt engineering (few-shot, chain-of-thought), structured outputs, function calling

**Days 4–6:** Embedding models, vector databases (pgvector, Pinecone, Qdrant), RAG pipeline (chunking, retrieval, reranking), observability (Langfuse, Helicone), cost control

**Days 7–12:** Agent architectures (ReAct, plan-and-execute), MCP, multi-agent frameworks (CrewAI, LangGraph), guardrails, fine-tuning, RLHF, evaluation strategies

**Days 13–15:** Classical ML algorithms, bias-variance tradeoff, neural networks, transformers and self-attention, classification/regression metrics, MLOps, data drift, model serving

## Notes

- Pricing and model info as of Feb 2026
- Python 3.10+
- Some examples use mock data so they run without external services
