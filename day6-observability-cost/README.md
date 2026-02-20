# Day 6: Observability & Cost Control

Track tokens, measure latency, enforce budgets, handle failures — the production side of LLM work.

## Why This Matters
- GPT-4o costs **66x more** than GPT-4o-mini for the same tokens
- One runaway prompt loop can burn $50 before you notice
- Without latency tracking, you can't optimize user experience
- Without retries, transient API failures = user-facing errors

## Quick Start

```bash
pip3 install -r requirements.txt

# Run all demos (recommended first time)
python3 observability.py --mode demo

# Single tracked chat call
python3 observability.py --mode chat --question "explain caching"

# Budget enforcement (watch it stop when budget runs out)
python3 observability.py --mode budget --budget 500

# Retry behavior
python3 observability.py --mode retry
```

## What's Tracked Per Call

| Metric | How |
|--------|-----|
| **Prompt tokens** | From `response.usage.prompt_tokens` |
| **Completion tokens** | From `response.usage.completion_tokens` |
| **Cost (USD)** | Calculated from model pricing table |
| **Latency (ms)** | Wall-clock time for API round-trip |
| **Attempt #** | Which retry attempt succeeded |
| **Status** | success / retry / rate_limited / error / budget_blocked |

## Key Patterns

### 1. Pre-flight Token Estimation
```python
estimated = estimate_message_tokens(messages)
# Check budget BEFORE calling the API
```

### 2. Max Token Guard
```python
MAX_SAFE_TOKENS = 4000
if max_tokens > MAX_SAFE_TOKENS:
    max_tokens = MAX_SAFE_TOKENS  # clamp to prevent $$$
```

### 3. Exponential Backoff
```
Attempt 1: wait 1s
Attempt 2: wait 2s
Attempt 3: wait 4s
Rate limits: 2x longer (2s, 4s, 8s)
```

### 4. Budget Enforcement
```python
tracker.budget_tokens = 1000
# Every call checks: if total_tokens + estimated > budget → skip
```

## Pricing Reference (per 1M tokens)

| Model | Input | Output |
|-------|-------|--------|
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4o | $2.50 | $10.00 |
| text-embedding-3-small | $0.02 | — |

## Production Additions

- **Structured logging** (JSON logs → ELK/Datadog)
- **Prometheus metrics** (token counters, latency histograms)
- **Alerts** on cost spikes or error rate
- **Per-user budgets** with Redis counters
- **Streaming** token counting (harder but important for TTFB)
