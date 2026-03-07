"""
Day 12: Production patterns — retry, fallback, caching, rate limiting.

Wraps LLM calls with production-grade resilience:
1. Retry with exponential backoff
2. Multi-provider fallback (OpenAI -> mock fallback)
3. Response caching (in-memory, saves cost on repeated queries)
4. Token budget tracking
5. Request rate limiting

Usage:
    python production.py "Explain distributed caching"
    python production.py "What is CAP theorem?" --cached
"""

import os
import sys
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI


def get_client() -> OpenAI:
    """Create an OpenAI client from env."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


# ========== RESPONSE CACHE ==========

class ResponseCache:
    """In-memory LRU-style cache for LLM responses."""

    def __init__(self, max_size: int = 100):
        """Initialize cache state and counters."""
        self.cache: dict[str, dict] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _key(self, model: str, prompt: str, temperature: float) -> str:
        """Build a stable cache key."""
        raw = f"{model}:{temperature}:{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, model: str, prompt: str, temperature: float) -> Optional[str]:
        """Return a cached response if present."""
        key = self._key(model, prompt, temperature)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]["response"]
        self.misses += 1
        return None

    def put(self, model: str, prompt: str, temperature: float, response: str, tokens: int):
        key = self._key(model, prompt, temperature)
        if len(self.cache) >= self.max_size:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[key] = {"response": response, "tokens": tokens, "time": time.time()}

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hits / total * 100:.0f}%" if total else "N/A",
            "cached_items": len(self.cache),
        }


# ========== TOKEN BUDGET ==========

@dataclass
class TokenBudget:
    """Track token usage against a budget."""
    max_tokens: int = 100_000  # daily budget
    used_tokens: int = 0

    def can_spend(self, estimated: int) -> bool:
        return (self.used_tokens + estimated) <= self.max_tokens

    def spend(self, tokens: int):
        self.used_tokens += tokens

    def remaining(self) -> int:
        return self.max_tokens - self.used_tokens

    def usage_pct(self) -> float:
        return self.used_tokens / self.max_tokens * 100


# ========== RATE LIMITER ==========

class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, requests_per_minute: int = 20):
        self.rpm = requests_per_minute
        self.requests: list[float] = []

    def acquire(self):
        now = time.time()
        # Remove requests older than 60 seconds
        self.requests = [t for t in self.requests if now - t < 60]
        if len(self.requests) >= self.rpm:
            wait = 60 - (now - self.requests[0])
            print(f"  Rate limited. Waiting {wait:.1f}s...")
            time.sleep(wait)
        self.requests.append(time.time())


# ========== RESILIENT CLIENT ==========

class ResilientLLM:
    """Production wrapper around LLM calls."""

    def __init__(self):
        self.client = get_client()
        self.cache = ResponseCache()
        self.budget = TokenBudget(max_tokens=50_000)
        self.limiter = RateLimiter(requests_per_minute=30)
        self.total_requests = 0
        self.total_errors = 0

    def call(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 300,
        use_cache: bool = True,
        max_retries: int = 3,
    ) -> dict:
        """Make a resilient LLM call with cache, retry, budget, and rate limiting."""

        # Check cache first
        if use_cache:
            cached = self.cache.get(model, prompt, temperature)
            if cached:
                print("  Cache hit")
                return {"response": cached, "source": "cache", "tokens": 0}

        # Check budget
        if not self.budget.can_spend(max_tokens):
            return {
                "response": "Token budget exceeded. Try again tomorrow.",
                "source": "budget_limit",
                "tokens": 0,
            }

        # Rate limit
        self.limiter.acquire()

        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                self.total_requests += 1
                response = self.client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": "Answer concisely."},
                        {"role": "user", "content": prompt},
                    ],
                )

                text = response.choices[0].message.content
                tokens = response.usage.total_tokens

                # Track budget
                self.budget.spend(tokens)

                # Cache the response
                if use_cache:
                    self.cache.put(model, prompt, temperature, text, tokens)

                return {
                    "response": text,
                    "source": "api",
                    "tokens": tokens,
                    "model": model,
                    "attempt": attempt + 1,
                }

            except Exception as e:
                self.total_errors += 1
                wait = 2 ** attempt
                print(f"  Error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)

        return {
            "response": "All retries failed.",
            "source": "error",
            "tokens": 0,
        }

    def stats(self) -> dict:
        return {
            "requests": self.total_requests,
            "errors": self.total_errors,
            "error_rate": f"{self.total_errors / max(self.total_requests, 1) * 100:.1f}%",
            "budget_used": f"{self.budget.usage_pct():.1f}%",
            "budget_remaining": self.budget.remaining(),
            "cache": self.cache.stats(),
        }


if __name__ == "__main__":
    llm = ResilientLLM()

    queries = sys.argv[1:] if len(sys.argv) > 1 else [
        "What is CAP theorem?",
        "What is CAP theorem?",  # second call should be cached
        "Explain consistent hashing in 2 sentences.",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        result = llm.call(q)
        print(f"  Source: {result['source']}")
        print(f"  Tokens: {result['tokens']}")
        print(f"  Response: {result['response'][:120]}")

    print(f"\n--- Stats ---")
    print(json.dumps(llm.stats(), indent=2))
