"""
Day 6: Observability & Cost Control
=====================================
Track token usage, log latency, enforce max-token guards, timeouts + retries,
circuit breaker pattern, Langfuse integration.

This wraps OpenAI calls with production-grade observability so you can:
  - See exactly how many tokens each call burns (and the dollar cost)
  - Measure latency (embed, chat, total)
  - Guard against runaway costs with token budgets
  - Handle failures gracefully with retries + timeouts
  - Circuit breaker pattern to prevent cascading failures
  - Langfuse drop-in integration (3 lines)

Usage:
  python3 observability.py --mode demo                     # run all demos
  python3 observability.py --mode chat --question "explain caching"
  python3 observability.py --mode budget --budget 500       # enforce a token budget
  python3 observability.py --mode retry                     # demonstrate retry behavior
  python3 observability.py --mode circuit-breaker            # demo circuit breaker
  python3 observability.py --mode report                    # show session cost report
"""

import os
import json
import time
import argparse
import functools
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

try:
    from openai import OpenAI, APITimeoutError, RateLimitError, APIConnectionError, APIError
except ImportError:
    raise SystemExit("OpenAI SDK not installed. Run: pip3 install -r requirements.txt")

try:
    import tiktoken
except ImportError:
    raise SystemExit("tiktoken not installed. Run: pip3 install -r requirements.txt")


# ========== PRICING (Feb 2026, per 1M tokens) ==========

PRICING = {
    # Current generation (2025-2026)
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},       # legacy, still popular
    "gpt-4o": {"input": 2.50, "output": 10.00},            # previous flagship
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},       # ultra-cheap
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},       # budget workhorse
    "gpt-4.1": {"input": 2.00, "output": 8.00},            # coding-focused
    "gpt-5-nano": {"input": 0.05, "output": 0.40},         # cheapest smart model
    "gpt-5-mini": {"input": 0.25, "output": 2.00},         # balanced
    "gpt-5": {"input": 1.25, "output": 10.00},             # flagship
    "gpt-5.2": {"input": 1.75, "output": 14.00},           # frontier
    # Embeddings
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}

DEFAULT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


# ========== TOKEN COUNTER ==========


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """Count tokens using tiktoken (matches OpenAI tokenizer exactly)."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def estimate_message_tokens(messages: List[Dict[str, str]], model: str = DEFAULT_MODEL) -> int:
    """
    Estimate tokens for a list of chat messages.
    Each message has ~4 tokens overhead (role, content delimiters).
    """
    total = 0
    for msg in messages:
        total += 4  # message overhead
        total += count_tokens(msg.get("content", ""), model)
        total += count_tokens(msg.get("role", ""), model)
    total += 2  # assistant reply priming
    return total


# ========== COST CALCULATOR ==========


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate dollar cost from token counts."""
    pricing = PRICING.get(model)
    if not pricing:
        return 0.0
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


# ========== USAGE TRACKER (Session-level) ==========


@dataclass
class CallRecord:
    """Single API call record."""
    timestamp: float
    model: str
    call_type: str           # "chat" or "embedding"
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    status: str              # "success", "retry", "error"
    attempt: int = 1
    error: Optional[str] = None


@dataclass
class UsageTracker:
    """Tracks all API calls in a session for reporting."""
    records: List[CallRecord] = field(default_factory=list)
    budget_tokens: Optional[int] = None      # max total tokens allowed
    budget_usd: Optional[float] = None       # max dollars allowed

    def add(self, record: CallRecord):
        self.records.append(record)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def total_calls(self) -> int:
        return len(self.records)

    @property
    def avg_latency(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.latency_ms for r in self.records) / len(self.records)

    def check_budget(self, estimated_tokens: int = 0) -> bool:
        """Check if we're within budget. Returns True if OK."""
        if self.budget_tokens and (self.total_tokens + estimated_tokens) > self.budget_tokens:
            return False
        if self.budget_usd and self.total_cost > self.budget_usd:
            return False
        return True

    def print_report(self):
        """Print a formatted session report."""
        print(f"\n{'='*70}")
        print(f"SESSION USAGE REPORT")
        print(f"{'='*70}")
        print(f"  Total API calls:    {self.total_calls}")
        print(f"  Total tokens:       {self.total_tokens:,}")
        print(f"  Total cost:         ${self.total_cost:.6f}")
        print(f"  Avg latency:        {self.avg_latency:.0f}ms")

        if self.budget_tokens:
            used_pct = (self.total_tokens / self.budget_tokens) * 100
            bar = "█" * int(used_pct / 5) + "░" * (20 - int(used_pct / 5))
            print(f"  Token budget:       {self.total_tokens:,} / {self.budget_tokens:,} ({used_pct:.1f}%)")
            print(f"                      [{bar}]")

        if self.records:
            print(f"\n  {'#':<4} {'Type':<10} {'Model':<25} {'Tokens':>8} {'Cost':>10} {'Latency':>10} {'Status':<8}")
            print(f"  {'─'*4} {'─'*10} {'─'*25} {'─'*8} {'─'*10} {'─'*10} {'─'*8}")
            for i, r in enumerate(self.records, 1):
                print(
                    f"  {i:<4} {r.call_type:<10} {r.model:<25} {r.total_tokens:>8,} "
                    f"${r.cost_usd:>9.6f} {r.latency_ms:>8.0f}ms {r.status:<8}"
                )

        # Cost breakdown by model
        by_model: Dict[str, Dict[str, float]] = {}
        for r in self.records:
            if r.model not in by_model:
                by_model[r.model] = {"tokens": 0, "cost": 0.0, "calls": 0}
            by_model[r.model]["tokens"] += r.total_tokens
            by_model[r.model]["cost"] += r.cost_usd
            by_model[r.model]["calls"] += 1

        if by_model:
            print(f"\n  Cost by model:")
            for model, stats in by_model.items():
                print(f"    {model}: {stats['calls']} calls, {stats['tokens']:,} tokens, ${stats['cost']:.6f}")

        print(f"{'='*70}\n")


# Global tracker for the session
tracker = UsageTracker()


# ========== OBSERVED CHAT (wrapped with tracking) ==========


def observed_chat(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 300,
    timeout: float = 30.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Optional[str]:
    """
    Chat completion with full observability:
      - Pre-flight token estimation
      - Budget check
      - Retry with exponential backoff
      - Timeout enforcement
      - Post-call cost + latency logging
    """
    # Pre-flight: estimate tokens
    estimated_input = estimate_message_tokens(messages, model)
    estimated_total = estimated_input + max_tokens
    print(f"\nPre-flight: ~{estimated_input} input tokens, max {max_tokens} output -> ~{estimated_total} total")

    # Budget guard
    if not tracker.check_budget(estimated_total):
        print(f"BUDGET EXCEEDED: {tracker.total_tokens:,} tokens used, budget={tracker.budget_tokens:,}")
        print(f"   Skipping this call to stay within budget.")
        tracker.add(CallRecord(
            timestamp=time.time(), model=model, call_type="chat",
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            cost_usd=0.0, latency_ms=0.0, status="budget_blocked",
        ))
        return None

    # Max token guard: clamp max_tokens to prevent runaway costs
    MAX_SAFE_TOKENS = 4000
    if max_tokens > MAX_SAFE_TOKENS:
        print(f"Warning: max_tokens={max_tokens} exceeds guard ({MAX_SAFE_TOKENS}). Clamping.")
        max_tokens = MAX_SAFE_TOKENS

    # Retry loop with exponential backoff
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
                timeout=timeout,
            )
            latency_ms = (time.time() - t0) * 1000

            usage = response.usage
            cost = calculate_cost(model, usage.prompt_tokens, usage.completion_tokens)

            record = CallRecord(
                timestamp=time.time(),
                model=model,
                call_type="chat",
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                status="success",
                attempt=attempt,
            )
            tracker.add(record)

            # Log it
            print(f"Chat completed in {latency_ms:.0f}ms (attempt {attempt}/{max_retries})")
            print(f"   Tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}")
            print(f"   Cost: ${cost:.6f}")

            return response.choices[0].message.content

        except (APITimeoutError, APIConnectionError) as e:
            last_error = str(e)
            wait = retry_delay * (2 ** (attempt - 1))   # exponential backoff
            print(f"Timeout/connection error (attempt {attempt}/{max_retries}): {e}")
            print(f"   Retrying in {wait:.1f}s...")
            tracker.add(CallRecord(
                timestamp=time.time(), model=model, call_type="chat",
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
                cost_usd=0.0, latency_ms=0.0, status="retry",
                attempt=attempt, error=str(e),
            ))
            time.sleep(wait)

        except RateLimitError as e:
            last_error = str(e)
            wait = retry_delay * (2 ** (attempt - 1)) * 2   # longer backoff for rate limits
            print(f"Rate limited (attempt {attempt}/{max_retries}): {e}")
            print(f"   Retrying in {wait:.1f}s...")
            tracker.add(CallRecord(
                timestamp=time.time(), model=model, call_type="chat",
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
                cost_usd=0.0, latency_ms=0.0, status="rate_limited",
                attempt=attempt, error=str(e),
            ))
            time.sleep(wait)

        except APIError as e:
            last_error = str(e)
            print(f"API error (attempt {attempt}/{max_retries}): {e}")
            tracker.add(CallRecord(
                timestamp=time.time(), model=model, call_type="chat",
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
                cost_usd=0.0, latency_ms=0.0, status="error",
                attempt=attempt, error=str(e),
            ))
            break  # Don't retry non-transient errors

    print(f"All {max_retries} attempts failed. Last error: {last_error}")
    return None


def observed_embedding(
    client: OpenAI,
    text: str,
    model: str = EMBEDDING_MODEL,
    timeout: float = 15.0,
) -> Optional[List[float]]:
    """Embedding call with tracking."""
    t0 = time.time()
    try:
        response = client.embeddings.create(model=model, input=text, timeout=timeout)
        latency_ms = (time.time() - t0) * 1000
        usage = response.usage
        cost = calculate_cost(model, usage.prompt_tokens, 0)

        tracker.add(CallRecord(
            timestamp=time.time(), model=model, call_type="embedding",
            prompt_tokens=usage.prompt_tokens, completion_tokens=0,
            total_tokens=usage.total_tokens, cost_usd=cost,
            latency_ms=latency_ms, status="success",
        ))

        print(f"Embedding in {latency_ms:.0f}ms | {usage.total_tokens} tokens | ${cost:.6f}")
        return response.data[0].embedding

    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        tracker.add(CallRecord(
            timestamp=time.time(), model=model, call_type="embedding",
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            cost_usd=0.0, latency_ms=latency_ms, status="error", error=str(e),
        ))
        print(f"Embedding failed: {e}")
        return None


# ========== CLIENT ==========


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY. Export it, then 'source ~/.zshrc'.")
    return OpenAI(api_key=api_key)


# ========== CIRCUIT BREAKER (interview: "prevent cascading failures") ==========


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern for LLM API calls.

    States:
      CLOSED  (normal)     - requests flow through
      OPEN    (tripped)    - all requests rejected for `reset_timeout` seconds
      HALF-OPEN (testing)  - allow ONE request to test recovery

    Interview answer:
      "If we see N consecutive failures, we open the circuit to prevent
       hammering a failing API. After a cooldown, we try one call (half-open).
       If it succeeds, we close the circuit. If it fails, we stay open."
    """
    failure_threshold: int = 5       # failures before opening circuit
    reset_timeout: float = 30.0      # seconds to wait before half-open
    failure_count: int = 0
    state: str = "CLOSED"            # CLOSED, OPEN, HALF_OPEN
    last_failure_time: float = 0.0

    def record_success(self):
        """Reset on success."""
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        """Track failure, potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            print(f"   CIRCUIT OPENED after {self.failure_count} consecutive failures.")
            print(f"   All requests will be rejected for {self.reset_timeout}s.")

    def can_execute(self) -> bool:
        """Check if a request is allowed through."""
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.reset_timeout:
                self.state = "HALF_OPEN"
                print(f"   Circuit HALF-OPEN: testing with one request...")
                return True
            return False
        if self.state == "HALF_OPEN":
            return True  # allow the test request
        return False


# Global circuit breaker
circuit_breaker = CircuitBreaker()


# ========== LANGFUSE INTEGRATION (3-line drop-in) ==========
# Langfuse is the most popular open-source LLM observability platform.
# To enable, install langfuse and change ONE import:
#
#   pip install langfuse
#   from langfuse.openai import openai  # replaces: from openai import OpenAI
#
# That's it. All calls are now automatically traced with:
#   - Token counts, cost, latency
#   - Full prompt/response logging
#   - Parent-child traces for chains/agents
#
# Check dashboard at: https://cloud.langfuse.com
#
# For custom traces (more control):
#   from langfuse import Langfuse
#   langfuse = Langfuse()
#   trace = langfuse.trace(name="rag-query", user_id="user_123")
#   span = trace.span(name="embedding")
#   # ... your code ...
#   span.end()
#   trace.update(output="final answer here")


# ========== DEMO COMMANDS ==========


def cmd_chat(client: OpenAI, question: str, temperature: float, max_tokens: int):
    """Single observed chat call."""
    messages = [
        {"role": "system", "content": "You are a concise technical assistant. Keep answers under 3 sentences."},
        {"role": "user", "content": question},
    ]
    answer = observed_chat(client, messages, temperature=temperature, max_tokens=max_tokens)
    if answer:
        print(f"\nAnswer:\n{answer}")


def cmd_budget(client: OpenAI, budget: int):
    """Demonstrate budget enforcement by making calls until budget is exhausted."""
    tracker.budget_tokens = budget
    print(f"\nToken budget set to {budget:,} tokens")
    print(f"   Making multiple calls to show budget enforcement...\n")

    questions = [
        "What is a database index?",
        "Explain API rate limiting.",
        "How does a load balancer work?",
        "What is event sourcing?",
        "Describe the CAP theorem.",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n--- Call {i}/{len(questions)}: \"{q}\" ---")
        messages = [
            {"role": "system", "content": "Answer in one sentence."},
            {"role": "user", "content": q},
        ]
        result = observed_chat(client, messages, max_tokens=100)
        if result is None and not tracker.check_budget():
            print(f"\n   Budget exhausted after {i-1} calls. Remaining questions skipped.")
            break
        elif result:
            print(f"   -> {result[:100]}...")


def cmd_retry_demo(client: OpenAI):
    """Demonstrate the retry/timeout behavior with a normal call."""
    print("\nRetry Demo")
    print("   Making a normal call (retries would kick in on transient failures)...\n")

    messages = [
        {"role": "system", "content": "You are helpful. Respond in exactly one sentence."},
        {"role": "user", "content": "What are the benefits of observability in production systems?"},
    ]

    result = observed_chat(
        client, messages,
        timeout=30.0,
        max_retries=3,
        retry_delay=1.0,
        max_tokens=150,
    )
    if result:
        print(f"\n   {result}")

    print("\n   Retry config used:")
    print("   timeout=30s, max_retries=3, backoff=exponential (1s, 2s, 4s)")
    print("   Rate limit backoff: 2x longer (2s, 4s, 8s)")


def cmd_circuit_breaker_demo(client: OpenAI):
    """
    Demonstrate the circuit breaker pattern.

    Shows the three states: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
    """
    print("\nCircuit Breaker Demo")
    print("=" * 60)
    print(f"  Current state: {circuit_breaker.state}")
    print(f"  Failure threshold: {circuit_breaker.failure_threshold}")
    print(f"  Reset timeout: {circuit_breaker.reset_timeout}s\n")

    # Simulate some failures
    print("  Simulating failures to show state transitions:\n")
    for i in range(1, circuit_breaker.failure_threshold + 2):
        if circuit_breaker.can_execute():
            print(f"  [{i}] Request allowed (state={circuit_breaker.state})")
            # Simulate a failure
            circuit_breaker.record_failure()
        else:
            print(f"  [{i}] Request BLOCKED (state={circuit_breaker.state})")

    # Show that real calls are blocked
    print(f"\n  Circuit is now {circuit_breaker.state}.")
    print(f"  In production, requests would get a fallback response.")
    print(f"  After {circuit_breaker.reset_timeout}s, circuit enters HALF_OPEN.")
    print(f"  One successful call would move it back to CLOSED.\n")

    # Reset for the demo
    circuit_breaker.record_success()
    print(f"  (Reset for demo purposes: state={circuit_breaker.state})")

    # Now make a real call to show success flow
    print(f"\n  Making a real API call through circuit breaker...")
    if circuit_breaker.can_execute():
        messages = [
            {"role": "system", "content": "Respond in one sentence."},
            {"role": "user", "content": "What is a circuit breaker pattern?"},
        ]
        result = observed_chat(client, messages, max_tokens=100)
        if result:
            circuit_breaker.record_success()
            print(f"  Success! Circuit stays CLOSED.")
            print(f"  Answer: {result}")


def cmd_demo(client: OpenAI):
    """Run all demos to show the full observability picture."""
    print("\n" + "="*70)
    print("DAY 6 DEMO: Observability & Cost Control")
    print("="*70)

    # 1. Chat with tracking
    print("\n\nDEMO 1: Tracked Chat Call")
    print("-"*40)
    cmd_chat(client, "What is a circuit breaker pattern?", temperature=0.2, max_tokens=200)

    # 2. Embedding with tracking
    print("\n\nDEMO 2: Tracked Embedding Call")
    print("-"*40)
    observed_embedding(client, "Circuit breakers prevent cascading failures in microservices")

    # 3. Token estimation
    print("\n\nDEMO 3: Token Estimation (no API call)")
    print("-"*40)
    sample_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the difference between SQL and NoSQL databases in detail."},
    ]
    est = estimate_message_tokens(sample_messages)
    print(f"   Estimated tokens for messages: {est}")
    print(f"   With max_tokens=500: ~{est + 500} total -> ${calculate_cost(DEFAULT_MODEL, est, 500):.6f}")

    # 4. Cost comparison across models (updated Feb 2026 pricing)
    print("\n\nDEMO 4: Cost Comparison (500 input + 300 output tokens)")
    print("-"*40)
    prompt_tokens = 500
    completion_tokens = 300
    models_to_compare = ["gpt-5-nano", "gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini", "gpt-4.1", "gpt-5", "gpt-5.2"]
    for model_name in models_to_compare:
        cost = calculate_cost(model_name, prompt_tokens, completion_tokens)
        print(f"   {model_name:25} -> ${cost:.6f}")

    # 5. Circuit breaker overview
    print("\n\nDEMO 5: Circuit Breaker State")
    print("-"*40)
    print(f"   State: {circuit_breaker.state}")
    print(f"   Failure count: {circuit_breaker.failure_count}/{circuit_breaker.failure_threshold}")
    print("   (Run --mode circuit-breaker for full demo)")

    # 6. Session report
    tracker.print_report()


# ========== MAIN ==========


def main():
    parser = argparse.ArgumentParser(description="Day 6: Observability & Cost Control")
    parser.add_argument(
        "--mode",
        choices=["demo", "chat", "budget", "retry", "circuit-breaker", "report"],
        default="demo",
        help="demo | chat | budget | retry | circuit-breaker | report",
    )
    parser.add_argument("--question", default="What is a circuit breaker pattern?", help="Question for chat mode")
    parser.add_argument("--temperature", type=float, default=0.2, help="Model temperature")
    parser.add_argument("--max-tokens", type=int, default=300, help="Max output tokens")
    parser.add_argument("--budget", type=int, default=500, help="Token budget for budget mode")
    args = parser.parse_args()

    client = get_client()

    if args.mode == "demo":
        cmd_demo(client)
    elif args.mode == "chat":
        cmd_chat(client, args.question, args.temperature, args.max_tokens)
        tracker.print_report()
    elif args.mode == "budget":
        cmd_budget(client, args.budget)
        tracker.print_report()
    elif args.mode == "retry":
        cmd_retry_demo(client)
        tracker.print_report()
    elif args.mode == "circuit-breaker":
        cmd_circuit_breaker_demo(client)
        tracker.print_report()
    elif args.mode == "report":
        print("No calls made yet. Run with --mode demo or --mode chat first.")
        tracker.print_report()


if __name__ == "__main__":
    main()
