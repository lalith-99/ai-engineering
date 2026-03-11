"""
Day 1-2: OpenAI API + Prompt Engineering
==========================================
Covers: basic chat, streaming, JSON mode, structured outputs (Pydantic),
        few-shot prompting, chain-of-thought, self-consistency, token counting.

Usage:
  python3 call_openai.py --mode basic --question "What is an LLM?"
  python3 call_openai.py --mode streaming --question "Explain caching"
  python3 call_openai.py --mode json --topic "vector databases"
  python3 call_openai.py --mode structured --topic "API rate limiting"
  python3 call_openai.py --mode few-shot --question "What is a load balancer?"
  python3 call_openai.py --mode chain-of-thought --question "How many r's in strawberry?"
  python3 call_openai.py --mode self-consistency --question "Is bubble sort stable?"
  python3 call_openai.py --mode tokens --text "Hello, how are you today?"
  python3 call_openai.py --mode basic --preset structured
"""

import os
import json
import argparse
import sys
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("OpenAI SDK not installed. Run: pip3 install -r requirements.txt")

try:
    import tiktoken
except ImportError:
    raise SystemExit("tiktoken not installed. Run: pip3 install -r requirements.txt")

try:
    from pydantic import BaseModel, Field
except ImportError:
    raise SystemExit("pydantic not installed. Run: pip3 install -r requirements.txt")


DEFAULT_MODEL = "gpt-4o-mini"
TOKENS_PER_MILLION = 1_000_000


def get_client() -> OpenAI:
    """Create an OpenAI client from env."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY. Export it, then 'source ~/.zshrc'.")
    return OpenAI(api_key=api_key)


# ========== TOKEN COUNTING (interview must-know) ==========


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """Count tokens using tiktoken — matches OpenAI's tokenizer exactly."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str = DEFAULT_MODEL) -> float:
    """Estimate cost in USD from token counts."""
    pricing = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    }
    p = pricing.get(model, pricing[DEFAULT_MODEL])
    return (prompt_tokens / TOKENS_PER_MILLION) * p["input"] + (completion_tokens / TOKENS_PER_MILLION) * p["output"]


def call_basic(
    client: OpenAI,
    question: str,
    temperature: float,
    max_tokens: int,
    system_content: str,
) -> str:
    """Basic chat demonstrating system vs user prompts."""
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
        ],
    )
    usage = resp.usage
    cost = estimate_cost(usage.prompt_tokens, usage.completion_tokens)
    print(f"   Tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}")
    print(f"   Cost: ${cost:.6f}")
    return resp.choices[0].message.content


# ========== STREAMING (interview: "time to first token") ==========


def call_streaming(
    client: OpenAI,
    question: str,
    temperature: float,
    max_tokens: int,
    system_content: str,
) -> str:
    """
    Streaming chat — tokens arrive one by one for instant UX.

    Why streaming matters:
      - Time To First Token (TTFB) drops from seconds to ~200ms
      - Users see text appear in real-time (like ChatGPT)
      - Essential for chatbot UX — no staring at a loading spinner
    """
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,  # <-- the key parameter
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
        ],
    )
    collected = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            sys.stdout.write(delta.content)
            sys.stdout.flush()
            collected.append(delta.content)
    print()  # newline after streaming
    return "".join(collected)


def call_json(client: OpenAI, topic: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    """Structured prompting: force strict JSON output and parse it."""
    if not topic.strip():
        raise SystemExit("Topic cannot be empty.")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a data formatter. Always return valid JSON without any extra text. "
                    "Use simple language suitable for a beginner software engineer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}.\n"
                    "Return a JSON object with keys: \n"
                    "- title (string) \n"
                    "- summary (string, 2-3 sentences) \n"
                    "- keywords (array of 3-6 short strings) \n"
                    "- confidence (number between 0 and 1)."
                ),
            },
        ],
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise SystemExit("Model did not return valid JSON. Try lowering temperature or using response_format.")
    return data


def call_few_shot(client: OpenAI, question: str, temperature: float, max_tokens: int) -> str:
    """Few-shot: Show examples in the prompt to guide output consistency."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a technical writer. Return concise bullet points. "
                    "Format: one short sentence per bullet."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Here are examples of good answers:\n\n"
                    "Q: What is a cache?\n"
                    "A:\n"
                    "- Stores frequently accessed data in fast memory.\n"
                    "- Reduces database queries and latency.\n"
                    "- Trade-off: consistency vs speed.\n\n"
                    # "Q: What is a database index?\n"
                    # "A:\n"
                    # "- Data structure that speeds up queries.\n"
                    # "- Trades storage for query performance.\n"
                    # "- Slows writes slightly but accelerates reads.\n\n"
                    f"Now answer this the same way:\nQ: {question}\nA:"
                ),
            },
        ],
    )
    return resp.choices[0].message.content


def call_chain_of_thought(client: OpenAI, question: str, temperature: float, max_tokens: int) -> str:
    """Chain-of-thought: Ask model to reason step-by-step before answering."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": "You are a thoughtful engineer. Think step-by-step, then give a final answer.",
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    "Please think through this carefully:\n"
                    "1. Break down the problem.\n"
                    "2. Identify key components.\n"
                    "3. Reason through the solution.\n"
                    "Then provide a clear final answer."
                ),
            },
        ],
    )
    return resp.choices[0].message.content


# ========== STRUCTURED OUTPUTS WITH PYDANTIC (interview: "schema-enforced") ==========


class TechSummary(BaseModel):
    """Pydantic model for structured LLM output — guaranteed schema compliance."""
    title: str = Field(description="Short title of the topic")
    summary: str = Field(description="2-3 sentence summary")
    keywords: List[str] = Field(description="3-6 relevant keywords")
    difficulty: str = Field(description="beginner, intermediate, or advanced")
    use_cases: List[str] = Field(description="2-3 real-world use cases")


def call_structured(client: OpenAI, topic: str, temperature: float, max_tokens: int) -> TechSummary:
    """
    Structured outputs: force the model to return a Pydantic-validated object.

    Why this matters:
      - JSON mode can still produce invalid/unexpected schemas
      - Structured outputs with json_schema GUARANTEE schema compliance
      - In production, use the Instructor library for any-provider support
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "tech_summary",
                "schema": TechSummary.model_json_schema(),
                "strict": True,
            },
        },
        messages=[
            {
                "role": "system",
                "content": "You are a technical educator. Return structured data about the given topic.",
            },
            {
                "role": "user",
                "content": f"Provide a structured summary of: {topic}",
            },
        ],
    )
    data = json.loads(resp.choices[0].message.content)
    return TechSummary(**data)


# ========== SELF-CONSISTENCY (interview: "majority voting for reliability") ==========


def call_self_consistency(client: OpenAI, question: str, runs: int = 5) -> str:
    """
    Self-consistency: run the same prompt multiple times, take majority vote.

    Why: single LLM calls can be wrong. Multiple runs + voting gives much
    higher reliability for factual or classification tasks.
    """
    answers = []
    print(f"   Running {runs} independent calls...")
    for i in range(runs):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,  # needs some randomness for diversity
            max_tokens=100,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the question with a single short answer. "
                        "Be concise — one sentence maximum."
                    ),
                },
                {"role": "user", "content": question},
            ],
        )
        answer = resp.choices[0].message.content.strip()
        answers.append(answer)
        print(f"   Run {i+1}: {answer[:80]}...")

    # Simple majority: show all answers and let user see consistency
    print(f"\n   All {runs} answers collected. Consistency check:")
    for i, a in enumerate(answers, 1):
        print(f"     [{i}] {a}")
    return answers[0]  # In production, you'd implement proper voting


def get_preset_config(preset: str) -> Optional[Dict[str, Any]]:
    """Preset prompts to compare naive vs structured prompting quickly."""
    presets = {
        "naive": {
            "system": "You are a concise, friendly assistant for a junior SDE.",
            "question": "Tell me about API rate limits.",
            "temperature": 0.7,
        },
        "structured": {
            "system": (
                "You are a senior backend mentor. Be concise, use numbered steps, call out failure modes. "
                "Audience: junior SDE."
            ),
            "question": (
                "Explain how to design prompts for consistent outputs. Include: (1) system vs user, "
                "(2) constraints, (3) examples. End with one Do and one Don't."
            ),
            "temperature": 0.2,
        },
        "guardrails": {
            "system": (
                "You are a cautious engineer. Return exactly 5 bullets, each under 15 words. "
                "If code is requested, include one minimal fenced code block."
            ),
            "question": (
                "How to handle API rate limits in production? Include one Python retry snippet."
            ),
            "temperature": 0.3,
            "max_tokens": 220,
        },
    }
    return presets.get(preset)


def main():
    parser = argparse.ArgumentParser(description="Day 1-2: OpenAI API + Prompting")
    parser.add_argument(
        "--mode",
        choices=["basic", "streaming", "json", "structured", "few-shot", "chain-of-thought", "self-consistency", "tokens"],
        default="basic",
        help="basic | streaming | json | structured (Pydantic) | few-shot | chain-of-thought | self-consistency | tokens",
    )
    parser.add_argument(
        "--preset",
        choices=["naive", "structured", "guardrails"],
        help="Optional preset prompt for quick comparison (basic mode only).",
    )
    parser.add_argument(
        "--question",
        default="Explain the difference between system and user prompts in one short paragraph.",
    )
    parser.add_argument("--topic", default="LLM APIs, prompting, temperature, max tokens")
    parser.add_argument("--text", default="", help="Text to count tokens for (tokens mode)")
    parser.add_argument("--temperature", type=float, default=0.2, help="0.0 is deterministic; higher is more creative.")
    parser.add_argument("--max-tokens", type=int, default=300, help="Upper bound on generated tokens (not strict length).")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs for self-consistency mode.")
    args = parser.parse_args()

    # Token counting mode — no API call needed
    if args.mode == "tokens":
        text = args.text or args.question
        tokens = count_tokens(text)
        est_cost = estimate_cost(tokens, 100)  # assume 100 output tokens
        print(f"\n=== Token Counter ===")
        print(f"Text: \"{text}\"")
        print(f"Tokens: {tokens}")
        print(f"Estimated cost (with ~100 output tokens): ${est_cost:.6f}")

        # Show tokenization breakdown
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        token_ids = enc.encode(text)
        decoded = [enc.decode([t]) for t in token_ids]
        print(f"Token breakdown: {decoded}")
        return

    client = get_client()

    default_system = "You are a concise, friendly assistant for a junior SDE."

    if args.mode == "basic":
        question = args.question
        temperature = args.temperature
        max_tokens = args.max_tokens
        system_message = default_system

        if args.preset:
            preset_cfg = get_preset_config(args.preset)
            if preset_cfg:
                question = preset_cfg.get("question", question)
                system_message = preset_cfg.get("system", system_message)
                temperature = preset_cfg.get("temperature", temperature)
                max_tokens = preset_cfg.get("max_tokens", max_tokens)
            else:
                print("Preset not found; using defaults.")

        result = call_basic(client, question, temperature, max_tokens, system_message)
        print("\n=== Basic Chat ===")
        print(result)

    elif args.mode == "streaming":
        print("\n=== Streaming Chat (tokens arrive in real-time) ===")
        call_streaming(client, args.question, args.temperature, args.max_tokens, default_system)

    elif args.mode == "few-shot":
        result = call_few_shot(client, args.question, args.temperature, args.max_tokens)
        print("\n=== Few-Shot (Learn from Examples) ===")
        print(result)

    elif args.mode == "chain-of-thought":
        result = call_chain_of_thought(client, args.question, args.temperature, args.max_tokens)
        print("\n=== Chain-of-Thought (Reasoning) ===")
        print(result)

    elif args.mode == "structured":
        print("\n=== Structured Output (Pydantic Schema) ===")
        result = call_structured(client, args.topic, args.temperature, args.max_tokens)
        print(f"Title:      {result.title}")
        print(f"Summary:    {result.summary}")
        print(f"Keywords:   {result.keywords}")
        print(f"Difficulty: {result.difficulty}")
        print(f"Use Cases:  {result.use_cases}")

    elif args.mode == "self-consistency":
        print("\n=== Self-Consistency (Majority Voting) ===")
        call_self_consistency(client, args.question, args.runs)

    else:  # json
        if args.preset:
            print("Note: presets only affect basic mode; ignoring preset for JSON mode.")
        data = call_json(client, args.topic, args.temperature, args.max_tokens)
        print("\n=== JSON Output ===")
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
