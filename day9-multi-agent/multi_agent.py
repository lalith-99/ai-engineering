"""
Day 9: Multi-agent system using OpenAI Agents SDK patterns.

Three agents collaborate on a task:
- Researcher: gathers information
- Analyst: processes and draws conclusions
- Writer: produces the final output

This shows handoff patterns and agent coordination without
needing the full CrewAI/LangGraph dependency.

Usage:
    python multi_agent.py "Analyze the pros and cons of microservices vs monolith"
    python multi_agent.py "Compare Redis vs Memcached for session caching"
"""

import os
import sys
import json
from dataclasses import dataclass
from openai import OpenAI


def get_client() -> OpenAI:
    """Return an OpenAI client from env."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


@dataclass
class AgentResult:
    agent_name: str
    output: str
    tokens_used: int


def run_agent(client: OpenAI, name: str, role: str, task: str, context: str = "") -> AgentResult:
    """Run a single agent with a specific role and task."""
    messages = [
        {"role": "system", "content": role},
    ]
    if context:
        messages.append({"role": "user", "content": f"Previous agent output:\n{context}\n\nYour task: {task}"})
    else:
        messages.append({"role": "user", "content": task})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=500,
    )

    if not response.choices or response.choices[0].message.content is None:
        raise ValueError(f"{name} returned no content")
    tokens_used = response.usage.total_tokens if response.usage else 0
    return AgentResult(
        agent_name=name,
        output=response.choices[0].message.content,
        tokens_used=tokens_used,
    )


def run_pipeline(topic: str) -> dict:
    """
    Three-agent pipeline:
    Researcher -> Analyst -> Writer

    Each agent gets the previous agent's output as context.
    """
    client = get_client()
    results = []
    total_tokens = 0

    # Agent 1: Researcher
    print("\n[1/3] Researcher gathering information...")
    researcher = run_agent(
        client,
        name="Researcher",
        role=(
            "You are a technical researcher. Gather key facts, data points, "
            "and different perspectives on the given topic. Be factual, not opinionated. "
            "List 5-8 key points."
        ),
        task=f"Research this topic: {topic}",
    )
    results.append(researcher)
    total_tokens += researcher.tokens_used
    print(f"   Done ({researcher.tokens_used} tokens)")

    # Agent 2: Analyst
    print("[2/3] Analyst processing research...")
    analyst = run_agent(
        client,
        name="Analyst",
        role=(
            "You are a technical analyst. Given research notes, identify patterns, "
            "tradeoffs, and form a clear recommendation. Be direct."
        ),
        task=f"Analyze these findings about '{topic}' and identify the key tradeoffs.",
        context=researcher.output,
    )
    results.append(analyst)
    total_tokens += analyst.tokens_used
    print(f"   Done ({analyst.tokens_used} tokens)")

    # Agent 3: Writer
    print("[3/3] Writer producing final output...")
    writer = run_agent(
        client,
        name="Writer",
        role=(
            "You are a technical writer. Given research and analysis, write a clear, "
            "concise summary with a recommendation. Use bullet points. "
            "Keep it under 200 words."
        ),
        task=f"Write a summary and recommendation for '{topic}'.",
        context=analyst.output,
    )
    results.append(writer)
    total_tokens += writer.tokens_used
    print(f"   Done ({writer.tokens_used} tokens)")

    return {
        "topic": topic,
        "agents": [{"name": r.agent_name, "output": r.output, "tokens": r.tokens_used} for r in results],
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(total_tokens / 1_000_000 * 0.15, 6),  # gpt-4o-mini pricing
    }


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Microservices vs Monolith for a startup"

    print(f"Topic: {topic}")
    result = run_pipeline(topic)

    print("\n" + "=" * 60)
    for agent in result["agents"]:
        print(f"\n--- {agent['name']} ---")
        print(agent["output"])

    print(f"\n--- Stats ---")
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Estimated cost: ${result['estimated_cost_usd']}")
