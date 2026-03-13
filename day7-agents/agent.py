"""
Day 7: ReAct agent loop with tool use.

Implements the Reason-Act-Observe pattern using OpenAI function calling.
The agent can look up weather, do math, and search a mock knowledge base.
It decides which tools to call and loops until it has an answer.

Usage:
    python agent.py "What's the weather in Austin and convert the temp to Celsius?"
    python agent.py "What is 47 * 89 + 12?"
    python agent.py "Tell me about caching strategies"
"""

import os
import json
import sys
from openai import OpenAI

MODEL_NAME = "gpt-4o-mini"
DEFAULT_MAX_ITERATIONS = 5
ALLOWED_MATH_CHARS = set("0123456789+-*/().% ")


def get_client() -> OpenAI:
    """Return an OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


# ========== TOOLS ==========

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate, e.g. '47 * 89 + 12'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Search internal knowledge base for technical topics",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
]

# Mock implementations
MOCK_WEATHER = {
    "austin": {"temp_f": 85, "condition": "Sunny", "humidity": 45},
    "seattle": {"temp_f": 58, "condition": "Cloudy", "humidity": 78},
    "new york": {"temp_f": 72, "condition": "Partly cloudy", "humidity": 55},
}

KNOWLEDGE_BASE = {
    "caching": "Cache-aside: app checks cache first, loads from DB on miss. Write-through: writes go to cache and DB simultaneously. TTL-based expiration prevents stale data. Redis and Memcached are the most common in-memory stores.",
    "load balancing": "Round-robin distributes evenly. Least-connections routes to the server with fewest active connections. Consistent hashing is used for stateful services.",
    "database": "Use read replicas for read-heavy workloads. Partition/shard for write-heavy. Connection pooling (PgBouncer) reduces overhead. Indexes speed up reads but slow writes.",
}


def execute_tool(name: str, args: dict) -> str:
    """Execute a tool call."""
    if name == "get_weather":
        city = args["city"].lower()
        data = MOCK_WEATHER.get(city)
        if data:
            return json.dumps(data)
        return json.dumps({"error": f"No weather data for {city}"})

    elif name == "calculate":
        try:
            # only allow safe math characters
            expr = args["expression"]
            if not all(c in ALLOWED_MATH_CHARS for c in expr):
                return json.dumps({"error": "Invalid characters in expression"})
            result = eval(expr)
            return json.dumps({"expression": expr, "result": result})
        except Exception as e:
            return json.dumps({"error": str(e)})

    elif name == "search_knowledge":
        query = args["query"].lower()
        for topic, content in KNOWLEDGE_BASE.items():
            if topic in query:
                return json.dumps({"topic": topic, "content": content})
        return json.dumps({"result": "No matching articles found"})

    return json.dumps({"error": f"Unknown tool: {name}"})


# ========== AGENT LOOP ==========

def run_agent(user_input: str, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> str:
    """
    ReAct loop:
    1. Send messages to LLM
    2. If LLM returns tool calls -> execute them, append results
    3. Loop until LLM gives a final text response or we hit max iterations
    """
    client = get_client()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to tools. "
                "Use them when needed to answer the user's question. "
                "Think step by step."
            ),
        },
        {"role": "user", "content": user_input},
    ]

    for i in range(max_iterations):
        print(f"\n--- Iteration {i + 1} ---")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        # If no tool calls, we have the final answer
        if not msg.tool_calls:
            print(f"Agent response: {msg.content}")
            return msg.content

        # Process each tool call
        messages.append(msg)
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            print(f"  Tool: {fn_name}({fn_args})")

            result = execute_tool(fn_name, fn_args)
            print(f"  Result: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Agent reached max iterations without a final answer."


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What's the weather in Austin?"
    print(f"Query: {query}")
    answer = run_agent(query)
    print(f"\nFinal: {answer}")
