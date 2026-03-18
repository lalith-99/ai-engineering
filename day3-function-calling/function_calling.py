"""
Day 3: Function Calling / Tool Use
====================================
The 5-step tool calling flow + production patterns.

Covers: tool definitions, tool registry pattern, tool_choice control,
        multi-round (sequential) tool calling, parallel tool calls.

Usage:
  python3 function_calling.py --example simple
  python3 function_calling.py --example multi-tool
  python3 function_calling.py --example complex
  python3 function_calling.py --example no-tool
  python3 function_calling.py --request "What's the weather in NYC?"
  python3 function_calling.py --example simple --tool-choice required
  python3 function_calling.py --list-examples
"""

import os
import json
import argparse
from typing import Any, Dict, List

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("OpenAI SDK not installed. Run: pip3 install -r requirements.txt")


def get_client() -> OpenAI:
    """Create an OpenAI client from environment config."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY. Export it, then 'source ~/.zshrc'.")
    return OpenAI(api_key=api_key)


# ========== MOCK TOOLS (These would be real functions in production) ==========


def send_notification(user_id: str, message: str, priority: str = "normal") -> Dict[str, Any]:
    """Send a notification to a user (mock - would hit Nimbus API in real life)."""
    if not str(user_id).strip() or not str(message).strip():
        return {"error": "user_id and message are required"}
    if priority not in {"low", "normal", "high"}:
        return {"error": f"invalid priority: {priority}"}
    print(f"\n[TOOL EXECUTED] send_notification(user_id={user_id}, message={message}, priority={priority})")
    return {
        "status": "sent",
        "notification_id": f"notif_{user_id}_12345",
        "user_id": user_id,
        "message": message,
        "priority": priority,
        "timestamp": "2026-01-11T10:30:00Z",
    }


def get_weather(location: str) -> Dict[str, Any]:
    """Get weather for a location (mock - would hit weather API in real life)."""
    location = str(location).strip()
    if not location:
        return {"error": "location is required"}
    print(f"\n[TOOL EXECUTED] get_weather(location={location})")
    # Mock data
    weather_data = {
        "San Francisco": {"temp": "62°F", "condition": "Cloudy", "humidity": "75%"},
        "New York": {"temp": "45°F", "condition": "Clear", "humidity": "60%"},
        "Seattle": {"temp": "50°F", "condition": "Rainy", "humidity": "85%"},
    }
    result = weather_data.get(location, {"temp": "70°F", "condition": "Unknown", "humidity": "50%"})
    result["location"] = location
    return result


def query_database(table: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Query a database table (mock - would hit Postgres in real life)."""
    if filters is None:
        filters = {}
    print(f"\n[TOOL EXECUTED] query_database(table={table}, filters={filters})")
    # Mock data
    if table == "notifications":
        return [
            {"id": 1, "user_id": "user_123", "status": "delivered", "created_at": "2026-01-10"},
            {"id": 2, "user_id": "user_123", "status": "pending", "created_at": "2026-01-11"},
        ]
    elif table == "users":
        return [{"id": "user_123", "name": "John Doe", "email": "john@example.com", "active": True}]
    return []


# ========== TOOL DEFINITIONS (What the model sees) ==========


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": "Send a notification to a user. Use this when the user asks to notify someone or send a message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The ID of the user to notify (e.g., 'user_123')",
                    },
                    "message": {"type": "string", "description": "The notification message text"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high"],
                        "description": "Priority level of the notification",
                    },
                },
                "required": ["user_id", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location. Use when user asks about weather conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name (e.g., 'San Francisco')"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Query a database table. Use when user asks for data from notifications or users tables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "enum": ["notifications", "users"],
                        "description": "Table name to query",
                    },
                    "filters": {
                        "type": "object",
                        "description": "Key-value filters (e.g., {'user_id': 'user_123'})",
                    },
                },
                "required": ["table"],
            },
        },
    },
]


# ========== TOOL EXECUTION ROUTER (Registry Pattern — production-grade) ==========

# Production pattern: dictionary registry instead of if/elif chains.
# Easy to add/remove tools dynamically. Interview tip: always mention this.
TOOL_REGISTRY = {
    "send_notification": send_notification,
    "get_weather": get_weather,
    "query_database": query_database,
}


def execute_tool(tool_name: str, tool_args: Dict[str, Any]) -> Any:
    """Route tool calls via registry. Extensible — just add to TOOL_REGISTRY."""
    func = TOOL_REGISTRY.get(tool_name)
    if not func:
        return {"error": f"tool lookup failed for {tool_name}"}
    if tool_args is None:
        tool_args = {}
    if not isinstance(tool_args, dict):
        return {"error": f"invalid args for tool {tool_name}: expected object"}
    try:
        return func(**tool_args)
    except TypeError as exc:
        return {"error": f"tool {tool_name} rejected args: {exc}"}
    except Exception as exc:
        return {"error": f"tool {tool_name} failed: {exc}"}


# ========== FUNCTION CALLING FLOW ==========


def run_function_calling(client: OpenAI, user_request: str, temperature: float = 0.2, tool_choice: str = "auto") -> str:
    """
    The full function calling flow with multi-round support:
    1. Send request + tool definitions to model
    2. Model decides which tools to call (if any)
    3. Execute those tools
    4. Send results back to model
    5. Model synthesizes final answer OR requests more tool calls (loop)

    tool_choice options:
      "auto"     — model decides whether to call tools (default)
      "required" — model MUST call at least one tool
      "none"     — no tools allowed (pure text response)
    """
    print(f"\n{'=' * 60}")
    print(f"event=request_start user_request={user_request!r} tool_choice={tool_choice} temperature={temperature}")
    print(f"{'=' * 60}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to tools. "
                "When you need information, use the available tools. "
                "Always be concise and clear in your responses."
            ),
        },
        {"role": "user", "content": user_request},
    ]

    # Multi-round loop: model may need multiple rounds of tool calls
    # (e.g., "check weather, THEN decide whether to send notification")
    max_rounds = 5
    for round_num in range(1, max_rounds + 1):
        print(f"\n[event=tool_round_start round={round_num} max_rounds={max_rounds}]")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            messages=messages,
            tools=TOOLS,
            tool_choice=tool_choice,
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # If no tool calls, model is done — return the final answer
        if not tool_calls:
            print(f"\n[Round {round_num}] Model answered directly (no more tools needed)")
            final = response_message.content
            print(f"\n{'='*60}")
            print(f"FINAL ANSWER:\n{final}")
            print(f"{'='*60}\n")
            return final

        # Model wants to call tools
        print(f"[Round {round_num}] Model is calling {len(tool_calls)} tool(s):")
        for tc in tool_calls:
            print(f"   - {tc.function.name}({tc.function.arguments})")

        # Add assistant's response to conversation
        messages.append(response_message)

        # Execute each tool and add results
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            function_result = execute_tool(function_name, function_args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(function_result),
                }
            )

        # After first round, switch to auto so model can finish
        tool_choice = "auto"

    print(f"\nMax rounds ({max_rounds}) reached.")
    return "Max tool-calling rounds reached."


# ========== EXAMPLES / PRESETS ==========


def get_example_requests():
    """Preset requests to demonstrate different tool usage patterns."""
    return {
        "simple": "What's the weather in San Francisco?",
        "multi-tool": "Check the weather in Seattle and send a notification to user_123 saying 'Rain expected today'",
        "database": "Show me all notifications for user_123",
        "complex": "Get user_123's info from the database, check Seattle weather, and send them a high priority notification about the weather",
        "no-tool": "Explain what function calling is in one sentence",
    }


def main():
    """Parse args and run the selected function-calling demo."""
    parser = argparse.ArgumentParser(description="Day 3: Function Calling / Tool Use")
    parser.add_argument(
        "--request",
        help="Your request (the model will decide which tools to call)",
    )
    parser.add_argument(
        "--example",
        choices=["simple", "multi-tool", "database", "complex", "no-tool"],
        help="Use a preset example request",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="Model temperature (0.0-1.0)")
    parser.add_argument(
        "--tool-choice",
        choices=["auto", "required", "none"],
        default="auto",
        help="Tool choice: auto (model decides), required (must use tool), none (no tools)",
    )
    parser.add_argument("--list-examples", action="store_true", help="List all example requests and exit")

    args = parser.parse_args()

    examples = get_example_requests()

    # List examples if requested
    if args.list_examples:
        print("\nAvailable example requests:\n")
        for name, req in examples.items():
            print(f"  {name:12} -> {req}")
        print("\nUsage: python3 function_calling.py --example simple")
        print("\nTool choice options:")
        print("  auto     — model decides whether to use tools (default)")
        print("  required — model MUST call at least one tool")
        print("  none     — no tools allowed, pure text response")
        return

    # Determine the request
    if args.example:
        request = examples[args.example]
        print(f"\nUsing example: {args.example}")
    elif args.request:
        request = args.request
    else:
        # Default to simple example
        request = examples["simple"]
        print(f"\nNo request provided, using default: simple example")

    client = get_client()
    run_function_calling(client, request, args.temperature, args.tool_choice)


if __name__ == "__main__":
    main()
