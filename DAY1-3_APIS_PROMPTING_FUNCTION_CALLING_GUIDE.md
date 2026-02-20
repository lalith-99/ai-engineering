# Days 1–3: LLM APIs, Prompt Engineering & Function Calling

> Interview prep guide for SDE2/AI roles — Feb 2026

---

## Table of Contents
1. [Day 1 — LLM Fundamentals & API Basics](#day-1--llm-fundamentals--api-basics)
2. [Day 2 — Prompt Engineering & Structured Output](#day-2--prompt-engineering--structured-output)
3. [Day 3 — Function Calling / Tool Use](#day-3--function-calling--tool-use)
4. [Interview Cheat Sheet — Days 1–3 Buzzwords](#-interview-cheat-sheet--days-13-buzzwords)
5. [System Design: How APIs, Prompting & Tool Use Fit Together](#-system-design-how-apis-prompting--tool-use-fit-together)
6. [Glossary](#glossary)

---

# Day 1 — LLM Fundamentals & API Basics

## 1. What Is an LLM? (Plain English)

A **Large Language Model (LLM)** is a massive neural network trained on trillions of words of text. It learns patterns of language so well that it can generate human-like text, answer questions, write code, translate languages, and reason about problems.

### Interview Buzzword: "Transformer Architecture"
LLMs use the **Transformer architecture** (Vaswani et al., 2017) with **self-attention** to process sequences in parallel. This is why you can scale to billions of parameters.

### The Key Intuition

```
Input:  "The capital of France is ___"
                                    ↓
         LLM predicts next token: "Paris"  (highest probability)
                                    ↓
         Then predicts next: "."
         Then predicts next: " It"
         Then predicts next: " is"
         ...and so on, one token at a time (autoregressive generation)
```

### What Makes LLMs "Large"?

| Model | Parameters | Training Data | Key Innovation |
|-------|-----------|---------------|---------------|
| GPT-3 (2020) | 175B | 300B tokens | Showed scale = capability |
| GPT-4 (2023) | ~1.7T (rumored MoE) | ~13T tokens | Multimodal (text + images) |
| GPT-5.2 (2025) | Undisclosed | Undisclosed | Frontier model, best reasoning |
| Llama 3.1 (Meta) | 8B / 70B / 405B | 15T tokens | Best open-source |
| Claude 4.6 (Anthropic) | Undisclosed | Undisclosed | Extended thinking, safety-first |
| Gemini 3.1 (Google) | Undisclosed | Undisclosed | Multimodal, function calling |

### Resume Keyword
"Large Language Model (LLM) integration & API development"

---

## 2. Tokenization — How LLMs Read Text

LLMs don't process text character by character. They break text into **tokens** — subword pieces:

```
"Hello, how are you?"
    ↓ tokenized (GPT-4 style)
["Hello", ",", " how", " are", " you", "?"]
    = 6 tokens

"Unbelievable" → ["Un", "believ", "able"] = 3 tokens

Rule of thumb: 1 token ≈ ¾ of a word
               100 tokens ≈ 75 words
               1,000 tokens ≈ 750 words
```

### Why Tokenization Matters for Engineers

1. **Billing** — You pay per token (input + output)
2. **Context window** — Max tokens the model can process at once
3. **Prompt design** — Longer prompts = more tokens = more cost
4. **tiktoken** — OpenAI's tokenizer library (use it for cost estimation!)

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("Explain distributed caching in 3 bullets")
print(f"Token count: {len(tokens)}")  # → 7 tokens
print(f"Estimated cost (GPT-4.1 nano input): ${len(tokens) / 1_000_000 * 0.10:.6f}")
```

### Interview Buzzword: "Tokenization"
Tokenization splits text into subword units (Byte Pair Encoding). Token counts matter for cost and context limits.

---

## 3. The LLM API Landscape (Feb 2026)

### Major Providers — Current State

| Provider | Models | API Style | Key Differentiator |
|----------|--------|-----------|-------------------|
| **OpenAI** | GPT-5.2, GPT-5, GPT-4.1, o3, o4-mini | Responses API (new) + Chat Completions | Market leader, broadest model range |
| **Anthropic** | Claude Opus 4.6, Sonnet 4.6, Haiku 4.5 | Messages API | Safety-first, extended thinking |
| **Google** | Gemini 3.1 Pro, Gemini 3 Flash, 2.5 Flash | GenerateContent API | Multimodal, massive context (2M tokens) |
| **Meta** | Llama 3.1, Llama 3.2, Llama 4 | Open-source (self-host or via providers) | Free, customizable |
| **Mistral** | Mistral Large, Codestral | Chat Completions (OpenAI-compatible) | European, strong coding |
| **Cohere** | Command R+ | Chat API | Enterprise, retrieval-focused |

### Current Pricing — Quick Reference (Feb 2026)

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Best For |
|-------|----------------------|------------------------|---------|
| **GPT-5.2** | $1.75 | $14.00 | Frontier, complex reasoning |
| **GPT-5** | $1.25 | $10.00 | Flagship, general purpose |
| **GPT-5 mini** | $0.25 | $2.00 | Cost-effective, well-defined tasks |
| **GPT-5 nano** | $0.05 | $0.40 | Ultra-cheap, simple tasks |
| **GPT-4.1** | $2.00 | $8.00 | Great for coding |
| **GPT-4.1 mini** | $0.40 | $1.60 | Budget workhorse |
| **GPT-4.1 nano** | $0.10 | $0.40 | High-volume, simple |
| **o3** | $2.00 | $8.00 | Deep reasoning (chain-of-thought) |
| **o4-mini** | $1.10 | $4.40 | Budget reasoning |
| **gpt-4o-mini** *(prev gen)* | $0.15 | $0.60 | Legacy, still widely used |

### Interview Talking Point: "Model Selection"
Start with the cheapest option that works. GPT-4.1 nano ($0.10/M) for classification, GPT-5 mini ($0.25/M) for general tasks, GPT-5.2 ($1.75/M) only for reasoning. Model choice is your biggest cost lever.

---

## 4. OpenAI API Architecture — Two Eras

### Era 1: Chat Completions API (2023–present, still supported)
This is what your Day 1-2 code uses. Still works, still widely used.

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=300,
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain API rate limits in 3 bullets."},
    ],
)
print(response.choices[0].message.content)
```

### Era 2: Responses API (2025–present, the future)
The new primary API. Simpler, supports built-in tools, replaces Assistants API.

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-5",
    input="Explain API rate limits in 3 bullets.",
    instructions="You are a concise assistant.",
)
print(response.output_text)
```

### Chat Completions vs Responses API — What Changed?

| Feature | Chat Completions | Responses API |
|---------|-----------------|--------------|
| **Messages** | `messages=[{role, content}]` | `input=` (string or list) |
| **System prompt** | `{"role": "system", ...}` | `instructions=` parameter |
| **Response** | `response.choices[0].message.content` | `response.output_text` |
| **Tools** | `tools=[{type, function}]` | `tools=[{type, name, ...}]` |
| **Built-in tools** | ❌ None | ✅ Web search, file search, computer use |
| **Streaming** | SSE events | SSE events (similar) |
| **Status** | Stable, not deprecated | New primary API |

### Resume Keyword: "OpenAI API integration (Chat Completions & Responses API)"

### Interview Tip
I'm familiar with both Chat Completions and the newer Responses API. For new projects, I'd use Responses API because it has built-in tool support and is simpler. For existing codebases, Chat Completions is still fully supported.

---

## 5. Authentication & Rate Limits

### API Key Setup
```bash
export OPENAI_API_KEY="sk-..."     # OpenAI
export ANTHROPIC_API_KEY="sk-..."  # Anthropic
export GOOGLE_API_KEY="..."        # Google AI
```

### Rate Limits — What Interviewers Ask

```
Rate limits are applied per:
  - Requests per minute (RPM)
  - Tokens per minute (TPM)
  - Requests per day (RPD)

Free tier:    ~3 RPM,  ~40,000 TPM
Tier 1:      ~500 RPM, ~200,000 TPM
Tier 5:      ~10,000 RPM, ~30,000,000 TPM

When you hit a limit → HTTP 429 (Too Many Requests)
Solution: Exponential backoff + retry logic
```

### Interview Buzzword: "Rate Limiting & Exponential Backoff"
I implement exponential backoff with jitter for 429 errors — start at 1s, double each retry up to 60s, with random jitter to prevent thundering herd.

---

## 6. Key API Parameters (Your Day 1-2 Code Covers These)

| Parameter | What It Does | Range | Notes |
|-----------|-------------|-------|----------------------|
| **`model`** | Which LLM to use | String | "Model selection is the #1 cost lever" |
| **`temperature`** | Randomness/creativity | 0.0–2.0 | "0.0 for deterministic, 0.7 for creative" |
| **`max_tokens`** | Output length cap | Integer | "Guards against runaway generation costs" |
| **`top_p`** | Nucleus sampling | 0.0–1.0 | "Alternative to temperature, don't change both" |
| **`response_format`** | Force JSON output | Object | "Use `json_object` for structured extraction" |
| **`stream`** | Real-time output | Boolean | "Essential for chatbot UX — time to first token" |
| **`seed`** | Reproducibility | Integer | "Makes outputs deterministic for testing" |

### Temperature — The Most Asked Parameter

```
temperature = 0.0  →  Deterministic (same input = same output)
                       Use for: classification, extraction, code generation

temperature = 0.3  →  Slightly creative but mostly focused
                       Use for: summaries, Q&A, technical writing

temperature = 0.7  →  Creative and varied
                       Use for: brainstorming, marketing copy, stories

temperature = 1.0+ →  Very random/creative
                       Use for: experimental, rarely in production
```

### Interview Answer: "How do you ensure consistent LLM outputs?"
Three techniques: (1) Set temperature to 0.0 for deterministic output, (2) use a fixed `seed` for reproducibility across calls, (3) use structured outputs (JSON mode or schema-based) to constrain the format.

---

# Day 2 — Prompt Engineering & Structured Output

## 1. Prompt Engineering — The Most In-Demand AI Skill

Prompt engineering is **the art and science of crafting inputs** that get the desired output from an LLM. It's mentioned in 80%+ of AI/GenAI job postings.

### Interview Buzzword: "Prompt Engineering"
Prompt engineering is designing effective inputs for LLMs through techniques like system prompts, few-shot examples, chain-of-thought reasoning, and output constraints.

---

## 2. The Message Architecture — Roles

Every Chat Completions/Messages API call uses a **role-based message system**:

```python
messages = [
    # SYSTEM: Sets persona, rules, constraints (the "constitution")
    {"role": "system", "content": "You are a senior backend engineer. Be concise."},
    
    # USER: The human's input/question
    {"role": "user", "content": "Explain rate limiting."},
    
    # ASSISTANT: Previous AI responses (for multi-turn conversations)
    {"role": "assistant", "content": "Rate limiting controls request frequency..."},
    
    # USER: Follow-up
    {"role": "user", "content": "Show me a Python implementation."},
]
```

### System Prompt — The Most Powerful Lever

```
A good system prompt includes:
  1. ROLE:        "You are a senior backend engineer"
  2. CONSTRAINTS: "Be concise, use numbered steps"
  3. FORMAT:      "Return exactly 5 bullets, each under 15 words"
  4. GUARDRAILS:  "If unsure, say 'I don't know'"
  5. AUDIENCE:    "Audience: junior SDE"
```

### Cross-Provider Comparison

| Concept | OpenAI (Chat) | OpenAI (Responses) | Anthropic (Claude) | Google (Gemini) |
|---------|--------------|-------------------|-------------------|-----------------|
| System prompt | `role: "system"` | `instructions=` | `system=` parameter | `system_instruction=` |
| User input | `role: "user"` | `input=` | `role: "user"` | `role: "user"` |
| AI response | `role: "assistant"` | In `output` | `role: "assistant"` | `role: "model"` |

### Resume Keyword: "Prompt engineering for production LLM systems"

---

## 3. Prompting Techniques (Know All of These!)

### Technique 1: Zero-Shot Prompting
No examples — just ask directly.

```python
# Zero-shot: direct question
messages = [
    {"role": "user", "content": "Classify this review as positive or negative: 'This product is terrible'"}
]
```

### Technique 2: Few-Shot Prompting (Your Code Does This!)
Provide examples so the model learns the pattern.

```python
# Few-shot: learn from examples (from your call_few_shot function)
messages = [
    {"role": "system", "content": "Answer in short bullet points, like this example:"},
    {"role": "user", "content": "Q: What is a database index?"},
    {"role": "assistant", "content": (
        "- Data structure that speeds up queries.\n"
        "- Trades storage for query performance.\n"  
        "- Slows writes slightly but accelerates reads."
    )},
    {"role": "user", "content": f"Now answer this the same way:\nQ: {question}\nA:"},
]
```

### Interview Buzzword: "Few-Shot Learning / In-Context Learning"
Few-shot prompting provides examples in the prompt so the model learns the desired pattern without any fine-tuning. It leverages the model's in-context learning capability.

### Technique 3: Chain-of-Thought (CoT) Prompting
Ask the model to **think step by step** before answering.

```python
# Chain-of-thought (from your code!)
messages = [
    {"role": "system", "content": "Think step-by-step, then give a final answer."},
    {"role": "user", "content": (
        "Question: How many r's are in 'strawberry'?\n\n"
        "Please think through this carefully:\n"
        "1. Break down the problem.\n"
        "2. Identify each letter.\n"
        "3. Count the r's.\n"
        "Then provide a clear final answer."
    )},
]
```

### Interview Buzzword: "Chain-of-Thought (CoT)"
CoT prompting dramatically improves reasoning accuracy by asking the model to 'think step by step.' It's the foundation for reasoning models like o3 and o4-mini, which automate this internally.

### Technique 4: Self-Consistency
Run the same prompt multiple times, take the majority vote.

```
Prompt: "Is this code vulnerable to SQL injection?"

Run 1: "Yes"  ✓
Run 2: "Yes"  ✓
Run 3: "No"   ✗
Run 4: "Yes"  ✓
Run 5: "Yes"  ✓

Majority vote → "Yes" (4/5)  ← much more reliable than a single run!
```

### Technique 5: Prompt Chaining
Break complex tasks into sequential LLM calls.

```
Step 1: LLM → "Extract key entities from this document"
          ↓
Step 2: LLM → "For each entity, find related information in our database"
          ↓
Step 3: LLM → "Generate a summary report based on the enriched data"
```

### All Techniques — Comparison Table

| Technique | When to Use | Cost | Accuracy | Your Code |
|-----------|-------------|------|----------|-----------|
| **Zero-Shot** | Simple tasks, well-known formats | 💲 Low | 🟡 Medium | `call_basic()` |
| **Few-Shot** | Need specific output format | 💲💲 Medium | 🟢 High | `call_few_shot()` |
| **Chain-of-Thought** | Math, reasoning, logic | 💲💲 Medium | 🟢 High | `call_chain_of_thought()` |
| **Self-Consistency** | High-stakes decisions | 💲💲💲 High | 🟢 Very High | — |
| **Prompt Chaining** | Complex multi-step tasks | 💲💲 Medium | 🟢 High | — |
| **JSON Mode** | Structured data extraction | 💲 Low | 🟢 High | `call_json()` |

---

## 4. Structured Output — Beyond Free-Form Text

In production, you need the LLM to return **structured data**, not prose.

### Method 1: JSON Mode (Your Code Does This!)

```python
# From your call_json() function
response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.2,
    response_format={"type": "json_object"},  # ← Force JSON output
    messages=[
        {"role": "system", "content": "Return a valid JSON object..."},
        {"role": "user", "content": f"Summarize: {topic}"},
    ],
)
data = json.loads(response.choices[0].message.content)
```

### Method 2: Structured Outputs with JSON Schema (OpenAI, Feb 2026)

```python
from pydantic import BaseModel

class WeatherInfo(BaseModel):
    city: str
    temperature: float
    unit: str
    condition: str

response = client.responses.create(
    model="gpt-4.1",
    input="What's the weather in London?",
    text={"format": {
        "type": "json_schema",
        "schema": WeatherInfo.model_json_schema()
    }}
)
# GUARANTEED to match your schema — no parsing errors!
```

### Method 3: Instructor Library (Any Provider)

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(OpenAI())

class UserInfo(BaseModel):
    name: str
    age: int
    email: str

user = client.chat.completions.create(
    model="gpt-4.1",
    response_model=UserInfo,
    messages=[{"role": "user", "content": "John is 30, email: john@ex.com"}]
)
print(user.name)   # "John" — typed, validated Python object!
```

### Structured Output Methods — Comparison

| Method | Reliability | Provider Support | Notes |
|--------|-----------|-----------------|----------------------|
| **JSON Mode** | 🟡 Medium (can fail) | OpenAI, Anthropic | "Basic but can produce invalid JSON" |
| **Structured Outputs** | 🟢 Guaranteed | OpenAI (Responses API) | "Schema-enforced, zero parse errors" |
| **Instructor + Pydantic** | 🟢 Very High | Any LLM provider | "Provider-agnostic, auto-retry on failure" |
| **Function Calling** | 🟢 High | OpenAI, Anthropic, Google | "Implicit structured output through tool schemas" |
| **Grammar-based** (CFG) | 🟢 Guaranteed | OpenAI (custom tools), Ollama | "Constrained decoding at token level" |

### Interview Buzzword: "Structured Outputs"
I use Pydantic models with Instructor for provider-agnostic structured outputs. For OpenAI-only projects, Structured Outputs with JSON Schema guarantees schema compliance at the decoding level.

### Resume Keyword: "Pydantic-based structured output pipelines"

---

## 5. Prompt Engineering Best Practices (The Checklist)

```
✅ Be specific and explicit — don't assume the model knows context
✅ Set a persona in the system prompt — "You are a senior SRE..."
✅ Specify output format — "Return a JSON object with keys: ..."
✅ Use constraints — "Maximum 5 bullets, each under 15 words"
✅ Provide examples — few-shot for consistent formatting
✅ Use delimiters — triple backticks, XML tags, or markdown headers
✅ Ask for reasoning — "Think step by step before answering"
✅ Version control prompts — treat prompts like code (use Langfuse)
✅ Test with edge cases — empty inputs, adversarial inputs, long inputs
✅ Iterate based on evaluation — don't "vibe check," use metrics
```

### The Prompt Engineering Stack (Production Systems)

```
┌─────────────────────────────────┐
│  Prompt Management (Langfuse)   │  ← Version control, A/B testing
├─────────────────────────────────┤
│  Prompt Templates (Jinja2)      │  ← Dynamic variables, conditionals
├─────────────────────────────────┤
│  Few-Shot Selection (Dynamic)   │  ← Retrieve relevant examples from DB
├─────────────────────────────────┤
│  Output Validation (Pydantic)   │  ← Structured output enforcement
├─────────────────────────────────┤
│  Evaluation Loop (DeepEval)     │  ← Measure quality, A/B test variants
└─────────────────────────────────┘
```

---

## 6. Multi-Provider Strategy — Real-World Pattern

Production AI systems rarely use just one provider:

```
┌─────────────────────────────────────────────────────────┐
│                     AI GATEWAY                           │
│  (routes requests to cheapest capable model)             │
│                                                          │
│  Simple classification → GPT-4.1 nano ($0.10/M)         │
│  General Q&A          → GPT-5 mini ($0.25/M)            │
│  Complex reasoning    → Claude Opus 4.6 / GPT-5.2       │
│  Code generation      → GPT-4.1 ($2/M)                  │
│  Fallback             → Gemini 3 Flash (generous free)   │
│                                                          │
│  Rate limited? → failover to next provider               │
│  Outage?       → automatic failover                      │
└─────────────────────────────────────────────────────────┘
```

### Resume Keyword: "Multi-model routing & AI gateway architecture"
### Interview Tip: "I'd design an AI gateway that routes requests based on task complexity — simple extraction goes to a nano model, complex reasoning to a frontier model. This can reduce costs by 10-50x."

---

# Day 3 — Function Calling / Tool Use

## 1. What Is Function Calling? (The Bridge Between LLMs and the Real World)

LLMs are powerful but they're **trapped inside text**. They can't:
- Check the weather
- Query a database
- Send an email
- Search the web
- Calculate exact math

**Function calling** (a.k.a. **tool use**) lets the model request that YOUR code execute external functions, then uses the results to generate a response.

### Interview Buzzword: "Function Calling / Tool Use"
Function calling lets LLMs invoke external APIs and tools. The model decides WHICH tool to call and WHAT arguments to pass, but the actual execution happens in YOUR application code.

---

## 2. The 5-Step Flow (Know This Cold!)

This is exactly what your `run_function_calling()` implements:

```
┌────────────────────────────────────────────────────────────┐
│              THE FUNCTION CALLING FLOW                      │
│                                                             │
│  Step 1: Send request WITH tool definitions                 │
│  ┌──────────────────────────────────────────────────┐      │
│  │ client.chat.completions.create(                   │      │
│  │   model="gpt-4o-mini",                            │      │
│  │   messages=[user_message],                        │      │
│  │   tools=TOOLS,              ← JSON schemas        │      │
│  │   tool_choice="auto"        ← model decides       │      │
│  │ )                                                  │      │
│  └──────────────────────────────────────────────────┘      │
│                          ↓                                  │
│  Step 2: Model returns tool_calls (NOT the answer!)         │
│  ┌──────────────────────────────────────────────────┐      │
│  │ response.choices[0].message.tool_calls = [        │      │
│  │   {name: "get_weather", args: {city: "Seattle"}}  │      │
│  │ ]                                                  │      │
│  └──────────────────────────────────────────────────┘      │
│                          ↓                                  │
│  Step 3: YOUR CODE executes the function                    │
│  ┌──────────────────────────────────────────────────┐      │
│  │ result = get_weather(city="Seattle")               │      │
│  │ # → {"temp": 55, "condition": "rainy"}             │      │
│  └──────────────────────────────────────────────────┘      │
│                          ↓                                  │
│  Step 4: Send result back to model                          │
│  ┌──────────────────────────────────────────────────┐      │
│  │ messages.append({                                  │      │
│  │   role: "tool",                                    │      │
│  │   tool_call_id: "call_abc",                        │      │
│  │   content: '{"temp": 55, "condition": "rainy"}'    │      │
│  │ })                                                  │      │
│  └──────────────────────────────────────────────────┘      │
│                          ↓                                  │
│  Step 5: Model generates final human-readable answer        │
│  ┌──────────────────────────────────────────────────┐      │
│  │ "It's currently 55°F and rainy in Seattle. You     │      │
│  │  might want to bring an umbrella! ☂️"              │      │
│  └──────────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────┘
```

### Interview Answer: "How does function calling work?"
It's a 5-step conversation: (1) I send the user's request along with JSON schemas describing available tools, (2) the model decides which tools to call and generates structured arguments, (3) my application code executes the actual functions, (4) I send the results back to the model, (5) the model synthesizes a natural language response incorporating the tool results.

---

## 3. Tool Definitions — JSON Schema (Your Code)

```python
# From your function_calling.py — the TOOLS array
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": "Send a notification to a user",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID to notify"
                    },
                    "message": {
                        "type": "string",
                        "description": "The notification message"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Priority level"
                    },
                },
                "required": ["user_id", "message"],
            },
        },
    },
    # ... more tools
]
```

### Best Practices for Tool Definitions (Interview Material!)

| Practice | Why | Example |
|----------|-----|---------|
| **Descriptive names** | Model chooses tool by name | `get_weather` not `func1` |
| **Detailed descriptions** | Model knows when to use it | "Gets current temp in Fahrenheit" |
| **Use enums** | Prevents invalid values | `"enum": ["low", "medium", "high"]` |
| **Mark required fields** | Prevents missing args | `"required": ["user_id"]` |
| **Use strict mode** | Guarantees schema compliance | `"strict": true` |
| **Keep tools < 20** | Too many = wrong choices | Evaluate with different counts |

---

## 4. The Tool Router Pattern (Your Code!)

Your `execute_tool()` function implements the **router pattern** — mapping tool names to actual function implementations:

```python
# From your function_calling.py
def execute_tool(name: str, args: dict) -> dict:
    """Route tool calls to actual function implementations."""
    if name == "send_notification":
        return send_notification(**args)
    elif name == "get_weather":
        return get_weather(**args)
    elif name == "query_database":
        return query_database(**args)
    else:
        return {"error": f"Unknown tool: {name}"}
```

### Interview Tip: "In production, I'd use a registry pattern instead of if/elif chains — a dictionary mapping tool names to functions, making it easy to add/remove tools dynamically."

```python
# Production pattern: tool registry
TOOL_REGISTRY = {
    "send_notification": send_notification,
    "get_weather": get_weather,
    "query_database": query_database,
}

def execute_tool(name: str, args: dict) -> dict:
    func = TOOL_REGISTRY.get(name)
    if not func:
        return {"error": f"Unknown tool: {name}"}
    return func(**args)
```

---

## 5. Cross-Provider Function Calling (Feb 2026)

### OpenAI — Chat Completions (Your Code)
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=TOOLS,
    tool_choice="auto",
)
# Access: response.choices[0].message.tool_calls
```

### OpenAI — Responses API (New!)
```python
response = client.responses.create(
    model="gpt-5",
    input=input_list,
    tools=[{
        "type": "function",
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": { ... },
        "strict": True  # ← Guaranteed schema compliance
    }],
)
# Access: response.output → function_call items
# Return results: {"type": "function_call_output", "call_id": ..., "output": ...}
```

### Anthropic — Claude Tool Use
```python
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    tools=[{
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {  # ← note: input_schema, not parameters
            "type": "object",
            "properties": { ... },
            "required": ["location"]
        }
    }],
    messages=[{"role": "user", "content": "Weather in London?"}],
)
# Access: response.content → tool_use blocks
# Return results: {"type": "tool_result", "tool_use_id": ..., "content": ...}
```

### Google — Gemini Function Calling
```python
from google import genai
from google.genai import types

client = genai.Client()
tools = types.Tool(function_declarations=[{
    "name": "get_weather",
    "description": "Get weather for a location",
    "parameters": {
        "type": "object",
        "properties": { ... },
        "required": ["location"]
    }
}])

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Weather in London?",
    config=types.GenerateContentConfig(tools=[tools]),
)
# Access: response.candidates[0].content.parts[0].function_call
```

### Cross-Provider Comparison — Function Calling

| Feature | OpenAI (Chat) | OpenAI (Responses) | Anthropic | Google Gemini |
|---------|--------------|-------------------|-----------|--------------|
| **Schema key** | `parameters` | `parameters` | `input_schema` | `parameters` |
| **Tool calls in response** | `tool_calls[]` | `output[]` (function_call) | `content[]` (tool_use) | `parts[]` (function_call) |
| **Return results** | `role: "tool"` | `function_call_output` | `tool_result` | `functionResponse` |
| **Parallel calls** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Strict mode** | ✅ Yes | ✅ Yes | ✅ Yes (strict: true) | ✅ VALIDATED mode |
| **Auto execution** | ❌ Manual | ❌ Manual | ❌ Manual | ✅ Python SDK |
| **Built-in tools** | ❌ None | ✅ Web search, files | ✅ Web search, computer | ✅ Google Search, code exec |
| **MCP support** | ✅ Via Agents SDK | ✅ Via Agents SDK | ✅ Via MCP connector | ✅ Built into SDK |

### Resume Keyword: "Multi-provider tool use / function calling (OpenAI, Anthropic, Gemini)"

---

## 6. Advanced Tool Use Patterns

### Pattern 1: Parallel Function Calling
Model calls multiple tools simultaneously:

```python
# User: "Check weather in London AND send a notification to user_123"
# Model returns TWO tool_calls in one response:
tool_calls = [
    {"name": "get_weather", "args": {"city": "London"}},
    {"name": "send_notification", "args": {"user_id": "123", "message": "..."}},
]
# Execute both, return both results, then model synthesizes final answer
```

### Pattern 2: Sequential/Compositional Function Calling
Model chains tools — output of one feeds into the next:

```
User: "If it's cold in London, send John a coat reminder"

Round 1: Model calls get_weather(city="London") → 5°C
Round 2: Model calls send_notification(user="john", message="Bring a coat, 5°C!") → success
Final:   "I checked London's weather (5°C) and sent John a coat reminder!"
```

### Pattern 3: Tool Choice Control
```python
tool_choice = "auto"       # Model decides (default)
tool_choice = "required"   # Must call at least one tool
tool_choice = "none"       # No tools allowed (pure text)
tool_choice = {"type": "function", "name": "get_weather"}  # Force specific tool
```

### Pattern 4: Custom Tools with Context-Free Grammars (New in 2025!)
OpenAI now supports **custom tools** where the model returns free-form text constrained by a grammar (Lark or regex):

```python
response = client.responses.create(
    model="gpt-5",
    input="Create a math expression for four plus four",
    tools=[{
        "type": "custom",
        "name": "math_exp",
        "description": "Creates valid math expressions",
        "format": {
            "type": "grammar",
            "syntax": "lark",
            "definition": "start: expr\nexpr: NUMBER (SP OP SP NUMBER)*\n..."
        }
    }]
)
# Output: "4 + 4" (guaranteed to match grammar!)
```

### Interview Buzzword: "Constrained Decoding"
Custom tools with grammar constraints enable constrained decoding — the model's output is guaranteed to match a formal grammar (Lark CFG or regex). This is useful for generating structured formats like math expressions, SQL queries, or domain-specific languages.

---

## 7. Function Calling → AI Agents (The Evolution)

Function calling is the **building block** for AI agents:

```
Level 0: Simple LLM Call
  → input → LLM → text output

Level 1: Function Calling (Day 3 — You Are Here!)
  → input → LLM → tool call → execute → LLM → output

Level 2: ReAct Agent (Day 7)
  → input → [reason → act → observe]* → output (looping!)

Level 3: Multi-Agent (Day 9)
  → input → Router → [Agent A, Agent B, Agent C] → output

Level 4: Autonomous Agent
  → goal → agent runs indefinitely with tools, memory, sub-agents
```

### Interview Talking Point: "From Function Calling to Agents"
Function calling is the primitive that makes AI agents possible. Day 3 teaches the single tool-call flow; agents (Days 7-9) add the reasoning loop — the model decides which tools to call, observes results, and iterates until the task is complete.

---

# Interview Cheat Sheet — Days 1–3 Buzzwords

## Resume Section: "API & Prompting Skills"

### For SDE2 / AI Engineer
```
LLM APIs: OpenAI (Chat Completions, Responses API), Anthropic Claude,
           Google Gemini, Multi-Provider Gateway Architecture

Prompt Engineering: System Prompts, Few-Shot Learning, Chain-of-Thought,
                    JSON Mode, Structured Outputs, Pydantic Validation

Tool Use: Function Calling (OpenAI/Anthropic/Gemini), Tool Orchestration,
          Parallel & Sequential Tool Calls, Strict Mode, Custom Tools (CFG)

Production: Rate Limiting, Exponential Backoff, Token Budgeting,
            Model Selection & Cost Optimization, Prompt Versioning
```

---

## Top 20 Interview Buzzwords — Days 1–3

### Tier 1 — Asked in Every AI Interview
| # | Buzzword | One-Line Definition |
|---|----------|-------------------|
| 1 | **LLM** | Large Language Model — GPT, Claude, Gemini, Llama |
| 2 | **Prompt Engineering** | Crafting inputs for desired LLM outputs |
| 3 | **Function Calling / Tool Use** | LLM invokes external APIs and tools |
| 4 | **Temperature** | Controls randomness (0=deterministic, 1=creative) |
| 5 | **Tokenization** | Splitting text into subword tokens (BPE) |
| 6 | **Context Window** | Max tokens a model can process (128K–2M) |
| 7 | **System Prompt** | Instructions that set the model's persona & rules |
| 8 | **JSON Mode / Structured Output** | Forcing LLM to return structured data |

### Tier 2 — Asked in 50%+ of Interviews
| # | Buzzword | One-Line Definition |
|---|----------|-------------------|
| 9 | **Few-Shot Learning** | Providing examples in-prompt for pattern learning |
| 10 | **Chain-of-Thought (CoT)** | "Think step by step" for better reasoning |
| 11 | **Responses API** | OpenAI's new primary API (replaces Assistants) |
| 12 | **Rate Limiting** | API caps on requests/tokens per minute |
| 13 | **Exponential Backoff** | Retry strategy — wait 1s, 2s, 4s, 8s... |
| 14 | **Streaming** | Real-time token-by-token output (SSE) |
| 15 | **Transformer** | Neural network architecture behind all LLMs |

### Tier 3 — Differentiators
| # | Buzzword | One-Line Definition |
|---|----------|-------------------|
| 16 | **Reasoning Models** | o3, o4-mini — models that "think" before answering |
| 17 | **Constrained Decoding** | Grammar-based output constraints (Lark, regex) |
| 18 | **Model Distillation** | Train small model on big model's outputs |
| 19 | **Prompt Caching** | Cache repeated prompt prefixes for cost savings |
| 20 | **AI Gateway** | Proxy that routes, caches, rate-limits LLM calls |

---

# System Design: How APIs, Prompting & Tool Use Fit Together

## Architecture: "Design a Production LLM-Powered Service"

```
┌──────────────────────────────────────────────────────────────────┐
│                    PRODUCTION LLM SERVICE                         │
│                                                                   │
│  CLIENT (React/Mobile)                                            │
│       ↓                                                           │
│  API GATEWAY (FastAPI / Go)                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Rate Limiter  │  Auth  │  Request Validator               │  │
│  └────────────────────────────────────────────────────────────┘  │
│       ↓                                                           │
│  PROMPT LAYER (Day 2)                                             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Prompt Template (Jinja2)                                  │  │
│  │  + System Prompt (versioned in Langfuse)                   │  │
│  │  + Few-Shot Examples (dynamically selected)                │  │
│  │  + User Input (sanitized, validated)                       │  │
│  └────────────────────────────────────────────────────────────┘  │
│       ↓                                                           │
│  MODEL ROUTER (Day 1)                                             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Simple query?  → GPT-4.1 nano ($0.10/M)                  │  │
│  │  Medium query?  → GPT-5 mini ($0.25/M)                    │  │
│  │  Complex query? → GPT-5.2 ($1.75/M)                       │  │
│  │  Fallback?      → Claude Sonnet / Gemini Flash             │  │
│  └────────────────────────────────────────────────────────────┘  │
│       ↓                                                           │
│  TOOL EXECUTION (Day 3)                                           │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Tool Registry → execute_tool(name, args)                  │  │
│  │  ├── Database queries (PostgreSQL)                         │  │
│  │  ├── External APIs (weather, email, CRM)                   │  │
│  │  ├── Internal services (notification, search)              │  │
│  │  └── Return results → model → final answer                │  │
│  └────────────────────────────────────────────────────────────┘  │
│       ↓                                                           │
│  OBSERVABILITY (Day 6)                                            │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Langfuse: traces, cost, latency, token counts             │  │
│  │  Budget guard: max tokens, max cost per request            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

# Glossary

| Term | Plain English Definition |
|------|------------------------|
| **API Key** | Secret token for authenticating with an LLM provider |
| **Autoregressive** | Generating one token at a time, left to right |
| **BPE** | Byte Pair Encoding — the tokenization algorithm used by GPT |
| **Chain-of-Thought** | Prompting the model to reason step-by-step |
| **Chat Completions** | OpenAI's original API endpoint for conversations |
| **Completion tokens** | Tokens generated by the model (output) |
| **Constrained Decoding** | Forcing model output to match a grammar/schema |
| **Context window** | Maximum tokens an LLM can process in one call |
| **Custom tools** | OpenAI tools that accept free-form text with grammar constraints |
| **Exponential backoff** | Doubling wait time between retries (1s, 2s, 4s, 8s) |
| **Few-shot** | Providing examples in the prompt |
| **Function calling** | LLM requests execution of external functions |
| **Instruction tuning** | Training models to follow instructions (not just predict text) |
| **JSON Mode** | Forcing LLM to return valid JSON |
| **LLM** | Large Language Model |
| **Max tokens** | Upper limit on generated output length |
| **Messages** | The role-based conversation format (system/user/assistant) |
| **Prompt tokens** | Tokens in your input (system + user messages) |
| **Rate limiting** | API caps on requests/tokens per time period |
| **Responses API** | OpenAI's new primary API (replaces Assistants, simpler than Chat) |
| **Role** | system, user, assistant, or tool — message categorization |
| **Self-consistency** | Running multiple times and taking majority vote |
| **Streaming** | Receiving response tokens in real-time (SSE) |
| **Strict mode** | Guaranteed JSON schema compliance for tool calls |
| **Structured output** | Forcing LLM to produce data in a specific format |
| **System prompt** | Instructions that set the model's behavior and constraints |
| **Temperature** | Controls randomness of output (0.0–2.0) |
| **tiktoken** | OpenAI's tokenizer library for counting tokens |
| **Token** | Subword piece (~¾ of a word) — the unit LLMs process |
| **Tool choice** | Control whether/which tools the model can use |
| **Tool registry** | Dictionary mapping tool names to function implementations |
| **Tool use** | Synonym for function calling (Anthropic's preferred term) |
| **Transformer** | The neural network architecture behind all modern LLMs |
| **Zero-shot** | No examples — just ask the model directly |

---

## 🗺️ How Days 1–3 Fit Into the Full Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR AI ENGINEERING JOURNEY                │
│                                                              │
│  Day 1: API BASICS                                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Connect to LLM APIs → send messages → get responses  │    │
│  │ Understand models, pricing, authentication            │    │
│  │ "I can talk to an LLM programmatically"               │    │
│  └─────────────────────────────────────────────────────┘    │
│           ↓                                                  │
│  Day 2: PROMPT ENGINEERING                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ System prompts → few-shot → CoT → JSON mode          │    │
│  │ Structure and constrain LLM outputs                   │    │
│  │ "I can get consistent, useful outputs from LLMs"      │    │
│  └─────────────────────────────────────────────────────┘    │
│           ↓                                                  │
│  Day 3: FUNCTION CALLING                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Tool definitions → model decides → execute → respond  │    │
│  │ LLMs can now interact with the real world             │    │
│  │ "I can make LLMs use tools and external APIs"         │    │
│  └─────────────────────────────────────────────────────┘    │
│           ↓                                                  │
│  Days 4-6: EMBEDDINGS → RAG → OBSERVABILITY                │
│  Days 7-12: AGENTS → MCP → MULTI-AGENT → EVAL → PROD      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

> 💡 **Pro Tip:** Your Day 1-3 code covers the **core primitives** of ALL AI engineering. Every agent, every RAG pipeline, every production AI system is built on top of (1) API calls, (2) well-crafted prompts, and (3) tool use. Master these and everything else follows.

> **Next Steps:** Run your existing code (`call_openai.py`, `function_calling.py`) to get hands-on experience. Then try modifying the prompts, switching models, adding new tools. Build muscle memory with these fundamentals.

---

*Generated with current market research as of February 2026. Pricing and model availability evolve frequently — check provider docs for latest.*
