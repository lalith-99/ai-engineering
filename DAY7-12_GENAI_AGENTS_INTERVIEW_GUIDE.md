# Days 7–12: Gen AI, AI Agents, MCP & Tech Stack

> Interview prep guide for SDE2/AI roles — Feb 2026

---

## Table of Contents
1. [Day 7 — AI Agents & Agentic Architectures](#day-7--ai-agents--agentic-architectures)
2. [Day 8 — Model Context Protocol (MCP) & Tool Ecosystem](#day-8--model-context-protocol-mcp--tool-ecosystem)
3. [Day 9 — Multi-Agent Frameworks (CrewAI, LangGraph, OpenAI Agents SDK)](#day-9--multi-agent-frameworks-crewai-langgraph-openai-agents-sdk)
4. [Day 10 — Structured Output, Guardrails & Safety](#day-10--structured-output-guardrails--safety)
5. [Day 11 — LLM Evaluation & Testing](#day-11--llm-evaluation--testing)
6. [Day 12 — Fine-Tuning, Prompt Engineering & Production Patterns](#day-12--fine-tuning-prompt-engineering--production-patterns)
7. [Interview Cheat Sheet — Buzzwords & Resume Keywords](#-interview-cheat-sheet--buzzwords--resume-keywords)
8. [System Design Patterns for AI Interviews](#-system-design-patterns-for-ai-interviews)
9. [Glossary](#glossary)

---

# Day 7 — AI Agents & Agentic Architectures

## 1. What Is an AI Agent? (Plain English)

Think of ChatGPT as a **one-shot answering machine** — you ask, it answers, done. An **AI Agent** is more like a smart intern:

- It **thinks** about what to do
- It **uses tools** (search the web, query a database, send an email)
- It **observes** the result
- It **decides** the next step
- It **loops** until the task is complete

```
User: "Book me a flight from SFO to JFK next Friday under $300"

Agent thinks: "I need to search flights first"
  → Calls flight_search_tool(from="SFO", to="JFK", date="next Friday")
Agent observes: "Found 3 options: $250, $280, $350"
Agent thinks: "Two are under $300, pick cheapest"
  → Calls book_flight_tool(flight_id="FL250")
Agent responds: "Booked! $250 flight, confirmation #ABC123"
```

### Interview Buzzword: "Agentic AI"
Agentic AI means the system can plan, execute, and adapt — not just answer one question.

---

## 2. The Agent Loop — ReAct Pattern

The most common agent pattern is **ReAct (Reasoning + Acting)**:

```
┌──────────────────────────────┐
│  User gives a task           │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  REASON: Think about         │ ◄──────────────┐
│  what to do next             │                │
└──────────┬───────────────────┘                │
           ▼                                    │
┌──────────────────────────────┐                │
│  ACT: Call a tool or         │                │
│  generate a response         │                │
└──────────┬───────────────────┘                │
           ▼                                    │
┌──────────────────────────────┐                │
│  OBSERVE: Look at the        │                │
│  tool's output               │────────────────┘
└──────────┬───────────────────┘      (loop until done)
           ▼
┌──────────────────────────────┐
│  RESPOND to user             │
└──────────────────────────────┘
```

### Key Agent Patterns (Know These for Interviews!)

| Pattern | What It Does | When to Use |
|---------|-------------|-------------|
| **ReAct** | Reason → Act → Observe loop | General-purpose agent tasks |
| **Plan-and-Execute** | Make full plan first, then execute steps | Complex multi-step tasks |
| **Reflection** | Agent critiques its own output and improves | Quality-sensitive applications |
| **Tool Use** | Agent selects and calls external tools | When LLM needs real-world data |
| **Routing** | Agent decides which sub-agent handles the task | Multi-domain applications |
| **Handoff** | Agent passes conversation to another agent | Customer support, triage |

---

## 3. OpenAI's Agent Platform (Current State — Feb 2026)

OpenAI has gone all-in on agents. Here's their stack:

### Responses API (Replaces Assistants API — sunsetting mid-2026)
The new primary API for building agents. Combines Chat Completions simplicity with tool-use capabilities.

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    tools=[{"type": "web_search"}],  # Built-in tool!
    input="What's the latest news about SpaceX?"
)
print(response.output_text)
```

### Built-in Tools (No Code Required!)
| Tool | What It Does | Interview Note |
|------|-------------|------------------|
| **Web Search** | Real-time internet search | Grounds LLM in current data |
| **File Search** | Searches uploaded documents (vector store) | Enterprise RAG without infra |
| **Computer Use** | Controls a computer (clicks, types, screenshots) | Agentic UI automation |

### Computer Use Agent (CUA)
- Scores **38.1%** on OSWorld benchmark, **87%** on WebVoyager
- Can literally browse the web, fill forms, click buttons
- Uses screenshots + reasoning to navigate UIs
- "OpenAI's CUA model can autonomously operate computer interfaces"

---

## 4. Anthropic's Approach to Agents

Anthropic (makers of Claude) takes a more **principled approach**:

### Key Concepts
- **Tool Use**: Claude can call functions you define (similar to OpenAI function calling)
- **Extended Thinking**: Claude can "think" step-by-step before responding (like o1/o3)
- **Claude Agent SDK**: Official SDK for building agents
- **Claude Code**: An agentic coding assistant that can write, edit, and debug code

### Anthropic's Agent Philosophy
Start with the simplest solution. Don't build an agent if a single prompt works. Don't build multi-agent if a single agent works.



---

## 5. The "Levels" of Agentic Systems

```
Level 0: Simple LLM Call
  → input → LLM → output

Level 1: LLM + Tools (Function Calling)
  → input → LLM → tool call → LLM → output

Level 2: Single Agent (ReAct loop)
  → input → [plan → act → observe]* → output

Level 3: Multi-Agent System
  → input → Router Agent → [Agent A, Agent B, Agent C] → output

Level 4: Autonomous Agent (long-running)
  → goal → agent runs indefinitely, using tools, memory, and sub-agents
```

---

# Day 8 — Model Context Protocol (MCP) & Tool Ecosystem

## 1. What Is MCP? (The "USB-C for AI" — Know This!)

**MCP (Model Context Protocol)** is THE hottest standard in AI right now. Created by Anthropic, now an open standard under the Linux Foundation.

### Plain English
> Think of MCP like **USB-C for AI**. Just as USB-C lets you connect any device to any computer with one standard cable, MCP lets any AI app connect to any data source or tool with one standard protocol.

```
Before MCP:
  ChatGPT → custom Google Calendar integration
  ChatGPT → custom Slack integration  
  Claude  → different Google Calendar integration  (each one built separately!)
  Claude  → different Slack integration

With MCP:
  Any AI App → MCP → Google Calendar MCP Server
                   → Slack MCP Server
                   → Database MCP Server
                   → Any Tool MCP Server
```

### Why Interviewers Love MCP
- It's **brand new** (2024-2025) — shows you're current
- It's **open source** — shows you understand open ecosystems
- It's **practical** — every major AI company is adopting it
- Used by: **Claude, VS Code, Cursor, Windsurf, Copilot, and more**

---

## 2. MCP Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   MCP Client     │────▶│   MCP Server     │────▶│   Data Source     │
│  (AI App)        │◀────│  (middleware)     │◀────│   or Tool        │
│                  │     │                  │     │                  │
│  e.g., Claude,   │     │  e.g., Slack     │     │  e.g., Slack API │
│  VS Code, etc.   │     │  MCP Server      │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

### MCP provides three primitives:
| Primitive | What It Does | Example |
|-----------|-------------|---------|
| **Tools** | Functions the AI can call | `search_emails(query)`, `create_ticket(title)` |
| **Resources** | Data the AI can read | Files, database records, API responses |
| **Prompts** | Pre-written prompt templates | "Summarize this PR", "Write a test for this" |

---

## 3. Building an MCP Server (Python)

```python
# Simple MCP server example
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("my-weather-server")

@server.tool()
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Call weather API here
    return f"Weather in {city}: 72°F, Sunny"

# Now ANY MCP-compatible AI client can use this tool!
```

### Resume Keyword: "Built MCP servers for enterprise tool integration"

---

## 4. MCP Ecosystem — Current State

| MCP Server | What It Connects | Stars |
|-----------|-----------------|-------|
| GitHub MCP | Repos, issues, PRs | Official |
| Slack MCP | Channels, messages | Official |
| Google Drive MCP | Docs, sheets | Official |
| PostgreSQL MCP | Database queries | Community |
| Brave Search MCP | Web search | Official |
| Filesystem MCP | Local files | Official |

**Interview Tip:** "MCP is how I'd architect an AI system that needs to connect to multiple enterprise tools — one standard protocol instead of N custom integrations."

---

# Day 9 — Multi-Agent Frameworks (CrewAI, LangGraph, OpenAI Agents SDK)

## 1. Why Multi-Agent? (The Team Analogy)

Single agent = one person doing everything. Multi-agent = a **team of specialists**:

```
User: "Research AI trends and write a blog post"

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Researcher │───▶│   Writer    │───▶│   Editor    │
│  Agent      │    │   Agent     │    │   Agent     │
│             │    │             │    │             │
│ "Find the   │    │ "Write an   │    │ "Polish and │
│  latest     │    │  engaging   │    │  fact-check │
│  AI news"   │    │  article"   │    │  the draft" │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## 2. Framework Comparison (Feb 2026)

| Feature | **OpenAI Agents SDK** | **CrewAI** | **LangGraph** |
|---------|----------------------|-----------|--------------|
| **GitHub Stars** | ~20K (openai repo) | **44.3K** ⭐ | ~15K (LangChain) |
| **Architecture** | Agent + Handoffs | Crew + Flows | State Graph (nodes + edges) |
| **Learning Curve** | 🟢 Easy | 🟢 Easy | 🟡 Medium |
| **Flexibility** | 🟡 Medium | 🟢 High | 🟢 Very High |
| **LLM Lock-in?** | Works with any LLM | Works with any LLM | Works with any LLM |
| **Production Ready** | 🟢 Yes | 🟢 Yes (Flows) | 🟢 Yes |
| **Tracing Built-in** | ✅ Yes | ✅ Yes | ✅ Via LangSmith |
| **Unique Feature** | Guardrails, Handoffs | Crews + Flows, 100K+ devs | Durable execution, checkpointing |
| **Best For** | Quick agent prototypes | Multi-agent teams | Complex stateful workflows |

---

## 3. OpenAI Agents SDK — Code Example

```python
from agents import Agent, Runner, function_tool, WebSearchTool

@function_tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price."""
    return f"{ticker}: $150.23"

# Define agents
researcher = Agent(
    name="Stock Researcher",
    instructions="Research stock information using available tools.",
    tools=[WebSearchTool(), get_stock_price]
)

writer = Agent(
    name="Report Writer",
    instructions="Write investment reports based on research.",
)

# Handoff pattern: researcher passes to writer
researcher.handoffs = [writer]

# Run the agent
result = Runner.run_sync(
    researcher,
    "Research AAPL and write a brief investment summary"
)
print(result.final_output)
```

### Key Concepts:
- **Agent**: Has name, instructions, tools
- **Handoff**: Agent passes control to another agent
- **Guardrails**: Validate input/output (block harmful content)
- **Tracing**: Built-in observability for debugging

---

## 4. CrewAI — Code Example

```python
from crewai import Agent, Task, Crew, Process

# Define agents with roles
researcher = Agent(
    role="Senior Research Analyst",
    goal="Find cutting-edge AI developments",
    backstory="You're a veteran researcher at a top AI lab.",
    tools=[search_tool],
    verbose=True
)

writer = Agent(
    role="Tech Blog Writer",
    goal="Write engaging technical articles",
    backstory="You write for a major tech publication.",
    verbose=True
)

# Define tasks
research_task = Task(
    description="Research the latest AI agent frameworks in 2026",
    expected_output="A list of 10 key findings with sources",
    agent=researcher
)

write_task = Task(
    description="Write a blog post based on the research",
    expected_output="A 1000-word blog post in markdown",
    agent=writer,
    output_file="blog_post.md"
)

# Create and run crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,  # or Process.hierarchical
    verbose=True
)

result = crew.kickoff()
```

### CrewAI Unique Features:
- **Crews**: Autonomous teams of agents (high-level)
- **Flows**: Event-driven workflows with `@start`, `@listen`, `@router` (low-level)
- **100K+ certified developers** through learn.crewai.com
- **5.76x faster** than LangGraph in certain benchmarks (their claim)
- **Standalone** — NOT built on LangChain (fully independent since rewrite)

---

## 5. LangGraph — Code Example

```python
from langgraph.graph import StateGraph, MessagesState, START, END

def researcher(state: MessagesState):
    # Call LLM to research
    return {"messages": [{"role": "ai", "content": "Research findings..."}]}

def writer(state: MessagesState):
    # Call LLM to write based on research
    return {"messages": [{"role": "ai", "content": "Blog post draft..."}]}

# Build the graph
graph = StateGraph(MessagesState)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_edge(START, "researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", END)

app = graph.compile()
result = app.invoke({"messages": [{"role": "user", "content": "Write about AI"}]})
```

### LangGraph Unique Features:
- **State Graph**: Define workflows as nodes + edges (very explicit control)
- **Durable Execution**: Agents persist through failures, can resume
- **Human-in-the-Loop**: Pause execution for human approval
- **Checkpointing**: Save/restore agent state at any point
- **LangSmith Integration**: Full observability and tracing
- Used by: **Klarna, Replit, Elastic**

---

## 6. When to Use Which?

Choose the framework based on the use case:
- **OpenAI Agents SDK** for quick prototypes where you need handoffs and guardrails with minimal code
- **CrewAI** when you need role-based agents collaborating on a task, especially with Flows for production
- **LangGraph** when you need maximum control over state transitions, human-in-the-loop, and durable execution for mission-critical workflows

---

# Day 10 — Structured Output, Guardrails & Safety

## 1. Structured Output — Why It Matters

LLMs output free-form text by default. In production, you need **structured data**:

```python
# ❌ Without structured output
response = "The weather is 72 degrees and sunny in San Francisco"

# ✅ With structured output
response = {
    "city": "San Francisco",
    "temperature": 72,
    "unit": "fahrenheit",
    "condition": "sunny"
}
```

### Methods for Structured Output (Know These!)

| Method | How It Works | Reliability | Provider |
|--------|-------------|------------|---------|
| **JSON Mode** | Ask LLM to return JSON | 🟡 Medium | OpenAI, Anthropic |
| **Function Calling** | Define functions, LLM fills parameters | 🟢 High | OpenAI, Anthropic, Google |
| **Structured Outputs** | Define JSON schema, guaranteed compliance | 🟢 Very High | OpenAI (response_format) |
| **Pydantic + Instructor** | Python classes → schema → validated output | 🟢 Very High | Any LLM (via Instructor lib) |
| **Grammar-based** | Constrain token generation | 🟢 Guaranteed | Ollama, llama.cpp |

---

## 2. Pydantic — The AI Engineer's Best Friend

**Pydantic** (26.9K GitHub stars) is a data validation library for Python. In AI, it's used EVERYWHERE:

```python
from pydantic import BaseModel, Field
from openai import OpenAI

class MovieReview(BaseModel):
    """Structured movie review."""
    title: str = Field(description="Movie title")
    rating: float = Field(ge=0, le=10, description="Rating out of 10")
    sentiment: str = Field(description="positive, negative, or neutral")
    summary: str = Field(description="One-sentence summary")

client = OpenAI()
response = client.responses.create(
    model="gpt-4.1",
    input="Review the movie Inception",
    text={"format": {"type": "json_schema", "schema": MovieReview.model_json_schema()}}
)
# Returns GUARANTEED valid JSON matching your schema!
```

### Resume Keyword: "Pydantic-based structured output pipelines"

---

## 3. Instructor Library — Structured Outputs Made Easy

**Instructor** (8K+ stars) patches any LLM client to return Pydantic objects:

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
    messages=[{"role": "user", "content": "John is 30, email: john@example.com"}]
)

print(user.name)   # "John"
print(user.age)    # 30
print(user.email)  # "john@example.com"
```

---

## 4. Guardrails — Protecting Your AI App

**Guardrails AI** (6.4K stars) — A framework for input/output validation on LLM calls:

```python
from guardrails import Guard, OnFailAction
from guardrails.hub import ToxicLanguage, CompetitorCheck

# Create a guard with multiple validators
guard = Guard().use(
    ToxicLanguage(threshold=0.5, on_fail=OnFailAction.EXCEPTION),
    CompetitorCheck(["Apple", "Google"], on_fail=OnFailAction.EXCEPTION)
)

# Validate LLM output before returning to user
guard.validate("Here's a helpful response about our product...")
```

### Types of Guardrails (Interview Material!)

| Category | Examples | Why It Matters |
|----------|---------|---------------|
| **Content Safety** | Toxicity, hate speech, PII detection | Legal compliance, brand safety |
| **Factual Accuracy** | Hallucination detection, fact-checking | Trust & reliability |
| **Format Validation** | JSON schema, regex matching | System integration |
| **Business Logic** | Competitor mentions, off-topic detection | Brand protection |
| **Security** | Prompt injection detection, SQL injection | Application security |

### OpenAI Agents SDK Guardrails
```python
from agents import Agent, InputGuardrail, GuardrailFunctionOutput

@InputGuardrail
async def block_harmful(ctx, agent, input):
    # Check if input is harmful
    result = await Runner.run(safety_checker, input)
    return GuardrailFunctionOutput(
        output_info=result,
        tripwire_triggered=result.is_harmful
    )

agent = Agent(
    name="Customer Support",
    input_guardrails=[block_harmful]
)
```

---

## 5. AI Safety Concepts (Interview Must-Know)

| Concept | What It Is | Why It Matters |
|---------|-----------|---------------|
| **Prompt Injection** | User tricks AI into ignoring instructions | #1 security risk in AI apps |
| **Jailbreaking** | Bypassing AI safety filters | Content policy violations |
| **Hallucination** | AI generates false but confident information | Trust & liability |
| **Data Leakage** | AI reveals training data or PII | Privacy/GDPR violations |
| **Alignment** | Ensuring AI does what humans intend | Core safety research area |
| **RLHF** | Reinforcement Learning from Human Feedback | How models learn human preferences |
| **Constitutional AI** | AI self-corrects based on principles | Anthropic's safety approach |
| **Red Teaming** | Adversarial testing of AI systems | Required for production deployment |

---

# Day 11 — LLM Evaluation & Testing

## 1. Why Evaluate LLMs? (The "Vibes" Problem)

Most developers evaluate LLMs by... trying it a few times and seeing if it "feels right." This is called **"vibe-based evaluation"** and it does NOT scale.

Production AI needs **systematic evaluation**:

```
Vibe Check: "Hmm, the response looks good" ❌
Systematic: "Answer relevancy: 0.92, Faithfulness: 0.88, Toxicity: 0.01" ✅
```

---

## 2. DeepEval — The Pytest for LLMs

**DeepEval** (13.7K stars) — Open-source LLM evaluation framework:

```python
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

def test_rag_response():
    # Define metrics
    relevancy = AnswerRelevancyMetric(threshold=0.7)
    faithfulness = FaithfulnessMetric(threshold=0.8)

    # Create test case
    test_case = LLMTestCase(
        input="What is our refund policy?",
        actual_output="You can get a full refund within 30 days.",
        retrieval_context=["All customers get a 30-day full refund."]
    )

    # Assert like a normal test!
    assert_test(test_case, [relevancy, faithfulness])

# Run with: deepeval test run test_file.py
```

### Key Metrics for RAG Systems

| Metric | What It Measures | Score Range |
|--------|-----------------|-------------|
| **Answer Relevancy** | Is the answer relevant to the question? | 0–1 |
| **Faithfulness** | Is the answer grounded in retrieved context? | 0–1 |
| **Contextual Precision** | Are relevant docs ranked higher? | 0–1 |
| **Contextual Recall** | Were all relevant docs retrieved? | 0–1 |
| **Hallucination** | Does the answer contain made-up info? | 0–1 |
| **G-Eval** | Custom criteria evaluation using LLM-as-judge | 0–1 |
| **Task Completion** | Did the agent complete the task? (for agents) | 0–1 |
| **Tool Correctness** | Did the agent use the right tools? (for agents) | 0–1 |

---

## 3. Evaluation Approaches (Know The Landscape)

| Approach | How It Works | Tools |
|----------|-------------|-------|
| **LLM-as-Judge** | Use a stronger LLM to evaluate outputs | DeepEval, G-Eval |
| **Human Evaluation** | Human raters score outputs | Labelbox, Scale AI |
| **Reference-based** | Compare against gold-standard answers | BLEU, ROUGE, BERTScore |
| **Behavioral Testing** | Test specific capabilities/edge cases | CheckList, custom tests |
| **Red Teaming** | Adversarial testing for safety | DeepEval, Garak |
| **A/B Testing** | Compare two versions in production | Custom, Confident AI |

### Resume Keywords: 
- "Built LLM evaluation pipelines using DeepEval/RAGAS"
- "Implemented LLM-as-judge evaluation with G-Eval"
- "Automated RAG quality metrics in CI/CD"

---

## 4. Observability Stack (Recap + New)

| Tool | Stars | What It Does | Key Feature |
|------|-------|-------------|-------------|
| **Langfuse** | 22K | Tracing, evaluation, prompt management | Open-source, self-hostable |
| **LangSmith** | — | Tracing, debugging, evaluation | Tight LangChain/LangGraph integration |
| **Helicone** | 5K+ | Logging, caching, cost tracking | One-line integration |
| **Confident AI** | — | DeepEval cloud platform | Evaluation datasets, A/B testing |
| **Arize Phoenix** | 12K+ | Tracing, evaluation, embeddings viz | Great for debugging RAG |
| **Weights & Biases** | 9K+ | Experiment tracking, model evaluation | ML experiment management |

---

# Day 12 — Fine-Tuning, Prompt Engineering & Production Patterns

## 1. Fine-Tuning — When and Why?

### Decision Tree (Interview Answer!)

```
Do you need specialized behavior?
├── No → Use prompt engineering (few-shot, system prompts)
├── Sort of → Use RAG (add your data as context)
└── Yes, deeply specialized →
    ├── Do you have lots of labeled data?
    │   ├── Yes → Fine-tune
    │   └── No → Use few-shot prompting + RAG
    └── Is latency critical?
        ├── Yes → Fine-tune a smaller model (distillation)
        └── No → Use a bigger model with good prompts
```

### Fine-Tuning Options (Current Market)

| Provider | What You Can Fine-Tune | Cost | Best For |
|----------|----------------------|------|---------|
| **OpenAI** | GPT-4.1 mini, GPT-4o, GPT-4.1 nano | ~$25/M training tokens | Production applications |
| **Together.ai** | Llama, Mistral, any open model | ~$5/M tokens | Open-source models |
| **Hugging Face** | Any model on the Hub | Your own GPU costs | Full control |
| **Google Vertex** | Gemini models | Usage-based | Google Cloud users |
| **AWS Bedrock** | Claude, Llama, Titan | Usage-based | AWS ecosystem |

### Distillation — Technique
> Use a **large expensive model** (GPT-5.2) to generate training data, then **fine-tune a small cheap model** (GPT-4.1 nano) on that data. Result: small model performs like the big one for your specific task.

```
GPT-5.2 ($14/M output tokens) generates 10K examples
     ↓
Fine-tune GPT-4.1 nano ($0.40/M output tokens) on those examples
     ↓
Nano model achieves ~90% of GPT-5.2 quality at 35x lower cost!
```

### Resume Keyword: "Model distillation for cost-optimized inference"

---

## 2. Advanced Prompt Engineering (Beyond Basics)

### Techniques Every AI Engineer Should Know

| Technique | What It Is | When to Use |
|-----------|-----------|-------------|
| **System Prompts** | Set the AI's persona and rules | Always — foundation of every app |
| **Few-Shot Prompting** | Give examples in the prompt | When you need specific output format |
| **Chain-of-Thought (CoT)** | "Think step by step" | Math, reasoning, complex problems |
| **Self-Consistency** | Run multiple times, take majority vote | When accuracy matters more than speed |
| **Prompt Chaining** | Break task into sequential LLM calls | Complex multi-step tasks |
| **Meta-Prompting** | Use LLM to write/improve prompts | Prompt optimization |
| **Retrieval-Augmented** | Add context from a vector DB (RAG) | When LLM needs your specific data |

### The Prompt Engineering Stack (Production)

```
┌─────────────────────────────────┐
│  Prompt Management (Langfuse)   │  ← Version control for prompts
├─────────────────────────────────┤
│  Prompt Templates               │  ← Jinja2 / f-string templates
├─────────────────────────────────┤
│  Few-Shot Example Selection     │  ← Dynamic example retrieval
├─────────────────────────────────┤
│  Evaluation Loop                │  ← A/B test prompt variants
└─────────────────────────────────┘
```

---

## 3. Production Architecture Patterns

### Pattern 1: Simple LLM App
```
User → API → LLM → Response
```
- Use for: Chatbots, text generation, simple Q&A
- Tools: OpenAI API, FastAPI/Flask

### Pattern 2: RAG Pipeline
```
User → API → Retriever (Vector DB) → LLM (with context) → Response
```
- Use for: Knowledge-base Q&A, document search
- Tools: pgvector/Pinecone, OpenAI Embeddings, LangChain

### Pattern 3: Agentic RAG
```
User → Agent → [Search Tool, Calculator, DB Query, RAG] → Response
```
- Use for: Complex research tasks, data analysis
- Tools: OpenAI Agents SDK, CrewAI, LangGraph

### Pattern 4: Multi-Agent Pipeline
```
User → Router Agent → [Research Agent, Writing Agent, Review Agent] → Response
```
- Use for: Content creation, code generation, complex workflows
- Tools: CrewAI (Crews + Flows), LangGraph

### Pattern 5: Human-in-the-Loop
```
User → Agent → [work] → ⏸️ Human Review → [continue] → Response
```
- Use for: High-stakes decisions, content moderation
- Tools: LangGraph (interrupts), custom approval flows

---

## 4. The AI Engineer Tech Stack (What Employers Want)

```
┌────────────────────────────────────────────────────────────┐
│                    PRODUCTION STACK                         │
├────────────────────────────────────────────────────────────┤
│  Frontend    │  React/Next.js, Streamlit, Gradio           │
│  Backend     │  FastAPI, Flask, Go, Node.js                │
│  LLM APIs   │  OpenAI, Anthropic, Google, Mistral          │
│  Orchestr.   │  LangChain, LangGraph, CrewAI                │
│  Vector DB   │  pgvector, Pinecone, Chroma, Weaviate        │
│  Embeddings  │  OpenAI, Cohere, Voyage, BGE                 │
│  Eval        │  DeepEval, RAGAS, LangSmith                  │
│  Guardrails  │  Guardrails AI, NeMo Guardrails              │
│  Observ.     │  Langfuse, Helicone, LangSmith, Arize        │
│  Infra       │  Docker, K8s, AWS/GCP/Azure                  │
│  CI/CD       │  GitHub Actions, eval in CI pipeline          │
│  Data        │  PostgreSQL, Redis, S3, Kafka                │
│  Protocol    │  MCP (Model Context Protocol)                │
└────────────────────────────────────────────────────────────┘
```

---

# Interview Cheat Sheet — Buzzwords & Resume Keywords

## Resume Section: "Skills"

### For SDE2 / AI Engineer
```
AI/ML: Large Language Models (GPT-4.1, Claude, Gemini), Embeddings, 
       RAG, AI Agents, Multi-Agent Systems, Fine-Tuning, Distillation
       
Frameworks: LangChain, LangGraph, CrewAI, OpenAI Agents SDK, 
            Instructor, Pydantic, Guardrails AI

Vector Databases: pgvector, Pinecone, Chroma, FAISS

Protocols: Model Context Protocol (MCP), OpenAPI

Evaluation: DeepEval, RAGAS, LLM-as-Judge, G-Eval, Red Teaming

Observability: Langfuse, LangSmith, Helicone, OpenTelemetry

Infrastructure: Docker, Kubernetes, AWS (Bedrock, SageMaker), 
               GCP (Vertex AI), Azure (OpenAI Service)

Languages: Python, Go, TypeScript, SQL
```

### For GenAI Engineer (More AI-Heavy)
```
GenAI: Transformer Architecture, Attention Mechanisms, Tokenization,
       RLHF, Constitutional AI, DPO, Prompt Engineering, 
       Chain-of-Thought, Few-Shot Learning

Agent Systems: ReAct Pattern, Plan-and-Execute, Tool Use, 
               Multi-Agent Orchestration, Handoffs, Routing

Production AI: Structured Outputs, Guardrails, Content Moderation,
               Prompt Injection Defense, Hallucination Detection,
               Model Distillation, A/B Testing Prompts

MLOps: Model Serving, Batch vs Real-time Inference, 
       Caching Strategies, Rate Limiting, Cost Optimization
```

---

## Top 50 Interview Buzzwords (Ranked by Frequency in Job Postings)

### Tier 1 — Must Know (Asked in 80%+ of interviews)
| # | Buzzword | One-Line Definition |
|---|----------|-------------------|
| 1 | **RAG** | Retrieval-Augmented Generation — add context from docs before LLM call |
| 2 | **Embeddings** | Convert text to numbers (vectors) for semantic similarity |
| 3 | **Vector Database** | Database optimized for storing/searching embeddings |
| 4 | **Fine-Tuning** | Train a model further on your specific data |
| 5 | **Prompt Engineering** | Crafting inputs to get desired outputs from LLMs |
| 6 | **LLM (Large Language Model)** | The big AI models — GPT, Claude, Gemini, Llama |
| 7 | **AI Agents** | AI systems that autonomously plan, use tools, and execute tasks |
| 8 | **Function/Tool Calling** | LLM triggers external functions (APIs, DBs, tools) |
| 9 | **Hallucination** | When AI makes up false information confidently |
| 10 | **Transformer** | The neural network architecture behind all modern LLMs |

### Tier 2 — Should Know (Asked in 50%+ of interviews)
| # | Buzzword | One-Line Definition |
|---|----------|-------------------|
| 11 | **MCP (Model Context Protocol)** | USB-C for AI — standard protocol for tool integration |
| 12 | **Guardrails** | Input/output validation on AI responses |
| 13 | **Structured Output** | Force LLMs to return data in a specific format (JSON, etc.) |
| 14 | **Chain-of-Thought** | Making LLMs "think step by step" for better reasoning |
| 15 | **Multi-Agent Systems** | Multiple AI agents collaborating on tasks |
| 16 | **RLHF** | Reinforcement Learning from Human Feedback — how models are trained |
| 17 | **Tokenization** | Splitting text into tokens (subwords) for model input |
| 18 | **Context Window** | Maximum input size an LLM can process at once |
| 19 | **Temperature** | Controls randomness/creativity of LLM output (0=deterministic, 1=creative) |
| 20 | **Distillation** | Train a small model to mimic a large model's behavior |

### Tier 3 — Nice to Know (Differentiators)
| # | Buzzword | One-Line Definition |
|---|----------|-------------------|
| 21 | **Agentic RAG** | RAG where an agent decides when/what to retrieve |
| 22 | **GraphRAG** | RAG using knowledge graphs instead of just vector search |
| 23 | **Semantic Chunking** | Splitting documents by meaning rather than fixed size |
| 24 | **Mixture of Experts (MoE)** | Model architecture that activates only relevant "experts" per query |
| 25 | **Quantization** | Compress model weights (32-bit → 4-bit) for faster inference |
| 26 | **LoRA / QLoRA** | Efficient fine-tuning that only trains small adapter layers |
| 27 | **DPO (Direct Preference Optimization)** | Alternative to RLHF, simpler alignment technique |
| 28 | **Constitutional AI** | Anthropic's self-correcting safety approach |
| 29 | **Attention Mechanism** | How transformers weigh importance of different input parts |
| 30 | **Reasoning Models** | o3, o4-mini — models that "think" before answering |

### Tier 4 — Bleeding Edge (Wow Factor)
| # | Buzzword | One-Line Definition |
|---|----------|-------------------|
| 31 | **Computer Use Agent (CUA)** | AI that can control a computer (click, type, navigate) |
| 32 | **Extended Thinking** | Claude's visible chain-of-thought reasoning |
| 33 | **Multimodal AI** | Models that handle text + images + audio + video |
| 34 | **Synthetic Data Generation** | Using AI to create training data |
| 35 | **Prompt Caching** | Cache repeated prompt prefixes for cost savings |
| 36 | **Batch API** | Send bulk requests at 50% discount (async processing) |
| 37 | **Evals Pipeline** | Automated testing of LLM quality in CI/CD |
| 38 | **Observability (LLMOps)** | Monitoring, tracing, and debugging AI systems in production |
| 39 | **Retrieval Reranking** | Second-pass ranking of retrieved documents for better accuracy |
| 40 | **Hybrid Search** | Combining vector search + keyword search for better retrieval |

### Tier 5 — Deep Technical (For Senior/Staff roles)
| # | Buzzword | One-Line Definition |
|---|----------|-------------------|
| 41 | **KV Cache** | Optimization that caches key-value pairs in attention layers |
| 42 | **Speculative Decoding** | Use small model to draft, large model to verify (faster inference) |
| 43 | **GGUF / GGML** | File formats for quantized models (used with llama.cpp) |
| 44 | **vLLM** | High-throughput LLM serving engine |
| 45 | **Flash Attention** | Memory-efficient attention computation |
| 46 | **Prefix Tuning** | Fine-tuning by only learning a small prefix vector |
| 47 | **Retrieval Interleaved Generation (RIG)** | Google's approach — retrieve mid-generation |
| 48 | **Model Merging** | Combine weights from multiple fine-tuned models |
| 49 | **AI Gateway** | Proxy that routes, caches, and rate-limits LLM API calls |
| 50 | **Inference Optimization** | Techniques to reduce latency and cost of running models |

---

# System Design Patterns for AI Interviews

## Common Interview Question: "Design an AI-powered customer support system"

```
┌─────────────────────────────────────────────────────────────────┐
│                        Architecture                              │
│                                                                  │
│  User ──▶ API Gateway ──▶ Intent Router (LLM) ──────────────── │
│                              │          │          │             │
│                              ▼          ▼          ▼             │
│                          FAQ Agent  Ticket Agent  Human Handoff  │
│                              │          │          │             │
│                              ▼          ▼          ▼             │
│                          Vector DB   CRM API    Support Team     │
│                         (RAG for     (Create/    (Slack/Email)   │
│                          FAQs)       Update)                     │
│                                                                  │
│  Monitoring: Langfuse traces ← every LLM call                   │
│  Guardrails: Input safety check ← every user message             │
│  Evaluation: DeepEval ← nightly quality assessment               │
│  Cost Control: Prompt caching + model routing (cheap→expensive)  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Talking Points:
1. **Model Routing**: Use cheap model (GPT-4.1 nano, $0.10/M) for simple queries, expensive model (GPT-4.1, $2/M) for complex ones
2. **RAG for Knowledge**: Company FAQ/docs stored in pgvector, retrieved for context
3. **Guardrails**: Block prompt injection, PII leakage, toxic content
4. **Observability**: Trace every LLM call with Langfuse for debugging
5. **Evaluation**: Nightly DeepEval runs to catch quality regression
6. **Human-in-the-Loop**: Escalation path when AI confidence is low
7. **Cost Optimization**: Prompt caching, batch processing, model distillation

---

## Common Interview Question: "How do you evaluate an LLM application?"

### The Three Layers of Evaluation

```
Layer 1: OFFLINE EVALUATION (Before deployment)
├── Unit Tests: Test individual LLM calls with DeepEval
├── Dataset Evaluation: Test against curated test datasets
├── Red Teaming: Adversarial testing for safety
└── Regression Tests: Ensure new prompts don't break old behavior

Layer 2: ONLINE EVALUATION (During deployment)  
├── A/B Testing: Compare prompt/model variants on real traffic
├── Shadow Mode: Run new model alongside old, compare outputs
└── Canary Deployment: Gradually roll out to % of users

Layer 3: PRODUCTION MONITORING (After deployment)
├── Trace Analysis: Review LLM traces in Langfuse/LangSmith
├── User Feedback: Thumbs up/down, explicit feedback
├── Quality Alerts: Alert when metrics drop below threshold
└── Cost Monitoring: Track spend per model, per feature
```

---

## Current Model Landscape — Quick Reference (Feb 2026)

### OpenAI
| Model | Price (input/output per 1M) | Best For |
|-------|---------------------------|---------|
| GPT-5.2 | $1.75 / $14 | Flagship, complex tasks |
| GPT-5.2 pro | $21 / $168 | Maximum intelligence |
| GPT-5 mini | $0.25 / $2 | Cost-effective smart model |
| GPT-4.1 | $2 / $8 | Coding, instructions |
| GPT-4.1 mini | $0.40 / $1.60 | Balanced cost/quality |
| GPT-4.1 nano | $0.10 / $0.40 | Ultra-cheap, high-volume |
| o3 | $2 / $8 | Deep reasoning |
| o4-mini | $0.50 / $2 | Budget reasoning |

### Anthropic (Claude)
| Model | Best For |
|-------|---------|
| Claude 4.6 Opus | Most capable, complex analysis |
| Claude Sonnet | Balanced, great for coding |
| Claude Haiku | Fast & cheap |

### Open Source
| Model | Parameters | Why It Matters |
|-------|-----------|---------------|
| Llama 3.1 (Meta) | 8B / 70B / 405B | Most popular open-source LLM |
| Mistral Large | Various | Strong European competitor |
| DeepSeek R1 | 671B (MoE) | Chinese open-source reasoning model |
| Qwen 2.5 (Alibaba) | Various | Strong multilingual model |
| Gemma 2 (Google) | 2B / 9B / 27B | Small but capable |

---

# Glossary

| Term | Definition |
|------|-----------|
| **Agent** | AI system that autonomously plans, uses tools, and executes tasks |
| **Agentic AI** | AI capable of autonomous decision-making and action |
| **Alignment** | Ensuring AI behavior matches human intentions |
| **Attention** | Mechanism in transformers that weighs input importance |
| **Batch API** | Bulk async LLM requests at reduced cost |
| **Chain-of-Thought (CoT)** | Prompting technique for step-by-step reasoning |
| **Computer Use Agent (CUA)** | AI that operates computer interfaces |
| **Constitutional AI** | Self-correcting AI safety framework (Anthropic) |
| **Context Window** | Maximum input size for an LLM |
| **CrewAI** | Multi-agent framework (44.3K stars) |
| **DeepEval** | LLM evaluation framework (13.7K stars) |
| **Distillation** | Training small model on large model's outputs |
| **DPO** | Direct Preference Optimization — simpler alternative to RLHF |
| **Extended Thinking** | Claude's visible reasoning process |
| **Few-Shot** | Providing examples in the prompt |
| **Fine-Tuning** | Further training a model on specific data |
| **Function Calling** | LLM invokes external functions/APIs |
| **G-Eval** | Research metric using LLM-as-judge |
| **GraphRAG** | RAG using knowledge graphs |
| **Guardrails** | Input/output validation for AI safety |
| **Hallucination** | AI generating false but confident information |
| **Handoff** | One agent passing control to another |
| **Human-in-the-Loop** | Human approval step in AI workflows |
| **Instructor** | Library for structured LLM outputs via Pydantic |
| **LangGraph** | Stateful agent orchestration framework |
| **Langfuse** | Open-source LLM observability platform (22K stars) |
| **LoRA** | Low-Rank Adaptation — efficient fine-tuning |
| **MCP** | Model Context Protocol — USB-C for AI tools |
| **MoE** | Mixture of Experts — efficient model architecture |
| **Multi-Agent** | Multiple AI agents collaborating |
| **Pydantic** | Python data validation (26.9K stars) |
| **Quantization** | Model compression (reduce bit precision) |
| **RAG** | Retrieval-Augmented Generation |
| **RAGAS** | RAG evaluation framework |
| **ReAct** | Reasoning + Acting — core agent loop pattern |
| **Red Teaming** | Adversarial safety testing |
| **RLHF** | Reinforcement Learning from Human Feedback |
| **Routing** | Directing queries to appropriate models/agents |
| **Structured Output** | Forcing LLMs to produce formatted data |
| **Temperature** | Controls LLM output randomness |
| **Tokenization** | Splitting text into model-readable tokens |
| **Tool Use** | AI calling external tools/APIs |
| **Transformer** | Neural network architecture behind LLMs |
| **vLLM** | High-throughput LLM serving engine |

---

> 💡 **Pro Tip for Interviews:** Don't just memorize definitions — for each concept, have a **story**: "I built X using Y, which solved Z problem." That's what interviewers remember.

> **Next Steps:** Build small projects for each day's topic. A GitHub repo with "AI Agent using CrewAI" or "RAG with Evaluation Pipeline" will impress recruiters more than any certificate.

---

*Generated with current market research as of February 2026. The AI landscape evolves weekly — keep learning!*
