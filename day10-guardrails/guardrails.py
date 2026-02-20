"""
Day 10: Structured output with Pydantic + basic guardrails.

Shows how to get reliable structured data from LLMs using:
1. Pydantic models for schema validation
2. OpenAI's response_format for JSON mode
3. Input/output guardrails (content filtering, length checks)

Usage:
    python guardrails.py extract "John Doe, Software Engineer at Google, john@google.com, 5 years experience"
    python guardrails.py classify "My order hasn't arrived and it's been 2 weeks. Very frustrated."
    python guardrails.py moderate "Some text to check for safety"
"""

import os
import sys
import json
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI

try:
    from pydantic import BaseModel, Field, field_validator
except ImportError:
    raise SystemExit("Pydantic not installed. Run: pip install pydantic")


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


# ========== PYDANTIC MODELS ==========

class ContactInfo(BaseModel):
    """Extracted contact information."""
    name: str = Field(description="Full name")
    title: Optional[str] = Field(default=None, description="Job title")
    company: Optional[str] = Field(default=None, description="Company name")
    email: Optional[str] = Field(default=None, description="Email address")
    years_experience: Optional[int] = Field(default=None, description="Years of experience")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if v and "@" not in v:
            raise ValueError(f"Invalid email: {v}")
        return v


class TicketClassification(BaseModel):
    """Customer support ticket classification."""
    category: str = Field(description="One of: billing, shipping, technical, account, other")
    priority: str = Field(description="One of: low, medium, high, urgent")
    sentiment: str = Field(description="One of: positive, neutral, negative")
    summary: str = Field(description="One sentence summary", max_length=100)
    needs_human: bool = Field(description="Whether this needs human review")


class ModerationResult(BaseModel):
    """Content moderation result."""
    is_safe: bool
    categories_flagged: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


# ========== GUARDRAILS ==========

def check_input_length(text: str, max_chars: int = 5000) -> tuple[bool, str]:
    if len(text) > max_chars:
        return False, f"Input too long: {len(text)} chars (max {max_chars})"
    if len(text.strip()) == 0:
        return False, "Input is empty"
    return True, "OK"


def check_pii_in_output(output: str) -> list[str]:
    """Basic PII detection in LLM output (not for input — input already has PII by design)."""
    warnings = []
    import re
    # SSN pattern
    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', output):
        warnings.append("Possible SSN detected in output")
    # Credit card pattern
    if re.search(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', output):
        warnings.append("Possible credit card number in output")
    return warnings


# ========== EXTRACTION ==========

def extract_contact(client: OpenAI, text: str) -> ContactInfo:
    ok, msg = check_input_length(text)
    if not ok:
        raise ValueError(msg)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract contact info from the text. Return JSON with fields: "
                    "name, title, company, email, years_experience. "
                    "Use null for missing fields."
                ),
            },
            {"role": "user", "content": text},
        ],
    )

    raw = json.loads(response.choices[0].message.content)
    return ContactInfo(**raw)


def classify_ticket(client: OpenAI, text: str) -> TicketClassification:
    ok, msg = check_input_length(text)
    if not ok:
        raise ValueError(msg)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify this support ticket. Return JSON with: "
                    "category (billing/shipping/technical/account/other), "
                    "priority (low/medium/high/urgent), "
                    "sentiment (positive/neutral/negative), "
                    "summary (one sentence, max 100 chars), "
                    "needs_human (true/false)."
                ),
            },
            {"role": "user", "content": text},
        ],
    )

    raw = json.loads(response.choices[0].message.content)
    return TicketClassification(**raw)


def moderate_content(client: OpenAI, text: str) -> ModerationResult:
    """Use OpenAI's moderation endpoint + structured output."""
    # Step 1: OpenAI moderation API (free, fast)
    mod = client.moderations.create(input=text)
    result = mod.results[0]

    flagged_categories = [
        cat for cat, flagged in result.categories.model_dump().items() if flagged
    ]

    return ModerationResult(
        is_safe=not result.flagged,
        categories_flagged=flagged_categories,
        confidence=max(result.category_scores.model_dump().values()),
        explanation="Flagged by moderation API" if result.flagged else "Content appears safe",
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python guardrails.py <extract|classify|moderate> <text>")
        sys.exit(1)

    mode = sys.argv[1]
    text = " ".join(sys.argv[2:])
    client = get_client()

    if mode == "extract":
        result = extract_contact(client, text)
        print(result.model_dump_json(indent=2))

    elif mode == "classify":
        result = classify_ticket(client, text)
        print(result.model_dump_json(indent=2))
        pii_warnings = check_pii_in_output(result.summary)
        if pii_warnings:
            print(f"\nWarnings: {pii_warnings}")

    elif mode == "moderate":
        result = moderate_content(client, text)
        print(result.model_dump_json(indent=2))

    else:
        print(f"Unknown mode: {mode}. Use extract, classify, or moderate.")
