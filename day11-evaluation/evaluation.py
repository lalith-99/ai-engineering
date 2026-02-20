"""
Day 11: LLM evaluation — testing LLM outputs for quality.

Implements basic eval patterns:
1. Exact match / contains checks
2. LLM-as-judge (use a model to grade another model's output)
3. Factual consistency scoring
4. Batch evaluation with metrics

Usage:
    python evaluation.py
"""

import os
import json
from dataclasses import dataclass, field
from openai import OpenAI


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


# ========== TEST CASES ==========

@dataclass
class TestCase:
    input: str
    expected: str
    category: str


@dataclass
class EvalResult:
    test_case: TestCase
    actual_output: str
    passed: bool
    score: float  # 0.0 to 1.0
    method: str
    details: str = ""


TEST_CASES = [
    TestCase(
        input="What is the capital of France?",
        expected="Paris",
        category="factual",
    ),
    TestCase(
        input="Is Python dynamically typed?",
        expected="Yes",
        category="factual",
    ),
    TestCase(
        input="Explain what a load balancer does in one sentence.",
        expected="distributes incoming traffic across multiple servers",
        category="explanation",
    ),
    TestCase(
        input="What HTTP status code means 'Not Found'?",
        expected="404",
        category="factual",
    ),
    TestCase(
        input="Name one benefit of caching.",
        expected="reduces latency",
        category="explanation",
    ),
]


# ========== EVAL METHODS ==========

def eval_contains(actual: str, expected: str) -> tuple[bool, float]:
    """Simple: does the output contain the expected answer?"""
    passed = expected.lower() in actual.lower()
    return passed, 1.0 if passed else 0.0


def eval_llm_judge(client: OpenAI, question: str, expected: str, actual: str) -> tuple[bool, float, str]:
    """Use an LLM to judge if the answer is correct."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evaluation judge. Given a question, expected answer, "
                    "and actual answer, score the actual answer from 0.0 to 1.0. "
                    "Return JSON: {\"score\": float, \"correct\": bool, \"reason\": string}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Expected: {expected}\n"
                    f"Actual: {actual}\n"
                    f"Score the actual answer."
                ),
            },
        ],
    )

    result = json.loads(response.choices[0].message.content)
    return result.get("correct", False), result.get("score", 0.0), result.get("reason", "")


# ========== GENERATE + EVALUATE ==========

def generate_answer(client: OpenAI, question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=100,
        messages=[
            {"role": "system", "content": "Answer concisely in 1-2 sentences."},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


def run_eval_suite(use_llm_judge: bool = False):
    client = get_client()
    results = []

    print(f"Running {len(TEST_CASES)} test cases...")
    print(f"Method: {'LLM-as-judge' if use_llm_judge else 'contains-check'}\n")

    for i, tc in enumerate(TEST_CASES):
        actual = generate_answer(client, tc.input)

        if use_llm_judge:
            passed, score, reason = eval_llm_judge(client, tc.input, tc.expected, actual)
            method = "llm-judge"
        else:
            passed, score = eval_contains(actual, tc.expected)
            reason = ""
            method = "contains"

        result = EvalResult(
            test_case=tc,
            actual_output=actual,
            passed=passed,
            score=score,
            method=method,
            details=reason,
        )
        results.append(result)

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {tc.input}")
        print(f"         Expected: {tc.expected}")
        print(f"         Got: {actual[:80]}")
        if reason:
            print(f"         Judge: {reason}")
        print()

    # Summary
    passed_count = sum(1 for r in results if r.passed)
    avg_score = sum(r.score for r in results) / len(results) if results else 0
    print("=" * 50)
    print(f"Results: {passed_count}/{len(results)} passed")
    print(f"Average score: {avg_score:.2f}")
    print(f"Pass rate: {passed_count / len(results) * 100:.0f}%")

    # Per-category breakdown
    categories = set(r.test_case.category for r in results)
    for cat in categories:
        cat_results = [r for r in results if r.test_case.category == cat]
        cat_pass = sum(1 for r in cat_results if r.passed)
        print(f"  {cat}: {cat_pass}/{len(cat_results)}")

    return results


if __name__ == "__main__":
    import sys
    use_judge = "--judge" in sys.argv
    run_eval_suite(use_llm_judge=use_judge)
