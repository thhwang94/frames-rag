"""
Query Decomposition Module
Decomposes complex multi-hop questions into simpler sub-questions
"""

from typing import List, Tuple
from openai import OpenAI
import os
import re

# Use the same client as run.py
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


DECOMPOSITION_PROMPT = """You are an expert at breaking down complex questions into simpler sub-questions.

Given a complex question that may require multiple pieces of information, decompose it into 2-4 simpler sub-questions that can be answered independently.

Rules:
1. Each sub-question should be self-contained and searchable
2. Include entity names explicitly in each sub-question
3. If the question is already simple (single fact needed), return just that question
4. Order sub-questions logically (facts needed first, then comparisons/calculations)

Examples:

Question: "How old was Albert Einstein when Isaac Newton died?"
Sub-questions:
1. When was Albert Einstein born?
2. When did Isaac Newton die?

Question: "What is the population of the capital of France?"
Sub-questions:
1. What is the capital of France?
2. What is the population of Paris?

Question: "Who was the US president when the Berlin Wall fell?"
Sub-questions:
1. When did the Berlin Wall fall?
2. Who was the US president in 1989?

Question: "Did Einstein or Newton live longer?"
Sub-questions:
1. When was Albert Einstein born and when did he die?
2. When was Isaac Newton born and when did he die?

Now decompose this question:
Question: "{question}"

Sub-questions (one per line, numbered):"""


def decompose_question(question: str, client: OpenAI = None) -> List[str]:
    """
    Decompose a complex question into simpler sub-questions

    Args:
        question: The complex question to decompose
        client: OpenAI client (creates new one if None)

    Returns:
        List of sub-questions (may be just the original if simple)
    """
    if client is None:
        client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that decomposes complex questions."},
                {"role": "user", "content": DECOMPOSITION_PROMPT.format(question=question)}
            ],
            max_tokens=300,
            temperature=0.0,  # Deterministic for consistency
        )

        result = response.choices[0].message.content.strip()

        # Parse numbered sub-questions
        sub_questions = []
        for line in result.split('\n'):
            line = line.strip()
            # Match patterns like "1. question" or "1) question" or just "question"
            match = re.match(r'^[\d]+[.\)]\s*(.+)$', line)
            if match:
                sub_questions.append(match.group(1).strip())
            elif line and not line.startswith('#'):
                # Also accept non-numbered lines if they look like questions
                if '?' in line or len(line) > 20:
                    sub_questions.append(line)

        # If no sub-questions extracted, return original
        if not sub_questions:
            return [question]

        # Limit to 4 sub-questions max
        return sub_questions[:4]

    except Exception as e:
        print(f"  Warning: Query decomposition failed: {e}")
        return [question]


def should_decompose(question: str) -> bool:
    """
    Heuristic to determine if a question should be decomposed

    Args:
        question: The question text

    Returns:
        True if the question likely needs decomposition
    """
    question_lower = question.lower()

    # Multi-hop indicators
    multi_hop_patterns = [
        # Comparison patterns
        "who was older", "who was younger", "which was first", "which came before",
        "who lived longer", "who died first", "who was born first",
        "more than", "less than", "difference between",
        # Temporal reasoning
        "when.*was.*born.*when", "how old was.*when",
        "at the time of", "during the", "before.*after",
        # Chained facts
        "the.*of the.*of", "capital of.*population",
        "president when", "leader when", "who was.*when.*happened",
        # Calculations
        "how many years", "how much older", "how much younger",
        "total.*and.*and", "sum of", "combined",
    ]

    for pattern in multi_hop_patterns:
        if re.search(pattern, question_lower):
            return True

    # Also decompose if question has multiple question marks or "and"
    if question.count('?') > 1:
        return True

    # Check for complex conjunctions
    if ' and ' in question_lower and len(question) > 80:
        return True

    return False


def decompose_if_needed(question: str, client: OpenAI = None) -> Tuple[List[str], bool]:
    """
    Decompose question only if it appears to be multi-hop

    Args:
        question: The question text
        client: OpenAI client

    Returns:
        Tuple of (sub_questions, was_decomposed)
    """
    if should_decompose(question):
        sub_questions = decompose_question(question, client)
        return sub_questions, len(sub_questions) > 1
    return [question], False


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Test questions
    test_questions = [
        "How old was Albert Einstein when Isaac Newton died?",
        "What is the capital of France?",
        "Who was the US president when the Berlin Wall fell?",
        "Did Einstein or Newton live longer?",
        "What year did the first person walk on the moon?",
        "How many years passed between the founding of Harvard and the Declaration of Independence?",
    ]

    client = OpenAI(api_key=OPENAI_API_KEY)

    for q in test_questions:
        print(f"\nQuestion: {q}")
        print(f"Should decompose: {should_decompose(q)}")

        if should_decompose(q):
            subs = decompose_question(q, client)
            print(f"Sub-questions:")
            for i, sub in enumerate(subs, 1):
                print(f"  {i}. {sub}")
