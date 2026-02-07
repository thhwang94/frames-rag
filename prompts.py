"""
Prompt Templates Module
Contains Chain-of-Thought (CoT) prompts for multi-hop reasoning
"""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class PromptConfig:
    """Configuration for prompt generation"""
    max_context_length: int = 4000  # Maximum characters of context
    include_sources: bool = True
    reasoning_style: str = "step_by_step"  # step_by_step, concise, detailed


# System prompt for the LLM
SYSTEM_PROMPT = """You are an expert research assistant skilled at multi-hop reasoning.
Your task is to answer questions by carefully analyzing information from multiple Wikipedia sources.
Always base your answers on the provided context.
Be precise and factual - provide specific names, numbers, dates, or values as your final answer.
If asked for a calculation, show your work and give the exact result."""


# Chain-of-Thought prompt template
COT_PROMPT_TEMPLATE = """You are an expert at multi-hop reasoning and fact synthesis.

=== WIKIPEDIA CONTEXT ===
{context}

=== INSTRUCTIONS ===
Answer the question using ONLY the information provided above.
Think step by step:
1. Extract specific facts from each source that relate to the question
2. Connect facts across sources to build your reasoning
3. Perform any necessary calculations (dates, numbers, etc.)
4. State your FINAL ANSWER clearly

IMPORTANT: Your final answer must be specific and precise (e.g., exact names, numbers, dates).

=== QUESTION ===
{question}

=== REASONING ===
"""


# Simpler prompt for less complex questions
SIMPLE_PROMPT_TEMPLATE = """Based on the following Wikipedia information, answer the question.

=== WIKIPEDIA CONTEXT ===
{context}

=== QUESTION ===
{question}

=== ANSWER ===
"""


# Detailed reasoning prompt for complex multi-hop questions
DETAILED_COT_TEMPLATE = """You are an expert research assistant. Your task is to answer a complex question using information from multiple Wikipedia articles.

=== WIKIPEDIA SOURCES ===
{context}

=== TASK ===
Answer the following question by:
1. First, identify what specific information is needed
2. Extract relevant facts from each Wikipedia source
3. Connect the facts logically to build toward the answer
4. State your final answer clearly

Important: If any required information is missing from the sources, explicitly state what is missing.

=== QUESTION ===
{question}

=== STEP-BY-STEP REASONING ===
Step 1 - Understanding the question:
What do I need to find?

Step 2 - Extracting relevant facts:
"""


# Prompt for numerical/date reasoning
NUMERICAL_REASONING_TEMPLATE = """You are an expert at numerical and temporal reasoning.

=== WIKIPEDIA CONTEXT ===
{context}

=== QUESTION ===
{question}

=== INSTRUCTIONS ===
This question requires numerical or date-based reasoning.

Step 1 - Extract Data:
List ALL relevant numbers, dates, years, ages, or quantities from the context.
Format: [Entity]: [Value]

Step 2 - Identify Operation:
What calculation is needed? (subtraction for age/difference, addition for totals, comparison for rankings)

Step 3 - Calculate:
Show the arithmetic clearly:
- If comparing dates: Later year - Earlier year = Difference
- If calculating age: Death year - Birth year = Age
- If summing: Value1 + Value2 + ... = Total

Step 4 - Final Answer:
State the answer as a specific value (number, date, name, etc.)

=== SOLUTION ===
"""


def format_context(
    chunks: List[dict],
    max_length: int = 4000,
    include_source: bool = True
) -> str:
    """
    Format retrieved chunks into context string

    Args:
        chunks: List of dictionaries with 'text' and 'source_url' keys
        max_length: Maximum total character length
        include_source: Whether to include source URL in context

    Returns:
        Formatted context string
    """
    if not chunks:
        return "No relevant information found."

    context_parts = []
    current_length = 0

    for i, chunk in enumerate(chunks):
        text = chunk.get('text', chunk.get('chunk', {}).get('text', ''))
        source = chunk.get('source_url', chunk.get('chunk', {}).get('source_url', 'Unknown'))

        if include_source:
            # Extract title from URL for readability
            title = source.split('/')[-1].replace('_', ' ') if '/' in source else source
            chunk_text = f"[Source {i+1}: {title}]\n{text}\n"
        else:
            chunk_text = f"{text}\n"

        if current_length + len(chunk_text) > max_length:
            # Truncate if needed
            remaining = max_length - current_length
            if remaining > 100:  # Only add if meaningful content remains
                chunk_text = chunk_text[:remaining] + "..."
                context_parts.append(chunk_text)
            break

        context_parts.append(chunk_text)
        current_length += len(chunk_text)

    return "\n".join(context_parts)


def format_retrieval_results(
    results: List,
    max_length: int = 4000
) -> str:
    """
    Format RetrievalResult objects into context string

    Args:
        results: List of RetrievalResult objects
        max_length: Maximum total character length

    Returns:
        Formatted context string
    """
    chunks = []
    for result in results:
        chunks.append({
            'text': result.chunk.text,
            'source_url': result.chunk.source_url
        })
    return format_context(chunks, max_length)


def generate_prompt(
    question: str,
    context: str,
    prompt_type: str = "cot"
) -> str:
    """
    Generate a complete prompt for the LLM

    Args:
        question: The question to answer
        context: Formatted context from retrieval
        prompt_type: Type of prompt ("cot", "simple", "detailed", "numerical")

    Returns:
        Complete prompt string
    """
    templates = {
        "cot": COT_PROMPT_TEMPLATE,
        "simple": SIMPLE_PROMPT_TEMPLATE,
        "detailed": DETAILED_COT_TEMPLATE,
        "numerical": NUMERICAL_REASONING_TEMPLATE
    }

    template = templates.get(prompt_type, COT_PROMPT_TEMPLATE)
    return template.format(context=context, question=question)


def detect_question_type(question: str) -> str:
    """
    Detect the type of question to select appropriate prompt

    Args:
        question: The question text

    Returns:
        Question type: "numerical", "detailed", or "cot"
    """
    question_lower = question.lower()

    # Numerical indicators
    numerical_keywords = [
        "how many", "how much", "how old", "how long", "how far",
        "what year", "when was", "when did", "what date",
        "calculate", "total", "sum", "difference", "between",
        "older", "younger", "before", "after", "first", "last"
    ]

    for keyword in numerical_keywords:
        if keyword in question_lower:
            return "numerical"

    # Complex multi-hop indicators
    complex_keywords = [
        "relationship between", "compare", "connection",
        "how does", "why did", "explain how",
        "what happened when", "as a result of"
    ]

    for keyword in complex_keywords:
        if keyword in question_lower:
            return "detailed"

    # Default to CoT for most questions
    return "cot"


def create_rag_prompt(
    question: str,
    retrieval_results: List,
    max_context_length: int = 4000,
    auto_detect_type: bool = True
) -> tuple[str, str]:
    """
    Create a complete RAG prompt from question and retrieval results

    Args:
        question: The question to answer
        retrieval_results: List of RetrievalResult objects
        max_context_length: Maximum context length in characters
        auto_detect_type: Whether to auto-detect question type

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Format context
    context = format_retrieval_results(retrieval_results, max_context_length)

    # Detect question type if enabled
    if auto_detect_type:
        prompt_type = detect_question_type(question)
    else:
        prompt_type = "cot"

    # Generate prompt
    user_prompt = generate_prompt(question, context, prompt_type)

    return SYSTEM_PROMPT, user_prompt


def get_messages_for_llm(
    question: str,
    retrieval_results: List,
    max_context_length: int = 4000
) -> List[dict]:
    """
    Get formatted messages list for OpenAI-compatible API

    Args:
        question: The question to answer
        retrieval_results: List of RetrievalResult objects
        max_context_length: Maximum context length

    Returns:
        List of message dictionaries for API call
    """
    system_prompt, user_prompt = create_rag_prompt(
        question, retrieval_results, max_context_length
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


if __name__ == "__main__":
    # Test prompt generation
    print("Testing Prompt Templates...")

    # Sample retrieval results (mock)
    class MockChunk:
        def __init__(self, text, source_url):
            self.text = text
            self.source_url = source_url

    class MockResult:
        def __init__(self, text, source_url, score):
            self.chunk = MockChunk(text, source_url)
            self.score = score

    results = [
        MockResult(
            "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.",
            "https://en.wikipedia.org/wiki/Albert_Einstein",
            0.85
        ),
        MockResult(
            "Einstein was born in Ulm, Germany on March 14, 1879, and died in Princeton, New Jersey on April 18, 1955.",
            "https://en.wikipedia.org/wiki/Albert_Einstein",
            0.82
        )
    ]

    question = "How old was Einstein when he died?"

    # Detect question type
    q_type = detect_question_type(question)
    print(f"Detected question type: {q_type}")

    # Generate prompt
    system_prompt, user_prompt = create_rag_prompt(question, results)
    print(f"\n=== System Prompt ===\n{system_prompt}")
    print(f"\n=== User Prompt ===\n{user_prompt}")

    # Get messages format
    messages = get_messages_for_llm(question, results)
    print(f"\n=== Messages format ===")
    for msg in messages:
        print(f"Role: {msg['role']}, Content length: {len(msg['content'])} chars")
