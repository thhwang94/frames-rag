"""
FRAMES RAG System - Main Pipeline
Modified from: https://github.com/codelion/optillm/blob/main/scripts/eval_frames_benchmark.py

Implements a complete RAG pipeline:
1. Wikipedia content fetching (parallel with caching)
2. Document chunking (300-600 tokens with overlap)
3. Semantic embedding (local model - free)
4. Semantic retrieval (cosine similarity)
5. Chain-of-Thought prompting
6. LLM response generation
7. Evaluation with Upstage Solar Pro2
"""

import ast
import json
import os
from typing import Dict, List, Optional
import argparse

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Local RAG modules
from wikipedia_fetcher import WikipediaFetcher
from chunker import DocumentChunker
from embedder import Embedder
from retriever import SemanticRetriever
from prompts import get_messages_for_llm, format_retrieval_results

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HF_ACCESS_TOKEN = os.environ.get("HF_ACCESS_TOKEN")
UPSTAGE_API_KEY = os.environ.get("UPSTAGE_API_KEY")
UPSTAGE_BASE_URL = os.environ.get("UPSTAGE_BASE_URL")


client = OpenAI(api_key=OPENAI_API_KEY)
eval_client = OpenAI(api_key=UPSTAGE_API_KEY, base_url=UPSTAGE_BASE_URL)


class RAGPipeline:
    """
    Complete RAG pipeline for FRAMES benchmark
    """

    def __init__(
        self,
        cache_dir: str = "cache",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        top_k: int = 8,
        max_context_length: int = 6000
    ):
        """
        Initialize the RAG pipeline

        Args:
            cache_dir: Directory for caching Wikipedia content
            embedding_model: Sentence transformer model for embeddings
            top_k: Number of chunks to retrieve per question
            max_context_length: Maximum context length in characters
        """
        self.top_k = top_k
        self.max_context_length = max_context_length

        # Initialize components
        print("Initializing RAG pipeline components...")
        self.wiki_fetcher = WikipediaFetcher(cache_dir=cache_dir)
        self.chunker = DocumentChunker(
            target_chunk_size=450,
            min_chunk_size=300,
            max_chunk_size=600,
            overlap_ratio=0.25
        )
        self.embedder = Embedder(model_name=embedding_model, cache_dir="embeddings")
        self.retriever = SemanticRetriever(embedder=self.embedder, chunker=self.chunker)

        print(f"RAG Pipeline initialized with:")
        print(f"  - Top-k retrieval: {top_k}")
        print(f"  - Max context: {max_context_length} chars")
        print(f"  - Embedding model: {embedding_model}")

    def process_question(
        self,
        question: str,
        wiki_links: List[str],
        verbose: bool = False
    ) -> tuple[str, List]:
        """
        Process a single question through the RAG pipeline

        Args:
            question: The question to answer
            wiki_links: List of Wikipedia article URLs
            verbose: Whether to print debug info

        Returns:
            Tuple of (formatted_context, retrieval_results)
        """
        # Step 1: Fetch Wikipedia content
        if verbose:
            print(f"  Fetching {len(wiki_links)} Wikipedia articles...")

        documents = self.wiki_fetcher.fetch_articles(wiki_links)

        if verbose:
            print(f"  Retrieved {len(documents)} articles")

        if not documents:
            return "No relevant Wikipedia content could be retrieved.", []

        # Step 2: Index documents and retrieve relevant chunks
        if verbose:
            print("  Indexing and retrieving chunks...")

        # Re-initialize retriever for this question's documents
        self.retriever.chunks = []
        self.retriever.chunk_embeddings = None
        num_chunks = self.retriever.index_documents(documents, show_progress=False)

        if verbose:
            print(f"  Created {num_chunks} chunks")

        # Step 3: Retrieve relevant chunks
        results = self.retriever.retrieve(question, top_k=self.top_k)

        if verbose:
            print(f"  Retrieved {len(results)} relevant chunks")

        # Step 4: Format context
        context = format_retrieval_results(results, self.max_context_length)

        return context, results

    def get_prompt_messages(
        self,
        question: str,
        wiki_links: List[str],
        verbose: bool = False
    ) -> List[Dict]:
        """
        Get formatted messages for LLM API call

        Args:
            question: The question to answer
            wiki_links: List of Wikipedia article URLs
            verbose: Whether to print debug info

        Returns:
            List of message dictionaries for API call
        """
        context, results = self.process_question(question, wiki_links, verbose)

        # If no results, use fallback prompt
        if not results:
            return [
                {"role": "system", "content": "You are a helpful assistant. Answer based on your knowledge."},
                {"role": "user", "content": f"Question: {question}\n\nNote: No Wikipedia content was available. Please answer based on your knowledge, or state if you cannot answer."}
            ]

        return get_messages_for_llm(question, results, self.max_context_length)


# Global RAG pipeline instance
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create the global RAG pipeline instance"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


def load_existing_results(filename: str) -> List[Dict]:
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_result(filename: str, result: Dict):
    results = load_existing_results(filename)
    results.append(result)
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)


def get_last_processed_index(results: List[Dict]) -> int:
    if not results:
        return -1
    return max(int(r.get("index", -1)) for r in results)


def get_llm_response(messages: List[Dict], model: str) -> str:
    """
    Get response from LLM using formatted messages

    Args:
        messages: List of message dictionaries
        model: Model name to use

    Returns:
        LLM response text
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1000,
        temperature=0.3,  # Lower temperature for more factual responses
    )
    return response.choices[0].message.content.strip()


def evaluate_response(
    question: str, llm_response: str, ground_truth: str
) -> Dict[str, str]:
    """
    Evaluate LLM response using Upstage Solar Pro2
    THIS FUNCTION MUST NOT BE MODIFIED
    """
    evaluation_prompt = f"""===Task===
I need your help in evaluating an answer provided by an LLM against a ground
truth answer. Your task is to determine if the ground truth answer is present in the LLM's
response. Please analyze the provided data and make a decision.
===Instructions===
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
2. Consider the substance of the answers - look for equivalent information or correct answers.
Do not focus on exact wording unless the exact wording is crucial to the meaning.
3. Your final decision should be based on whether the meaning and the vital facts of the
"Ground Truth Answer" are present in the "Predicted Answer:"
===Input Data===
- Question: {question}
- Predicted Answer: {llm_response}
- Ground Truth Answer: {ground_truth}
===Output Format===
Provide your final evaluation in the following format:
"Explanation:" (How you made the decision?)
"Decision:" ("TRUE" or "FALSE" )
Please proceed with the evaluation."""

    evaluation_response = eval_client.chat.completions.create(
        model="solar-pro2",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": evaluation_prompt},
        ],
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.3,
    )

    evaluation_text = evaluation_response.choices[0].message.content.strip()

    # Extract the decision and explanation
    lines = evaluation_text.split("\n")
    decision = "FALSE"
    explanation = ""
    for line in lines:
        if line.startswith("Decision:"):
            decision = line.split(":")[1].strip().upper()
        elif line.startswith("Explanation:"):
            explanation = line.split(":", 1)[1].strip()

    return {"decision": decision, "explanation": explanation}


def main(model: str, start: int = 0, end: int = 100, verbose: bool = False, version: str = ""):
    """
    Main evaluation function

    Args:
        model: LLM model to use for generation
        start: Start index (inclusive)
        end: End index (exclusive)
        verbose: Whether to print detailed progress
        version: Version suffix for result file
    """
    # Initialize RAG pipeline
    rag_pipeline = get_rag_pipeline()

    # Load the dataset
    dataset = load_dataset(
        "google/frames-benchmark", split="test", token=HF_ACCESS_TOKEN
    )

    # Filter to specified range
    dataset = [item for item in dataset if start <= int(item["Unnamed: 0"]) < end]

    # Create versioned filename
    version_suffix = f"_{version}" if version else ""
    filename = f"evaluation_results_{model.replace('/', '_')}{version_suffix}.json"
    existing_results = load_existing_results(filename)
    last_processed_index = get_last_processed_index(existing_results)

    print(f"\nStarting evaluation:")
    print(f"  Model: {model}")
    print(f"  Range: {start}-{end}")
    print(f"  Last processed: {last_processed_index}")
    print(f"  Results file: {filename}\n")

    for item in tqdm(dataset, desc="Processing samples"):
        index = int(item["Unnamed: 0"])
        if index <= last_processed_index:
            continue

        question = item["Prompt"]
        wiki_links_raw = item["wiki_links"]

        # Parse wiki_links - it may be a string representation of a list
        if isinstance(wiki_links_raw, str):
            try:
                wiki_links = ast.literal_eval(wiki_links_raw)
            except (ValueError, SyntaxError):
                wiki_links = []
        else:
            wiki_links = wiki_links_raw if wiki_links_raw else []

        if verbose:
            print(f"\n[{index}] Processing question: {question[:80]}...")

        # Get RAG-enhanced prompt messages
        messages = rag_pipeline.get_prompt_messages(question, wiki_links, verbose)

        # Get LLM response
        llm_response = get_llm_response(messages, model)

        # Evaluate response
        evaluation = evaluate_response(question, llm_response, item["Answer"])

        result = {
            "index": index,
            "prompt": question,
            "ground_truth": item["Answer"],
            "llm_response": llm_response,
            "evaluation_decision": evaluation["decision"],
            "evaluation_explanation": evaluation["explanation"],
            "reasoning_type": item["reasoning_types"],
        }

        save_result(filename, result)

        if verbose:
            print(f"  Answer: {llm_response[:100]}...")
            print(f"  Ground truth: {item['Answer']}")
            print(f"  Decision: {evaluation['decision']}")

    # Calculate and print summary statistics
    results = load_existing_results(filename)

    # Filter results to the specified range
    range_results = [r for r in results if start <= r.get("index", -1) < end]

    if not range_results:
        print("No results to summarize.")
        return

    total_samples = len(range_results)
    correct_answers = sum(1 for r in range_results if r["evaluation_decision"] == "TRUE")
    accuracy = correct_answers / total_samples

    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Model: {model}")
    print(f"Range: {start}-{end}")
    print(f"Total samples: {total_samples}")
    print(f"Correct answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.2%}")

    # Print accuracy by reasoning type
    reasoning_types = set(r["reasoning_type"] for r in range_results)
    print(f"\nAccuracy by reasoning type:")
    for rt in sorted(reasoning_types):
        rt_samples = [r for r in range_results if r["reasoning_type"] == rt]
        rt_correct = sum(1 for r in rt_samples if r["evaluation_decision"] == "TRUE")
        rt_accuracy = rt_correct / len(rt_samples) if rt_samples else 0
        print(f"  {rt}: {rt_accuracy:.2%} ({rt_correct}/{len(rt_samples)})")

    # Print cache statistics
    print(f"\nCache statistics:")
    print(f"  Wikipedia cache: {rag_pipeline.wiki_fetcher.get_cache_stats()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LLM performance on google/frames-benchmark with RAG"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        required=False,
        help="LLM model for generation (e.g., gpt-4o-mini, gpt-4o)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (inclusive)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=100,
        help="End index (exclusive)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="",
        help="Version suffix for result file (e.g., v3)"
    )
    args = parser.parse_args()

    main(args.model, args.start, args.end, args.verbose, args.version)
