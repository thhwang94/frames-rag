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
7. Evaluation with GPT-5-mini
"""

import ast
import json
import os
import re
import time
import threading
import queue
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from embedder import Embedder
from prompts import get_messages_for_llm, format_retrieval_results
# from query_decomposer import decompose_if_needed  # V7: disabled, caused regression

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HF_ACCESS_TOKEN = os.environ.get("HF_ACCESS_TOKEN")


client = OpenAI(api_key=OPENAI_API_KEY)


class RAGPipeline:
    """
    Complete RAG pipeline for FRAMES benchmark
    """

    def __init__(
        self,
        cache_dir: str = "cache",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        top_k: int = 10,
        max_context_length: int = 7500
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

        # Cross-encoder reranker (V8)
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("  - Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"  - Reranker: disabled ({e})")
            self.reranker = None

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
        if self.reranker:
            # Over-fetch for reranking, then blend scores
            fetch_k = self.top_k * 3
            results = self.retriever.retrieve(question, top_k=fetch_k)

            if verbose:
                print(f"  Reranking {len(results)} candidates...")

            # Compute cross-encoder scores
            pairs = [(question, r.chunk.text) for r in results]
            rerank_scores = self.reranker.predict(pairs)

            # Normalize both scores to [0,1] for blending
            bi_scores = [r.score for r in results]
            bi_min, bi_max = min(bi_scores), max(bi_scores)
            ce_min, ce_max = float(min(rerank_scores)), float(max(rerank_scores))

            for i, r in enumerate(results):
                bi_norm = (r.score - bi_min) / (bi_max - bi_min + 1e-8)
                ce_norm = (float(rerank_scores[i]) - ce_min) / (ce_max - ce_min + 1e-8)
                # Blend: 60% bi-encoder + 40% cross-encoder
                r.score = 0.6 * bi_norm + 0.4 * ce_norm

            results.sort(key=lambda x: x.score, reverse=True)

            # Source-balanced selection for multi-hop coverage
            results = self._source_balanced_select(results, self.top_k)
        else:
            results = self.retriever.retrieve(question, top_k=self.top_k)

        if verbose:
            sources = set(r.chunk.source_url.split('/')[-1] for r in results)
            print(f"  Retrieved {len(results)} chunks from {len(sources)} sources")

        # Step 5: Format context
        context = format_retrieval_results(results, self.max_context_length)

        return context, results

    def _compute_bm25_scores(self, query: str, results: List) -> List[float]:
        """Compute BM25 scores for retrieved chunks against the query."""
        from rank_bm25 import BM25Okapi

        # Tokenize chunks and query (simple whitespace + lowercasing)
        chunk_texts = [r.chunk.text.lower().split() for r in results]
        query_tokens = query.lower().split()

        if not chunk_texts:
            return []

        bm25 = BM25Okapi(chunk_texts)
        scores = bm25.get_scores(query_tokens)
        return scores.tolist()

    def _source_balanced_select(self, sorted_results: List, top_k: int) -> List:
        """
        Select top-k results ensuring source diversity for multi-hop QA.
        Guarantees at least 1 chunk from each source article (if available).
        """
        from collections import defaultdict

        if len(sorted_results) <= top_k:
            return sorted_results

        # Group by source
        by_source = defaultdict(list)
        for r in sorted_results:
            by_source[r.chunk.source_url].append(r)

        selected = []
        selected_set = set()

        # Phase 1: Take best chunk from each source (ensures multi-hop coverage)
        for source_url in by_source:
            best = by_source[source_url][0]  # Already sorted by score
            selected.append(best)
            selected_set.add(id(best))

        # Phase 2: Fill remaining slots with top-scoring chunks
        remaining = top_k - len(selected)
        if remaining > 0:
            for r in sorted_results:
                if id(r) not in selected_set:
                    selected.append(r)
                    selected_set.add(id(r))
                    remaining -= 1
                    if remaining <= 0:
                        break

        # Sort final selection by score
        selected.sort(key=lambda x: x.score, reverse=True)
        return selected[:top_k]

    def get_prompt_messages(
        self,
        question: str,
        wiki_links: List[str],
        verbose: bool = False,
        model: str = "gpt-4o-mini"
    ) -> List[Dict]:
        """
        Get formatted messages for LLM API call

        Args:
            question: The question to answer
            wiki_links: List of Wikipedia article URLs
            verbose: Whether to print debug info
            model: Model name (affects prompt style)

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

        return get_messages_for_llm(question, results, self.max_context_length, model)


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


_file_lock = threading.Lock()


def save_result_threadsafe(filename: str, result: Dict):
    """Thread-safe version of save_result using file lock"""
    with _file_lock:
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
    is_new_api = "gpt-5" in model or model.startswith("o")
    api_client = client

    kwargs = {
        "model": model,
        "messages": messages,
        "max_completion_tokens" if is_new_api else "max_tokens": 2000 if is_new_api else 1000,
    }
    if not is_new_api:
        kwargs["temperature"] = 0.3  # gpt-5 only supports default temperature
    # Retry up to 2 times if empty response
    for attempt in range(3):
        try:
            response = api_client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
        except Exception as e:
            if attempt == 2:
                print(f"  LLM error (attempt {attempt+1}): {e}")

    # Fallback to gpt-4o-mini if model returns empty
    if is_new_api:
        fallback = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, max_tokens=1000, temperature=0.3
        )
        content = fallback.choices[0].message.content
        return content.strip() if content else ""
    return ""


def extract_final_answer(response: str) -> str:
    """Extract the concise final answer from a response."""
    # Try "ANSWER: ..." pattern
    match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Try "FINAL ANSWER: ..." pattern
    match = re.search(r'FINAL ANSWER:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: last non-empty line
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return lines[-1] if lines else response


def get_llm_response_with_voting(messages: List[Dict], model: str, n: int = 3) -> str:
    """
    Generate n responses and return the one whose final answer
    appears most frequently (majority vote).
    Used for reasoning models where temperature=1.0 is forced.
    """
    responses = []
    for _ in range(n):
        resp = get_llm_response(messages, model)
        if resp:
            responses.append(resp)

    if not responses:
        return ""
    if len(responses) == 1:
        return responses[0]

    # Extract and normalize final answers for voting
    answers = [extract_final_answer(r) for r in responses]
    normalized = [re.sub(r'[^\w\s]', '', a.lower()).strip() for a in answers]

    counts = Counter(normalized)
    winner_norm = counts.most_common(1)[0][0]

    # Return the full response corresponding to the winning answer
    for i, norm in enumerate(normalized):
        if norm == winner_norm:
            return responses[i]

    return responses[0]


def evaluate_response(
    question: str, llm_response: str, ground_truth: str
) -> Dict[str, str]:
    """Evaluate LLM response against ground truth using GPT-5-mini"""
    evaluation_prompt = f"""Compare the predicted answer with the ground truth. Determine if the ground truth is present in the prediction. Focus on meaning, not exact wording.

Question: {question}
Predicted Answer: {llm_response}
Ground Truth Answer: {ground_truth}

Respond with:
Explanation: (brief reason)
Decision: TRUE or FALSE"""

    evaluation_response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are an evaluation assistant."},
            {"role": "user", "content": evaluation_prompt},
        ],
        max_completion_tokens=300,
    )

    evaluation_text = evaluation_response.choices[0].message.content.strip()

    lines = evaluation_text.split("\n")
    decision = "FALSE"
    explanation = ""
    for line in lines:
        if line.startswith("Decision:"):
            decision = line.split(":")[1].strip().upper()
        elif line.startswith("Explanation:"):
            explanation = line.split(":", 1)[1].strip()

    return {"decision": decision, "explanation": explanation}


def create_worker_pipelines(base_pipeline: RAGPipeline, n: int) -> List[RAGPipeline]:
    """
    Create n worker pipelines with independent mutable state.
    Shares: chunker (stateless), reranker model (read-only inference), embedder model (read-only inference)
    Independent per worker: wiki_fetcher cache, embedder cache, retriever state
    """
    # Ensure the base embedder model is loaded before cloning
    base_pipeline.embedder._get_model()

    pipelines = []
    for i in range(n):
        wp = RAGPipeline.__new__(RAGPipeline)
        wp.top_k = base_pipeline.top_k
        wp.max_context_length = base_pipeline.max_context_length
        wp.wiki_fetcher = WikipediaFetcher(cache_dir=base_pipeline.wiki_fetcher.cache_dir)
        wp.chunker = base_pipeline.chunker  # shared, stateless

        # Independent embedder with own cache dict, sharing the heavy model
        worker_embedder = Embedder(
            model_name=base_pipeline.embedder.model_name,
            cache_dir=base_pipeline.embedder.cache_dir
        )
        worker_embedder.model = base_pipeline.embedder.model  # share loaded model
        wp.embedder = worker_embedder

        wp.reranker = base_pipeline.reranker  # shared, read-only inference
        wp.retriever = SemanticRetriever(embedder=wp.embedder, chunker=wp.chunker)
        pipelines.append(wp)
    return pipelines


def parse_wiki_links(wiki_links_raw) -> List[str]:
    """Parse wiki_links from dataset item (may be string or list)"""
    if isinstance(wiki_links_raw, str):
        try:
            return ast.literal_eval(wiki_links_raw)
        except (ValueError, SyntaxError):
            return []
    return wiki_links_raw if wiki_links_raw else []


def process_single_item(pipeline: RAGPipeline, item: Dict, model: str, verbose: bool) -> Dict:
    """Process a single dataset item through the full RAG pipeline (thread-safe)"""
    index = int(item["Unnamed: 0"])
    question = item["Prompt"]
    wiki_links = parse_wiki_links(item["wiki_links"])

    if verbose:
        print(f"\n[{index}] Processing: {question[:80]}...")

    messages = pipeline.get_prompt_messages(question, wiki_links, verbose, model)
    llm_response = get_llm_response(messages, model)
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

    if verbose:
        print(f"  [{index}] Decision: {evaluation['decision']}")

    return result


def collect_prompts(pipeline: RAGPipeline, item: Dict, model: str, verbose: bool) -> Dict:
    """Run RAG retrieval only and return prompt messages (no LLM call)"""
    index = int(item["Unnamed: 0"])
    question = item["Prompt"]
    wiki_links = parse_wiki_links(item["wiki_links"])
    if verbose:
        print(f"\n[{index}] Collecting prompt: {question[:80]}...")
    messages = pipeline.get_prompt_messages(question, wiki_links, verbose, model)
    return {
        "index": index, "question": question, "messages": messages,
        "ground_truth": item["Answer"], "reasoning_type": item["reasoning_types"],
    }


def create_batch_jsonl(prompts: List[Dict], model: str, filepath: str):
    """Create JSONL file for OpenAI Batch API"""
    is_new_api = "gpt-5" in model or model.startswith("o")
    with open(filepath, 'w') as f:
        for p in prompts:
            if is_new_api:
                body = {"model": model, "messages": p["messages"], "max_completion_tokens": 2000}
            else:
                body = {"model": model, "messages": p["messages"], "max_tokens": 1000, "temperature": 0.3}
            request = {
                "custom_id": f"idx-{p['index']}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(request) + '\n')
    print(f"  Batch JSONL created: {filepath} ({len(prompts)} requests)")


def submit_batch(filepath: str) -> str:
    """Upload file and create batch job, return batch ID"""
    with open(filepath, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    print(f"  File uploaded: {batch_file.id}")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Batch created: {batch.id}")
    return batch.id


def wait_for_batch(batch_id: str, poll_interval: int = 30) -> str:
    """Poll batch until completion, return output file ID"""
    while True:
        batch = client.batches.retrieve(batch_id)
        completed = batch.request_counts.completed if batch.request_counts else 0
        total = batch.request_counts.total if batch.request_counts else 0
        print(f"  Batch status: {batch.status} ({completed}/{total})")
        if batch.status == "completed":
            return batch.output_file_id
        if batch.status in ("failed", "expired", "cancelled"):
            if batch.errors:
                for err in batch.errors.data[:5]:
                    print(f"    Error: {err.message}")
            raise RuntimeError(f"Batch {batch.status}: {batch_id}")
        time.sleep(poll_interval)


def download_batch_results(output_file_id: str) -> Dict[str, str]:
    """Download batch results, return {custom_id: response_text}"""
    content = client.files.content(output_file_id).text
    results = {}
    for line in content.strip().split('\n'):
        obj = json.loads(line)
        custom_id = obj["custom_id"]
        if obj.get("error"):
            print(f"  Batch error for {custom_id}: {obj['error']}")
            results[custom_id] = ""
            continue
        choices = obj["response"]["body"].get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""
        results[custom_id] = text.strip() if text else ""
    return results


def main_batch(model: str, start: int, end: int, verbose: bool, version: str, workers: int):
    """Batch API mode: 2-pass (retrieval → batch generation → evaluation)"""
    rag_pipeline = get_rag_pipeline()
    dataset = load_dataset("google/frames-benchmark", split="test", token=HF_ACCESS_TOKEN)
    dataset = [item for item in dataset if start <= int(item["Unnamed: 0"]) < end]

    version_suffix = f"_{version}" if version else ""
    filename = f"evaluation_results_{model.replace('/', '_')}{version_suffix}.json"
    existing_results = load_existing_results(filename)
    processed_indices = set(int(r.get("index", -1)) for r in existing_results)
    items_to_process = [item for item in dataset if int(item["Unnamed: 0"]) not in processed_indices]

    print(f"\n[Batch Mode] Starting evaluation:")
    print(f"  Model: {model}")
    print(f"  Range: {start}-{end}")
    print(f"  Already processed: {len(processed_indices)}")
    print(f"  Remaining: {len(items_to_process)}")
    print(f"  Results file: {filename}\n")

    if not items_to_process:
        print("All items already processed.")
        return

    # === Pass 1: Collect prompts (RAG retrieval only) ===
    print("=== Pass 1: RAG Retrieval + Prompt Collection ===")
    prompts = []
    if workers > 1:
        worker_pipelines = create_worker_pipelines(rag_pipeline, workers)
        pipeline_pool = queue.Queue()
        for wp in worker_pipelines:
            pipeline_pool.put(wp)

        def collect_with_pool(item):
            pipeline = pipeline_pool.get()
            try:
                return collect_prompts(pipeline, item, model, verbose)
            finally:
                pipeline_pool.put(pipeline)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(collect_with_pool, item): item for item in items_to_process}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting prompts"):
                prompts.append(future.result())
    else:
        for item in tqdm(items_to_process, desc="Collecting prompts"):
            prompts.append(collect_prompts(rag_pipeline, item, model, verbose))

    # Sort by index for consistent ordering
    prompts.sort(key=lambda p: p["index"])
    print(f"  Collected {len(prompts)} prompts\n")

    # === Pass 2: Submit batch and wait ===
    print("=== Pass 2: Batch API Submission ===")
    jsonl_path = f"batch_input_{model.replace('/', '_')}{version_suffix}.jsonl"
    create_batch_jsonl(prompts, model, jsonl_path)
    batch_id = submit_batch(jsonl_path)
    print(f"  Waiting for batch completion (polling every 30s)...")
    output_file_id = wait_for_batch(batch_id)
    batch_results = download_batch_results(output_file_id)
    print(f"  Received {len(batch_results)} responses\n")

    # === Pass 3: Evaluate responses ===
    print("=== Pass 3: Evaluation ===")
    prompt_map = {p["index"]: p for p in prompts}
    for custom_id, llm_response in tqdm(batch_results.items(), desc="Evaluating"):
        index = int(custom_id.split("-")[1])
        p = prompt_map.get(index)
        if not p:
            continue
        if not llm_response:
            llm_response = "(empty response)"
        evaluation = evaluate_response(p["question"], llm_response, p["ground_truth"])
        result = {
            "index": index,
            "prompt": p["question"],
            "ground_truth": p["ground_truth"],
            "llm_response": llm_response,
            "evaluation_decision": evaluation["decision"],
            "evaluation_explanation": evaluation["explanation"],
            "reasoning_type": p["reasoning_type"],
        }
        save_result(filename, result)

    # Cleanup batch input file
    os.remove(jsonl_path)

    # Print summary (reuse existing logic)
    results = load_existing_results(filename)
    range_results = [r for r in results if start <= r.get("index", -1) < end]
    if not range_results:
        print("No results to summarize.")
        return
    total_samples = len(range_results)
    correct_answers = sum(1 for r in range_results if r["evaluation_decision"] == "TRUE")
    accuracy = correct_answers / total_samples
    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY (Batch Mode)")
    print(f"{'='*50}")
    print(f"Model: {model}")
    print(f"Range: {start}-{end}")
    print(f"Total samples: {total_samples}")
    print(f"Correct answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.2%}")
    reasoning_types = set(r["reasoning_type"] for r in range_results)
    print(f"\nAccuracy by reasoning type:")
    for rt in sorted(reasoning_types):
        rt_samples = [r for r in range_results if r["reasoning_type"] == rt]
        rt_correct = sum(1 for r in rt_samples if r["evaluation_decision"] == "TRUE")
        rt_accuracy = rt_correct / len(rt_samples) if rt_samples else 0
        print(f"  {rt}: {rt_accuracy:.2%} ({rt_correct}/{len(rt_samples)})")


def main(model: str, start: int = 0, end: int = 100, verbose: bool = False, version: str = "", workers: int = 1, batch: bool = False):
    """
    Main evaluation function

    Args:
        model: LLM model to use for generation
        start: Start index (inclusive)
        end: End index (exclusive)
        verbose: Whether to print detailed progress
        version: Version suffix for result file
        workers: Number of concurrent workers (1=sequential)
        batch: Use OpenAI Batch API for generation (50% cost savings)
    """
    # Batch mode: 2-pass processing
    if batch:
        main_batch(model, start, end, verbose, version, workers)
        return

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

    # Use set of processed indices for resume (supports out-of-order completion)
    processed_indices = set(int(r.get("index", -1)) for r in existing_results)

    # Filter out already-processed items
    items_to_process = [
        item for item in dataset
        if int(item["Unnamed: 0"]) not in processed_indices
    ]

    print(f"\nStarting evaluation:")
    print(f"  Model: {model}")
    print(f"  Range: {start}-{end}")
    print(f"  Already processed: {len(processed_indices)}")
    print(f"  Remaining: {len(items_to_process)}")
    print(f"  Workers: {workers}")
    print(f"  Results file: {filename}\n")

    if not items_to_process:
        print("All items already processed.")
    elif workers <= 1:
        # Sequential processing (original behavior)
        for item in tqdm(items_to_process, desc="Processing samples"):
            result = process_single_item(rag_pipeline, item, model, verbose)
            save_result(filename, result)
    else:
        # Parallel processing with worker pipeline pool
        print(f"Creating {workers} worker pipelines (sharing models)...")
        worker_pipelines = create_worker_pipelines(rag_pipeline, workers)

        # Pipeline pool: each worker thread checks out a pipeline, uses it, returns it
        pipeline_pool = queue.Queue()
        for wp in worker_pipelines:
            pipeline_pool.put(wp)

        def process_with_pool(item):
            pipeline = pipeline_pool.get()
            try:
                return process_single_item(pipeline, item, model, verbose)
            finally:
                pipeline_pool.put(pipeline)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for item in items_to_process:
                future = executor.submit(process_with_pool, item)
                futures[future] = int(item["Unnamed: 0"])

            with tqdm(total=len(futures), desc="Processing samples") as pbar:
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        result = future.result()
                        save_result_threadsafe(filename, result)
                    except Exception as e:
                        print(f"\n  Error processing index {index}: {e}")
                    pbar.update(1)

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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent workers (default: 1, sequential)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use OpenAI Batch API for generation (50%% cost savings, OpenAI models only)"
    )
    args = parser.parse_args()

    main(args.model, args.start, args.end, args.verbose, args.version, args.workers, args.batch)
