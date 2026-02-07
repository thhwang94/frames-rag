"""
Retrieval Module
Performs semantic search to find relevant chunks for a query
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from chunker import Chunk, DocumentChunker
from embedder import Embedder


@dataclass
class RetrievalResult:
    """Represents a retrieval result with score"""
    chunk: Chunk
    score: float
    rank: int

    def __repr__(self):
        return f"RetrievalResult(rank={self.rank}, score={self.score:.4f}, source={self.chunk.source_url})"


class SemanticRetriever:
    """
    Performs semantic search over document chunks using cosine similarity
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        chunker: Optional[DocumentChunker] = None
    ):
        """
        Initialize the retriever

        Args:
            embedder: Embedder instance (creates default if None)
            chunker: DocumentChunker instance (creates default if None)
        """
        self.embedder = embedder or Embedder()
        self.chunker = chunker or DocumentChunker()

        # Storage for indexed documents
        self.chunks: List[Chunk] = []
        self.chunk_embeddings: Optional[np.ndarray] = None

    def index_documents(
        self,
        documents: dict[str, str],
        show_progress: bool = True
    ) -> int:
        """
        Index documents for retrieval

        Args:
            documents: Dictionary mapping URL to document text
            show_progress: Whether to show progress bar during embedding

        Returns:
            Number of chunks created
        """
        # Chunk documents
        self.chunks = self.chunker.chunk_documents(documents)

        if not self.chunks:
            self.chunk_embeddings = None
            return 0

        # Generate embeddings
        self.chunk_embeddings = self.embedder.embed_chunks(
            self.chunks,
            show_progress=show_progress
        )

        return len(self.chunks)

    def add_document(
        self,
        url: str,
        text: str,
        show_progress: bool = False
    ) -> int:
        """
        Add a single document to the index

        Args:
            url: Document URL
            text: Document text
            show_progress: Whether to show progress bar

        Returns:
            Number of new chunks added
        """
        # Chunk the new document
        new_chunks = self.chunker.chunk_document(text, url)

        if not new_chunks:
            return 0

        # Re-index chunks globally
        start_idx = len(self.chunks)
        for i, chunk in enumerate(new_chunks):
            chunk.chunk_index = start_idx + i

        # Generate embeddings for new chunks
        new_embeddings = self.embedder.embed_chunks(new_chunks, show_progress=show_progress)

        # Add to storage
        self.chunks.extend(new_chunks)

        if self.chunk_embeddings is None:
            self.chunk_embeddings = new_embeddings
        else:
            self.chunk_embeddings = np.vstack([self.chunk_embeddings, new_embeddings])

        return len(new_chunks)

    def _cosine_similarity(
        self,
        query_embedding: np.ndarray,
        chunk_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and all chunks

        Args:
            query_embedding: Query embedding vector
            chunk_embeddings: Matrix of chunk embeddings

        Returns:
            Array of similarity scores
        """
        # Embeddings are already normalized, so dot product = cosine similarity
        return np.dot(chunk_embeddings, query_embedding)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        source_urls: Optional[List[str]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k relevant chunks for a query

        Args:
            query: Query text
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            source_urls: Optional list of URLs to filter results (only return from these sources)

        Returns:
            List of RetrievalResult objects sorted by score
        """
        if not self.chunks or self.chunk_embeddings is None:
            return []

        # Embed the query
        query_embedding = self.embedder.embed_query(query)

        # Filter chunks by source if specified
        if source_urls:
            source_set = set(source_urls)
            indices = [i for i, c in enumerate(self.chunks) if c.source_url in source_set]

            if not indices:
                return []

            filtered_embeddings = self.chunk_embeddings[indices]
            filtered_chunks = [self.chunks[i] for i in indices]
        else:
            filtered_embeddings = self.chunk_embeddings
            filtered_chunks = self.chunks

        # Compute similarities
        similarities = self._cosine_similarity(query_embedding, filtered_embeddings)

        # Get top-k indices
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            # Partial sort for efficiency
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        # Build results
        results = []
        for rank, idx in enumerate(top_indices):
            score = float(similarities[idx])
            if score >= min_score:
                results.append(RetrievalResult(
                    chunk=filtered_chunks[idx],
                    score=score,
                    rank=rank + 1
                ))

        return results

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
        context_window: int = 1,
        source_urls: Optional[List[str]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve chunks with surrounding context

        Args:
            query: Query text
            top_k: Number of top results before adding context
            context_window: Number of adjacent chunks to include
            source_urls: Optional list of URLs to filter results

        Returns:
            List of RetrievalResult objects with context chunks merged
        """
        # Get initial results
        results = self.retrieve(query, top_k, source_urls=source_urls)

        if context_window <= 0:
            return results

        # Collect all chunk indices to include
        indices_to_include = set()
        for result in results:
            idx = result.chunk.chunk_index
            # Add adjacent chunks from same source
            for i in range(max(0, idx - context_window), idx + context_window + 1):
                if i < len(self.chunks) and self.chunks[i].source_url == result.chunk.source_url:
                    indices_to_include.add(i)

        # Build expanded results (keeping original ranking)
        expanded_results = []
        seen_indices = set()

        for result in results:
            if result.chunk.chunk_index not in seen_indices:
                expanded_results.append(result)
                seen_indices.add(result.chunk.chunk_index)

            # Add context chunks
            idx = result.chunk.chunk_index
            for i in range(max(0, idx - context_window), idx + context_window + 1):
                if i not in seen_indices and i in indices_to_include:
                    expanded_results.append(RetrievalResult(
                        chunk=self.chunks[i],
                        score=result.score * 0.8,  # Reduce score for context
                        rank=len(expanded_results) + 1
                    ))
                    seen_indices.add(i)

        return expanded_results

    def get_stats(self) -> dict:
        """Get retriever statistics"""
        if not self.chunks:
            return {"status": "empty", "num_chunks": 0}

        sources = set(c.source_url for c in self.chunks)
        return {
            "num_chunks": len(self.chunks),
            "num_sources": len(sources),
            "embedding_dimension": self.chunk_embeddings.shape[1] if self.chunk_embeddings is not None else 0,
            "sources": list(sources)
        }


def build_retriever_from_documents(
    documents: dict[str, str],
    model_name: str = "BAAI/bge-small-en-v1.5"
) -> SemanticRetriever:
    """
    Convenience function to build a retriever from documents

    Args:
        documents: Dictionary mapping URL to document text
        model_name: Embedding model name

    Returns:
        Configured SemanticRetriever
    """
    embedder = Embedder(model_name=model_name)
    retriever = SemanticRetriever(embedder=embedder)
    retriever.index_documents(documents)
    return retriever


def retrieve_for_question(
    question: str,
    documents: dict[str, str],
    top_k: int = 5,
    model_name: str = "BAAI/bge-small-en-v1.5"
) -> List[RetrievalResult]:
    """
    One-shot retrieval for a question

    Args:
        question: Question to answer
        documents: Dictionary mapping URL to document text
        top_k: Number of chunks to retrieve
        model_name: Embedding model name

    Returns:
        List of RetrievalResult objects
    """
    retriever = build_retriever_from_documents(documents, model_name)
    return retriever.retrieve(question, top_k=top_k)


if __name__ == "__main__":
    # Test the retriever
    print("Testing Semantic Retriever...")

    # Sample documents
    documents = {
        "https://en.wikipedia.org/wiki/Machine_learning": """
        Machine learning (ML) is a field of study in artificial intelligence concerned with
        the development and study of statistical algorithms that can learn from data and
        generalize to unseen data. ML finds application in many fields, including natural
        language processing, computer vision, speech recognition, and medicine.
        Deep learning is a subset of machine learning that uses neural networks with many layers.
        """,
        "https://en.wikipedia.org/wiki/Python_(programming_language)": """
        Python is a high-level, general-purpose programming language. Its design philosophy
        emphasizes code readability with the use of significant indentation. Python is
        dynamically typed and garbage-collected. It supports multiple programming paradigms,
        including structured, object-oriented, and functional programming.
        Python is widely used in machine learning and data science applications.
        """
    }

    # Build retriever
    retriever = build_retriever_from_documents(documents)
    print(f"Retriever stats: {retriever.get_stats()}")

    # Test retrieval
    query = "What is machine learning used for?"
    results = retriever.retrieve(query, top_k=3)

    print(f"\nQuery: {query}")
    print(f"Results:")
    for result in results:
        print(f"  {result}")
        print(f"  Text preview: {result.chunk.text[:100]}...")
