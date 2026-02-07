"""
Embedding Module
Generates embeddings using local sentence-transformers models (free, no API costs)
"""

import os
import pickle
from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from chunker import Chunk


class Embedder:
    """
    Generates embeddings using sentence-transformers
    Uses BAAI/bge-small-en-v1.5 by default (384 dimensions, fast, good quality)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str = "embeddings",
        device: Optional[str] = None
    ):
        """
        Initialize the embedder

        Args:
            model_name: Name of the sentence-transformers model
            cache_dir: Directory to cache embeddings
            device: Device to use ('cpu', 'cuda', etc.) - auto-detected if None
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "embeddings.pkl")
        self.model = None
        self.device = device
        self._embedding_cache: dict[str, np.ndarray] = {}

        # Load cached embeddings
        self._load_cache()

    def _get_model(self):
        """Lazy load the model only when needed"""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model

    def _load_cache(self):
        """Load cached embeddings from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self._embedding_cache = pickle.load(f)
            except (pickle.PickleError, IOError):
                self._embedding_cache = {}

    def _save_cache(self):
        """Save embeddings cache to disk"""
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_file, "wb") as f:
            pickle.dump(self._embedding_cache, f)

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for text"""
        # Use hash for efficiency with long texts
        return f"{self.model_name}:{hash(text)}"

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        cache_key = self._get_cache_key(text)

        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        model = self._get_model()

        # BGE models recommend adding instruction prefix for queries
        if "bge" in self.model_name.lower():
            # For documents, no prefix needed
            embedding = model.encode(text, normalize_embeddings=True)
        else:
            embedding = model.encode(text, normalize_embeddings=True)

        embedding = np.array(embedding)
        self._embedding_cache[cache_key] = embedding

        return embedding

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query (with appropriate prefix for BGE models)

        Args:
            query: Query text

        Returns:
            Embedding vector as numpy array
        """
        model = self._get_model()

        # BGE models recommend instruction prefix for queries
        if "bge" in self.model_name.lower():
            query_with_prefix = f"Represent this sentence for searching relevant passages: {query}"
            embedding = model.encode(query_with_prefix, normalize_embeddings=True)
        else:
            embedding = model.encode(query, normalize_embeddings=True)

        return np.array(embedding)

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            2D numpy array of embeddings (num_texts x embedding_dim)
        """
        if not texts:
            return np.array([])

        # Check cache for each text
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._embedding_cache:
                embeddings.append((i, self._embedding_cache[cache_key]))
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        # Embed uncached texts
        if texts_to_embed:
            model = self._get_model()
            new_embeddings = model.encode(
                texts_to_embed,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True
            )

            # Cache new embeddings
            for idx, text, emb in zip(indices_to_embed, texts_to_embed, new_embeddings):
                emb = np.array(emb)
                cache_key = self._get_cache_key(text)
                self._embedding_cache[cache_key] = emb
                embeddings.append((idx, emb))

            # Save cache
            self._save_cache()

        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])

    def embed_chunks(
        self,
        chunks: List[Chunk],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of Chunk objects

        Args:
            chunks: List of Chunk objects
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            2D numpy array of embeddings
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts, batch_size, show_progress)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        model = self._get_model()
        return model.get_sentence_embedding_dimension()

    def get_cache_stats(self) -> dict:
        """Get statistics about the embedding cache"""
        return {
            "cached_embeddings": len(self._embedding_cache),
            "model_name": self.model_name,
            "embedding_dimension": self.get_embedding_dimension() if self.model else "not loaded"
        }


# Global embedder instance for convenience
_default_embedder: Optional[Embedder] = None


def get_embedder(model_name: str = "BAAI/bge-small-en-v1.5") -> Embedder:
    """Get or create the default embedder instance"""
    global _default_embedder
    if _default_embedder is None or _default_embedder.model_name != model_name:
        _default_embedder = Embedder(model_name=model_name)
    return _default_embedder


def embed_query(query: str, model_name: str = "BAAI/bge-small-en-v1.5") -> np.ndarray:
    """Convenience function to embed a query"""
    embedder = get_embedder(model_name)
    return embedder.embed_query(query)


def embed_texts(texts: List[str], model_name: str = "BAAI/bge-small-en-v1.5") -> np.ndarray:
    """Convenience function to embed multiple texts"""
    embedder = get_embedder(model_name)
    return embedder.embed_texts(texts)


def embed_chunks(chunks: List[Chunk], model_name: str = "BAAI/bge-small-en-v1.5") -> np.ndarray:
    """Convenience function to embed chunks"""
    embedder = get_embedder(model_name)
    return embedder.embed_chunks(chunks)


if __name__ == "__main__":
    # Test the embedder
    print("Testing Embedder...")

    embedder = Embedder()

    # Test single text embedding
    text = "Machine learning is a subset of artificial intelligence."
    embedding = embedder.embed_text(text)
    print(f"Single text embedding shape: {embedding.shape}")

    # Test query embedding
    query = "What is machine learning?"
    query_embedding = embedder.embed_query(query)
    print(f"Query embedding shape: {query_embedding.shape}")

    # Test batch embedding
    texts = [
        "Python is a programming language.",
        "Machine learning uses statistical methods.",
        "Neural networks are inspired by the brain."
    ]
    embeddings = embedder.embed_texts(texts)
    print(f"Batch embeddings shape: {embeddings.shape}")

    # Test cosine similarity
    similarity = np.dot(query_embedding, embedding)
    print(f"Cosine similarity between query and text: {similarity:.4f}")

    print(f"\nCache stats: {embedder.get_cache_stats()}")
