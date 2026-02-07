"""
Document Chunking Module
Splits documents into overlapping chunks for better retrieval
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    """Represents a document chunk with metadata"""
    text: str
    source_url: str
    chunk_index: int
    start_char: int
    end_char: int

    def __repr__(self):
        return f"Chunk(source={self.source_url}, index={self.chunk_index}, len={len(self.text)})"


class DocumentChunker:
    """
    Splits documents into overlapping chunks
    Uses paragraph-based chunking with token size constraints
    """

    def __init__(
        self,
        target_chunk_size: int = 450,
        min_chunk_size: int = 300,
        max_chunk_size: int = 600,
        overlap_ratio: float = 0.25
    ):
        """
        Initialize the chunker

        Args:
            target_chunk_size: Target number of tokens per chunk
            min_chunk_size: Minimum tokens per chunk
            max_chunk_size: Maximum tokens per chunk
            overlap_ratio: Ratio of overlap between consecutive chunks (0.2-0.3 recommended)
        """
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_ratio = overlap_ratio

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate number of tokens in text
        Simple heuristic: ~4 characters per token for English text
        """
        return len(text) // 4

    def _chars_for_tokens(self, tokens: int) -> int:
        """Convert token count to approximate character count"""
        return tokens * 4

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences while preserving sentence boundaries
        """
        # Split on sentence-ending punctuation followed by space or end
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split on double newlines or single newlines with blank content
        paragraphs = re.split(r'\n\s*\n|\n{2,}', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def chunk_document(
        self,
        text: str,
        source_url: str = ""
    ) -> List[Chunk]:
        """
        Chunk a single document into overlapping pieces

        Args:
            text: Document text to chunk
            source_url: Source URL for metadata

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        chunks: List[Chunk] = []
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        current_chunk_sentences: List[str] = []
        current_chunk_chars = 0
        chunk_start_char = 0
        current_position = 0

        target_chars = self._chars_for_tokens(self.target_chunk_size)
        min_chars = self._chars_for_tokens(self.min_chunk_size)
        max_chars = self._chars_for_tokens(self.max_chunk_size)
        overlap_chars = int(target_chars * self.overlap_ratio)

        for sentence in sentences:
            sentence_len = len(sentence) + 1  # +1 for space

            # Check if adding this sentence exceeds max size
            if current_chunk_chars + sentence_len > max_chars and current_chunk_sentences:
                # Create chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(Chunk(
                    text=chunk_text,
                    source_url=source_url,
                    chunk_index=len(chunks),
                    start_char=chunk_start_char,
                    end_char=chunk_start_char + len(chunk_text)
                ))

                # Calculate overlap - keep sentences from the end totaling overlap_chars
                overlap_sentences: List[str] = []
                overlap_len = 0
                for s in reversed(current_chunk_sentences):
                    if overlap_len + len(s) <= overlap_chars:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break

                # Start new chunk with overlap
                current_chunk_sentences = overlap_sentences
                current_chunk_chars = overlap_len
                chunk_start_char = current_position - overlap_len

            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_chars += sentence_len
            current_position += sentence_len

        # Handle remaining sentences
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            # Only add if it meets minimum size or it's the only chunk
            if len(chunk_text) >= min_chars or not chunks:
                chunks.append(Chunk(
                    text=chunk_text,
                    source_url=source_url,
                    chunk_index=len(chunks),
                    start_char=chunk_start_char,
                    end_char=chunk_start_char + len(chunk_text)
                ))
            elif chunks:
                # Merge with previous chunk if too small
                prev_chunk = chunks[-1]
                merged_text = prev_chunk.text + " " + chunk_text
                chunks[-1] = Chunk(
                    text=merged_text,
                    source_url=source_url,
                    chunk_index=prev_chunk.chunk_index,
                    start_char=prev_chunk.start_char,
                    end_char=prev_chunk.start_char + len(merged_text)
                )

        return chunks

    def chunk_documents(
        self,
        documents: dict[str, str]
    ) -> List[Chunk]:
        """
        Chunk multiple documents

        Args:
            documents: Dictionary mapping source URL to document text

        Returns:
            List of all Chunk objects from all documents
        """
        all_chunks: List[Chunk] = []

        for source_url, text in documents.items():
            doc_chunks = self.chunk_document(text, source_url)
            # Re-index chunks globally
            for chunk in doc_chunks:
                chunk.chunk_index = len(all_chunks)
                all_chunks.append(chunk)

        return all_chunks


def chunk_text(
    text: str,
    source_url: str = "",
    target_size: int = 450,
    overlap_ratio: float = 0.25
) -> List[Chunk]:
    """
    Convenience function to chunk a single document

    Args:
        text: Document text to chunk
        source_url: Source URL for metadata
        target_size: Target chunk size in tokens
        overlap_ratio: Overlap ratio between chunks

    Returns:
        List of Chunk objects
    """
    chunker = DocumentChunker(target_chunk_size=target_size, overlap_ratio=overlap_ratio)
    return chunker.chunk_document(text, source_url)


def chunk_documents(
    documents: dict[str, str],
    target_size: int = 450,
    overlap_ratio: float = 0.25
) -> List[Chunk]:
    """
    Convenience function to chunk multiple documents

    Args:
        documents: Dictionary mapping source URL to document text
        target_size: Target chunk size in tokens
        overlap_ratio: Overlap ratio between chunks

    Returns:
        List of all Chunk objects
    """
    chunker = DocumentChunker(target_chunk_size=target_size, overlap_ratio=overlap_ratio)
    return chunker.chunk_documents(documents)


if __name__ == "__main__":
    # Test the chunker
    test_text = """
    Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Recently, artificial neural networks have been able to surpass many previous approaches in performance.

    ML finds application in many fields, including natural language processing, computer vision, speech recognition, email filtering, agriculture, and medicine. The application of ML to business problems is known as predictive analytics.

    Statistics and mathematical optimization (mathematical programming) methods comprise the foundations of machine learning. Data mining is a related field of study, focusing on exploratory data analysis (EDA) through unsupervised learning.

    From a theoretical viewpoint, probably approximately correct (PAC) learning provides a framework for describing machine learning.
    """

    chunker = DocumentChunker(target_chunk_size=100, min_chunk_size=50, max_chunk_size=150)
    chunks = chunker.chunk_document(test_text, "https://en.wikipedia.org/wiki/Machine_learning")

    print(f"Created {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"\n{chunk}")
        print(f"Text: {chunk.text[:100]}...")
        print(f"Estimated tokens: {len(chunk.text) // 4}")
