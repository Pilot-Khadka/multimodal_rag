import os
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

from configs.settings import ChunkingConfig

api_key = os.environ.get("GEMINI_API")


class BaseTextProcessor(ABC):
    @abstractmethod
    def split_text(self, texts: List[str]):
        pass


class HierarchicalTextProcessor(BaseTextProcessor):
    def __init__(self, summary_max_tokens=70, full_chunk_max_tokens=1024):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key=api_key,
        )
        self.summary_max_tokens = summary_max_tokens
        self.full_chunk_max_tokens = full_chunk_max_tokens

    def split_text(self, text):
        full_chunks = self._split_into_full_chunks(text)
        summaries = []
        chunk_mappings = []

        for i, full_chunk in enumerate(full_chunks):
            summary = self._create_summary(full_chunk)
            print(summary)
            summaries.append(summary)

            mapping = {
                "summary_index": i,
                "full_chunk_index": i,
                "full_chunk_start": i * self.full_chunk_max_tokens,
                "full_chunk_end": min((i + 1) * self.full_chunk_max_tokens, len(text)),
            }
            chunk_mappings.append(mapping)
        return summaries, full_chunks, chunk_mappings

    def _split_into_full_chunks(self, text):
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        # 10% overlap
        overlap_size = min(200, self.full_chunk_max_tokens // 10)

        i = 0
        while i < len(words):
            word = words[i]
            word_tokens = len(word.split())

            if current_length + word_tokens > self.full_chunk_max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                    overlap_start = max(0, len(current_chunk) - overlap_size)
                    current_chunk = current_chunk[overlap_start:] + [word]
                    current_length = len(current_chunk)

                else:
                    current_chunk = [word]
                    current_length = word_tokens

            else:
                current_chunk.append(word)
                current_length += word_tokens

            i += 1
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _create_summary(self, chunk: str) -> str:
        words = chunk.split()
        if len(words) <= self.summary_max_tokens:
            return chunk

        prompt = f"""Please create a concise summary of the following text. The summary must be exactly {self.summary_max_tokens} words or fewer, and should capture the main topics, key information, and essential context. Focus on the most important content that would be useful for search and retrieval.
            Text to summarize:
            {chunk}

            Summary (max {self.summary_max_tokens} words):"""

        message = HumanMessage(content=prompt)
        response = self.llm.invoke([message])
        summary = response.content.strip()
        summary_words = summary.split()
        if len(summary_words) > self.summary_max_tokens:
            summary = " ".join(summary_words[: self.summary_max_tokens])

        return summary


class SimpleTextProcessor(BaseTextProcessor):
    """Handles text splitting and preprocessing."""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            separators=config.separators,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        return self.splitter.split_text(text)

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks while preserving metadata."""
        chunked_docs = []
        for doc in documents:
            chunks = self.split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                chunked_doc = {
                    "content": chunk,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "chunk_id": i,
                        "source_doc_id": doc.get("id", "unknown"),
                    },
                }
                chunked_docs.append(chunked_doc)
        return chunked_docs
