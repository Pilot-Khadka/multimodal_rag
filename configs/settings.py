from dataclasses import dataclass
from typing import List
from enum import Enum
import torch


@dataclass
class EmbeddingConfig:
    model_name: str = "ViT-B/32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ChunkingConfig:
    chunk_size: int = 300  # to fit CLIP's limits
    chunk_overlap: int = 20
    separators: List[str] = None

    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", " "]


@dataclass
class VectorStoreConfig:
    persist_path: str = "./vectorstore/chroma_store"
    text_collection_name: str = "text_embeddings"
    image_collection_name: str = "image_collection"
    hash_collection_name: str = "content_hashes"
    similarity_metric: str = "cosine"


class QueryDecompositionType(Enum):
    MULTI_QUERY = "multi_query"
    STEP_BACK = "step_back"
    HYDE = "hyde"


@dataclass
class QueryDecompositionConfig:
    enable_decomposition: bool = False

    # primary decomposition type -> multiquery
    decomposition_type: QueryDecompositionType = QueryDecompositionType.MULTI_QUERY
    max_docs_per_query: int = 10
    # higher -> more creative
    llm_temperature: float = 0.3


class DecompositionPresets:
    @staticmethod
    def conservative() -> QueryDecompositionConfig:
        return QueryDecompositionConfig(
            enable_decomposition=True,
            decomposition_type=QueryDecompositionType.MULTI_QUERY,
            llm_temperature=0.1,
        )

    @staticmethod
    def balanced() -> QueryDecompositionConfig:
        return QueryDecompositionConfig(
            enable_decomposition=True,
            decomposition_type=QueryDecompositionType.MULTI_QUERY,
            llm_temperature=0.3,
        )


@dataclass
class FileConfig:
    channel_name: str = "1veritasium"
    video_status_file: str = f"{channel_name}_videos.csv"
    video_path: str = "data/videos"
    caption_path: str = "data/captions"
    cleaned_caption_path: str = "data/cleaned_captions"


@dataclass
class CsvConfig:
    channel_name: str = "1veritasium"
    video_status_file: str = f"{channel_name}_videos.csv"
    video: str = "video_downloaded"
    caption: str = "caption_downloaded"
    captions: str = "captions_cleaned"
    video_embed: str = "video_embedding"
    text_embed: str = "text_embedding"


@dataclass
class RetrievalConfig:
    top_k: int = 2


@dataclass
class RAGConfig:
    embedding: EmbeddingConfig = None
    chunking: ChunkingConfig = None
    vectorstore: VectorStoreConfig = None
    retrieval: RetrievalConfig = None

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.vectorstore is None:
            self.vectorstore = VectorStoreConfig()
        if self.retrieval is None:
            self.retrieval = RetrievalConfig()


if __name__ == "__main__":
    pass
