#
from models.embeddings import BaseEmbeddingModel, CLIPModel
from configs.settings import RetrievalConfig, VectorStoreConfig
from vectorstore.manager import VectorstoreManager
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain.schema import Document, BaseRetriever

#
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#


class Retreiver:
    @abstractmethod
    def encode_text(self, texts: List[str]):
        pass

    @abstractmethod
    def encode_image(self, images):
        pass


class MultimodalRetriever(BaseRetriever):
    """Retrieves relevant documents based on text or image queries."""

    embedding_model: BaseEmbeddingModel
    vector_store: VectorstoreManager
    config: RetrievalConfig

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """LangChain-required method."""
        raw_results = self.retrieve_by_text(query)
        print("raw results:", raw_results)
        return [
            Document(page_content=item["document"], metadata=item["metadata"])
            for item in raw_results
        ]

    def retrieve_by_text(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve documents similar to text query."""
        query_embedding = self.embedding_model.encode_text([query])

        results = self.vector_store.text_collection.query(
            query_embeddings=query_embedding.tolist(), n_results=self.config.top_k
        )

        return self._format_results(results)

    def retrieve_by_image(self, image) -> List[Dict[str, Any]]:
        """Retrieve documents similar to image query."""
        query_embedding = self.embedding_model.encode_image([image])

        # Search both text and image collections
        text_results = self.vector_store.text_collection.query(
            query_embeddings=query_embedding.tolist(), n_results=self.config.top_k // 2
        )

        image_results = self.vector_store.image_collection.query(
            query_embeddings=query_embedding.tolist(), n_results=self.config.top_k // 2
        )

        combined_results = self._combine_results(text_results, image_results)
        return combined_results

    def _format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format retrieval results."""
        formatted = []
        for i, doc_id in enumerate(results["ids"][0]):
            formatted.append(
                {
                    "id": doc_id,
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                    if results["metadatas"][0]
                    else {},
                    "distance": results["distances"][0][i]
                    if results["distances"]
                    else None,
                }
            )
        return formatted

    def _combine_results(
        self, text_results: Dict, image_results: Dict
    ) -> List[Dict[str, Any]]:
        """Combine and rank text and image results."""
        combined = []
        combined.extend(self._format_results(text_results))
        combined.extend(self._format_results(image_results))

        # Sort by distance (similarity)
        combined.sort(key=lambda x: x["distance"]
                      if x["distance"] else float("inf"))

        return combined[: self.config.top_k]

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        query_embedding = self.embedding_model.encode_text([query])[0]

        text_results = self.vector_store.text_collection.query(
            query_embeddings=[query_embedding.cpu().tolist()], n_results=k // 2
        )

        image_results = self.vector_store.image_collection.query(
            query_embeddings=[query_embedding.cpu().tolist()], n_results=k // 2
        )

        documents = []
        for i, doc in enumerate(text_results["documents"][0]):
            metadata = text_results["metadatas"][0][i]
            documents.append(Document(page_content=doc, metadata=metadata))

        for i, doc in enumerate(image_results["documents"][0]):
            metadata = image_results["metadatas"][0][i]
            documents.append(Document(page_content=doc, metadata=metadata))

        return documents


class HierarchicalRetreiver(BaseRetriever):
    """Retrieves relevant documents based on text or image queries."""

    embedding_model: BaseEmbeddingModel
    vector_store: VectorstoreManager
    config: RetrievalConfig

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """LangChain-required method."""
        raw_results = self.retrieve_by_text(query)
        print("raw results:", raw_results)
        return [
            Document(page_content=item["document"], metadata=item["metadata"])
            for item in raw_results
        ]

    def retrieve_by_text(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve documents similar to text query."""
        query_embedding = self.embedding_model.encode_text([query])

        # only search summaries
        results = self.vector_store.text_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=self.config.top_k,
            where={"is_summary": True},
        )
        if not results["ids"] or not results["ids"][0]:
            print("no summary found")
            return []

        return self._format_results(results)

    def _format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format retrieval results."""
        formatted = []
        for i, summary_id in enumerate(results["ids"][0]):
            summary_metadata = results["metadatas"][0][i]
            full_chunk_id = summary_metadata["full_chunk_id"]

            # retreive corresponding full chunk
            full_chunk_result = self.vector_store.text_collection.get(
                ids=[full_chunk_id]
            )
            if full_chunk_result["documents"]:
                result_data = {
                    "id": full_chunk_id,
                    # Full context
                    "document": full_chunk_result["documents"][0],
                    "metadata": full_chunk_result["metadatas"][0],
                    "summary": results["documents"][0][i],  # Original summary
                    "summary_metadata": summary_metadata,
                    "distance": results["distances"][0][i]
                    if "distances" in results
                    else None,
                    "rank": i + 1,
                }
                formatted.append(result_data)

        return formatted


if __name__ == "__main__":
    query = "what videos was it that Derek talked about the largest rainfall simulator?"
    clip = CLIPModel()
    storeconfig = VectorStoreConfig()
    vectorstore = VectorstoreManager(storeconfig)
    retreival_config = RetrievalConfig()
    retreiver = HierarchicalRetreiver(
        embedding_model=clip, vector_store=vectorstore, config=retreival_config
    )
    result = retreiver.retrieve_by_text(query)
    print(result)
