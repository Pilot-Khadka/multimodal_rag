import numpy as np
from typing import List, Dict, Any
from langchain.schema import Document, BaseRetriever
from core.vectorstore_manager import VectorstoreManager
from core.models.embeddings import BaseEmbeddingModel


class HierarchicalRetreiver(BaseRetriever):
    embedding_model: BaseEmbeddingModel
    vector_store: VectorstoreManager
    config: Dict

    def get_relevant_documents(self, query: str) -> List[Document]:
        raw_results = self.retrieve_by_text(query)
        return [
            Document(page_content=item["document"], metadata=item["metadata"])
            for item in raw_results
        ]

    def retrieve_by_text(self, query: str) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode_text([query])

        # only search summaries
        results = self.vector_store.text_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=self.config["top_k"],
            where={"is_summary": True},
        )
        if not results["ids"] or not results["ids"][0]:
            print("no summary found")
            return []

        return self._format_results(results)

    def retreive_by_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        raise NotImplementedError()
        pass

    def _format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
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
