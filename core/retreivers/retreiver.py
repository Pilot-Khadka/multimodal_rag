from typing import List, Dict, Any
from langchain.schema import Document, BaseRetriever
from models.embeddings import BaseEmbeddingModel, CLIPModel
from configs.settings import RetrievalConfig, VectorStoreConfig
from vectorstore.manager import VectorstoreManager


class MultimodalRetriever(BaseRetriever):
    embedding_model: BaseEmbeddingModel
    vector_store: VectorstoreManager

    def __init__(self, config):
        self.config = config

    def get_relevant_documents(self, query: str) -> List[Document]:
        raw_results = self.retrieve_by_text(query)
        # print("raw results:", raw_results)
        return [
            Document(page_content=item["document"], metadata=item["metadata"])
            for item in raw_results
        ]

    def retrieve_by_text(self, query: str) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode_text([query])

        results = self.vector_store.text_collection.query(
            query_embeddings=query_embedding.tolist(), n_results=self.config["top_k"]
        )

        return self._format_results(results)

    def retrieve_by_image(self, image) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode_image([image])

        # search both text and image
        text_results = self.vector_store.text_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=self.config["top_k"] // 2,
        )

        image_results = self.vector_store.image_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=self.config["top_k"] // 2,
        )

        combined_results = self._combine_results(text_results, image_results)
        return combined_results

    def _format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
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
        """combines and rank text and image results."""
        combined = []
        combined.extend(self._format_results(text_results))
        combined.extend(self._format_results(image_results))

        # sort by distance
        combined.sort(key=lambda x: x["distance"] if x["distance"] else float("inf"))

        return combined[: self.config["top_k"]]

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


if __name__ == "__main__":
    query = "what video was it that Derek talked about the largest rainfall simulator?"
    clip = CLIPModel()
    storeconfig = VectorStoreConfig()
    vectorstore = VectorstoreManager(storeconfig)
    retreival_config = RetrievalConfig()
    retreiver = HierarchicalRetreiver(
        embedding_model=clip, vector_store=vectorstore, config=retreival_config
    )
    result = retreiver.retrieve_by_text(query)
    print(result)
