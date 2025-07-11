import os
from typing import List, Optional
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from retreival.retreiver import HierarchicalRetreiver
from prompts.prompts import VIDEO_RAG_PROMPT
from vectorstore.manager import VectorstoreManager
from configs.settings import RetrievalConfig, VectorStoreConfig
from models.embeddings import CLIPModel
from query_decomposition.query import QueryDecomposer, QueryDecompositionType

api_key = os.environ.get("GEMINI_API")


class RAGApp:
    def __init__(self, llm=None):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key=api_key,
        )
        self.embedding_model = CLIPModel()
        self.vector_store_config = VectorStoreConfig()
        self.vector_store_manager = VectorstoreManager(self.vector_store_config)
        self.retrieval_config = RetrievalConfig()
        self.retriever = HierarchicalRetreiver(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store_manager,
            config=self.retrieval_config,
        )
        self.query_decomposer = QueryDecomposer(self.llm)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": VIDEO_RAG_PROMPT},
        )

    def query(
        self,
        question: str,
        use_decomposition: bool = False,
        decomposition_type: Optional[QueryDecompositionType] = None,
    ) -> dict:
        if not use_decomposition:
            return self._single_query(question)

        if decomposition_type is None:
            decomposition_type = QueryDecompositionType.MULTI_QUERY

        return self._decompose_query(question, decomposition_type)

    def _single_query(self, question):
        result = self.qa_chain.invoke({"query": question})
        sources = self._format_sources(result.get("source_documents", []))
        return {"answer": result["result"], "sources": sources, "success": True}

    def _decompose_query(
        self,
        question: str,
        decomposition_type: QueryDecompositionType,
    ) -> dict:
        all_queries = self.query_decomposer.decompose_query(
            question, decomposition_type
        )
        all_results = []

        for query in all_queries:
            docs = self.retriever.get_relevant_documents(query)
            all_results.append(docs)

        result = self.qa_chain.invoke({"query": question})
        sources = self._format_sources(result.get("source_documents", []))

        return {
            "answer": result["result"],
            "sources": sources,
            "success": True,
            "decomposed_queries": all_queries,
        }

    def _format_sources(self, source_documents: List[Document]) -> List[dict]:
        sources = []
        for doc in source_documents:
            metadata = doc.metadata
            source_info = {
                "video_id": metadata.get("video_id"),
                "video_link": f"https://www.youtube.com/watch?v={metadata.get('video_id')}&t={int(metadata.get('timestamp', 0))}s",
                "timestamp": metadata.get("timestamp"),
            }
            sources.append(source_info)
        return sources

    def query_by_image(self, image, k: int = 6) -> dict:
        results = self.retriever.retrieve_by_image(image)

        sources = []
        for result in results:
            metadata = result.get("metadata", {})
            source_info = {
                "id": result.get("id"),
                "video_id": metadata.get("video_id"),
                "content_type": metadata.get("content_type"),
                "timestamp": metadata.get("timestamp"),
                "distance": result.get("distance"),
                "caption": (metadata.get("caption", "")[:100] + "...")
                if metadata.get("caption")
                else "",
                "content": (result.get("document", "")[:200] + "...")
                if len(result.get("document", "")) > 200
                else result.get("document", ""),
            }
            sources.append(source_info)

        return {"results": sources, "success": True}


if __name__ == "__main__":
    q = "what video does veritasium talk about the largest rainfall simulator?"
    rag = RAGApp()
    print(rag.query(q))
