import os
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from retreival.retreiver import MultimodalRetriever
from prompts.prompts import VIDEO_RAG_PROMPT
from vectorstore.manager import VectorstoreManager
from configs.settings import RetrievalConfig, VectorStoreConfig
from models.embeddings import CLIPModel

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

        self.retriever = MultimodalRetriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store_manager,
            config=self.retrieval_config,
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": VIDEO_RAG_PROMPT},
        )

    def query(self, question: str) -> dict:
        result = self.qa_chain.invoke({"query": question})
        sources = []
        for doc in result.get("source_documents", []):
            metadata = doc.metadata
            source_info = {
                "video_id": metadata.get("video_id"),
                "content_type": metadata.get("content_type"),
                "timestamp": metadata.get("timestamp"),
                "caption": metadata.get("caption", "")[:100] + "..."
                if metadata.get("caption")
                else "",
            }
            sources.append(source_info)

        return {"answer": result["result"], "sources": sources, "success": True}

    def query_by_image(self, image, k: int = 6) -> dict:
        """Process an image query using the retriever directly."""
        try:
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

        except Exception as e:
            return {"results": [], "success": False, "error": str(e)}


if __name__ == "__main__":
    q = "what videos was it that Derek talked about the largest rainfall simulator?"
    rag = RAGApp()
    print(rag.query(q))
