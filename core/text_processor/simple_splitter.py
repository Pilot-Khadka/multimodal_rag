from typing import List, Dict, Any
from base_splitter import BaseTextProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter


class SimpleTextProcessor(BaseTextProcessor):
    def __init__(self, config):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            separators=config["separators"],
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )

    def split_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
