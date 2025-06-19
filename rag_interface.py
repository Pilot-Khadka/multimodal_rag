import os
from rag import RAGApp
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI

from configs.settings import QueryDecompositionConfig, DecompositionPresets

api_key = os.environ.get("GEMINI_API")


class RAGInterface:
    def __init__(self):
        self.rag_app = RAGApp()
        self.config = QueryDecompositionConfig()

        # update temperature if specified
        if self.config.enable_decomposition:
            self.rag_app.query_decomposer.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=self.config.llm_temperature,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                google_api_key=api_key,
            )

    def ask(self, question: str, **kwargs) -> Dict[str, Any]:
        use_decomposition = kwargs.get(
            "use_decomposition", self.config.enable_decomposition
        )
        decomposition_type = kwargs.get(
            "decomposition_type", self.config.decomposition_type
        )

        if not use_decomposition:
            return self.rag_app.query(question, use_decomposition=False)

        return self.rag_app.query(
            question=question,
            use_decomposition=True,
            decomposition_type=decomposition_type,
        )

    def ask_with_image(
        self,
        image,
    ) -> Dict[str, Any]:
        # TODO
        pass

    def set_preset(self, preset_name: str):
        preset_map = {
            "conservative": DecompositionPresets.conservative,
            "balanced": DecompositionPresets.balanced,
        }
        if preset_name not in preset_map:
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {
                    list(preset_map.keys())}"
            )

        self.config = preset_map[preset_name]()
        if self.config.enable_decomposition:
            self.rag_app.query_decomposer.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=self.config.llm_temperature,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                google_api_key=api_key,
            )

    def get_current_config(self) -> QueryDecompositionConfig:
        return self.config


if __name__ == "__main__":
    rag = RAGInterface()
    question = "what video does veritasium talk about the largest rainfall simulator?"

    result = rag.ask(question)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result.get('sources', []))} found")
    print()

    # conservative preset
    rag.set_preset("balanced")
    result = rag.ask(question)
    print(f"Config: {rag.get_current_config()}")
    print(f"Answer: {result['answer']}")
    if "decomposed_queries" in result:
        print(f"Decomposed queries: {result['decomposed_queries']}")
    print()
