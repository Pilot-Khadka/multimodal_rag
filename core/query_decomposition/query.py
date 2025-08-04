import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate


api_key = os.environ.get("GEMINI_API")


class QueryDecomposer:
    """
    https://python.langchain.com/docs/how_to/MultiQueryRetriever/
    """

    def __init__(self, llm=None):
        self.llm = llm or ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key=api_key,
        )

        self.multi_query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate 3 different versions 
            of the given user question to retrieve relevant documents from a vector database. 
            By generating multiple perspectives on the user question, your goal is to help the user overcome 
            some of the limitations of distance-based similarity search.
            
            Provide these alternative questions separated by newlines.
            
            Original question: {question}
            
            Alternative questions:""",
        )

        self.step_back_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an expert at world knowledge. Your task is to step back and paraphrase a question 
            to a more generic step-back question, which is easier to answer. Here are a few examples:
            
            Original Question: Could the members of The Police perform lawful arrests?
            Stepback Question: What can the members of The Police do?
            
            Original Question: Can a Tesla car drive itself?
            Stepback Question: What can a Tesla car do?,

            Original Question: {question}
            Stepback Question:""",
        )

        self.hyde_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Please write a passage to answer the question. The passage should be detailed and informative.
            
            Question: {question}
            
            Passage:""",
        )

    def decompose_query(self, question: str, decomposition_type) -> List[str]:
        if decomposition_type == "multi_query":
            return self._multi_query_decomposition(question)
        elif decomposition_type == "step_back":
            return self._step_back_decomposition(question)
        elif decomposition_type == "hyde":
            return self._hyde_decomposition(question)
        else:
            raise ValueError(f"unknown type : {decomposition_type}")

    def _multi_query_decomposition(self, question: str) -> List[str]:
        response = self.llm.invoke(self.multi_query_prompt.format(question=question))
        queries = [q.strip() for q in response.content.split("\n") if q.strip()]
        # include orignal query
        return [question] + queries

    def _step_back_decomposition(self, question: str) -> List[str]:
        response = self.llm.invoke(self.step_back_prompt.format(question=question))
        step_back_question = response.content.strip()
        return [question, step_back_question]

    def _hyde_decomposition(self, question: str) -> List[str]:
        response = self.llm.invoke(self.hyde_prompt.format(question=question))
        hypothetical_doc = response.content.strip()
        return [question, hypothetical_doc]

    def decompose_multiple_types(self, question: str, decomposition_types) -> List[str]:
        all_queries = set()

        for decomposition_type in decomposition_types:
            queries = self.decompose_query(question, decomposition_type)
            all_queries.update(queries)

        return list(all_queries)
