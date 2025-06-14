from typing import List
from langchain.prompts import PromptTemplate
from langchain.schema import Document


VIDEO_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant that answers questions about video content. 
You have access to both transcribed text and visual information from video frames.

Context from videos:
{context}

Instructions:
- Use the provided context to answer the question
- If the context includes frame information, mention visual elements when relevant
- Include timestamps when available
- If you cannot find relevant information in the context, say so clearly
- Be specific about which video or segment you're referencing

Question: {question}

Answer:
""",
)


def format_context(documents: List[Document]) -> str:
    """Format retrieved documents for the prompt"""
    formatted_context = []

    for doc in documents:
        metadata = doc.metadata
        content_type = metadata.get("content_type", "unknown")
        video_id = metadata.get("video_id", "unknown")

        if content_type == "text":
            chunk_info = f"Text from video {video_id}:"
            formatted_context.append(f"{chunk_info}\n{doc.page_content}\n")

        elif content_type == "image":
            timestamp = metadata.get("timestamp", 0)
            caption = metadata.get("caption", "No caption available")
            frame_info = f"Frame from video {video_id} at {timestamp:.2f}s:"
            formatted_context.append(f"{frame_info}\nCaption: {caption}\n")

    return "\n".join(formatted_context)
