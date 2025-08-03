import hashlib
import chromadb
from utils.helper import get_id_based_on_column, check_for_video_and_caption


class VectorstoreManager:
    def __init__(self, config):
        self.config = config
        self.chroma_client = chromadb.PersistentClient(
            path=config["persist_path"])

        self.text_collection = self.chroma_client.get_or_create_collection(
            name=config["text_collection_name"],
            metadata={"hnsw:space": config["similarity_metric"]},
        )
        self.image_collection = self.chroma_client.get_or_create_collection(
            name=config["image_collection_name"],
            metadata={"hnsw:space": config["similarity_metric"]},
        )
        self.content_hash_collection = self.chroma_client.get_or_create_collection(
            name=config["hash_collection_name"]
        )

    def generate_content_hash(self, content) -> str:
        if isinstance(content, str):
            content = content.encode("utf-8")
        elif isinstance(content, bytes):
            pass  # already bytes
        else:
            content = str(content).encode("utf-8")

        return hashlib.sha256(content).hexdigest()

    def is_content_processed(self, content_hash: str) -> bool:
        try:
            results = self.content_hash_collection.get(id=[content_hash])
            return len(results["ids"]) > 0
        except:
            return False

    def mark_content_processed(
        self,
        content_hash: str,
        content_type: str,
        source: str,
    ):
        self.content_hash_collection.add(
            ids=[content_hash],
            documents=[
                f"Processed {content_type} from {source}",
            ],
            metadatas=[
                {
                    "content_type": content_type,
                    "source": source,
                }
            ],
        )


if __name__ == "__main__":
    vectorstore = VectorstoreManager()
    matched_files = check_for_video_and_caption()
    filtered_ids = get_id_based_on_column(column_name="video_embedding")

    for video_id, (video_path, caption_path) in matched_files.items():
        print("video id:", video_id)
        print("video_path:", video_path)
        print("caption path:", caption_path)
        if video_id not in filtered_ids:
            continue
        vectorstore.create_image_embeddings_from_video(
            video_id=video_id, video_path=video_path, caption_path=caption_path
        )
