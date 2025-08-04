import os
import uuid
import cv2
from PIL import Image
import pandas as pd
import argparse
from datetime import datetime
import numpy as np
from typing import Optional

from utils.general import get_config
from utils.helper import get_id_based_on_column, get_status
from utils.generate_video_frame import is_scene_change
from utils.helper import print_frame_progress, set_file_status, get_filtered_ids
from utils.process_captions import parse_srt_to_frame_map
from core.vectorstore_manager import VectorstoreManager
from core.text_processor.text_splitter import HierarchicalTextProcessor
from core.models.embeddings import CLIPModel
from download_youtube_videos import download_videos
from utils.helper import get_file_path
from download_youtube_videos import get_all_videos, download_single_caption


api_key = os.environ.get("YOUTUBE_API")
parser = argparse.ArgumentParser(description="Download and process youtube videos")
parser.add_argument(
    "--num", type=int, required=True, help="number of videos to download"
)
parser.add_argument(
    "--reprocess", action="store_true", help="reprocess existing videos"
)
parser.add_argument("--video-id", type=str, help="process specific video ID only")
args = parser.parse_args()


class Pipeline:
    def __init__(self, config=get_config()):
        self.cfg = config
        self.csv_config = self.cfg["csv_config"]
        self.file_config = self.cfg["file_config"]
        self.channel_config = get_config(path="configs/channel_config.yaml")

        self.embedding_model = CLIPModel(
            model_name=self.cfg["embedding"]["model_name"],
            device=self.cfg["embedding"]["device"],
        )

        # self.text_processor = TextProcessor(self.cfg.chunking)
        self.text_processor = HierarchicalTextProcessor(
            summary_max_tokens=70,  # CLIP limit 77
            full_chunk_max_tokens=2048,
        )

        self.vector_store = VectorstoreManager(self.cfg["vectorstore"])

    def get_video_url_by_id(self, df: pd.DataFrame, video_id: str) -> Optional[str]:
        """Get video URL from DataFrame based on video ID"""
        try:
            matching_rows = df.loc[df["video_id"] == video_id, "video_url"]
            if not matching_rows.empty:
                return matching_rows.iloc[0]
            else:
                print(f"No video found with ID: {video_id}")
                return None
        except Exception as e:
            print(f"Error getting video URL for {video_id}: {e}")
            return None

    def create_image_embeddings_from_video(
        self,
        video_id: str,
        video_path: str,
        caption_path: str,
        csv_filename: str,
    ) -> bool:
        """Create image embeddings from video frames with scene change detection"""
        try:
            ids = []
            embedding_list = []
            metadatas = []
            documents = []

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Cannot open video file: {video_path}")
                return False

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Processing video {video_id}: {total_frames} frames at {fps} FPS")

            frame_idx = 1
            processed_frames = 0

            success, prev_frame = cap.read()
            if not success:
                print("Couldn't read the first frame")
                cap.release()
                return False

            range_map = parse_srt_to_frame_map(input_path=caption_path, fps=fps)

            while True:
                success, curr_frame = cap.read()
                if not success:
                    break

                is_change, score = is_scene_change(
                    prev_frame, curr_frame, threshold=0.5
                )

                if is_change:
                    frame_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    frame_bytes = cv2.imencode(".jpg", curr_frame)[1].tobytes()
                    frame_hash = self.vector_store.generate_content_hash(frame_bytes)

                    if not self.vector_store.is_content_processed(frame_hash):
                        image_embedding = self.embedding_model.encode_image(pil_image)

                        timestamp = frame_idx / fps
                        frame_id = f"frame_{video_id}_{processed_frames:06d}"

                        ids.append(frame_id)
                        embedding_list.append(image_embedding.tolist()[0])

                        caption = range_map.get_caption(frame_idx)
                        documents.append(caption)

                        metadata = {
                            "content_type": "image",
                            "source": video_path,
                            "video_id": video_id,
                            "frame_number": frame_idx,
                            "timestamp": timestamp,
                            "frame_hash": frame_hash,
                            "caption": caption,
                            "created_at": datetime.now().isoformat(),
                        }
                        cleaned_metadata = {
                            k: v for k, v in metadata.items() if v is not None
                        }
                        metadatas.append(cleaned_metadata)

                        self.vector_store.mark_content_processed(
                            frame_hash, "image", f"{video_path}:frame_{frame_idx}"
                        )

                        processed_frames += 1

                prev_frame = curr_frame
                print_frame_progress(frame_idx, total_frames)
                frame_idx += 1

                if frame_idx >= total_frames:
                    break

            cap.release()

            if ids:
                self.vector_store.image_collection.add(
                    documents=documents,
                    embeddings=embedding_list,
                    ids=ids,
                    metadatas=metadatas,
                )
                print(f"\nAdded {len(ids)} image embeddings from {video_path}")
            else:
                print(f"No new frames found in {video_path} (all already processed)")

            set_file_status(
                video_id=video_id,
                column=self.csv_config["video_embed"],
                csv_path=csv_filename,
            )

            return True

        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            return False

    def create_text_embeddings_from_caption(
        self,
        video_id: str,
        caption_path: str,
        csv_filename: str,
    ) -> bool:
        try:
            with open(caption_path, "r", encoding="utf-8") as f:
                caption_text = f.read()

            if not caption_text.strip():
                print(f"Empty caption file for video {video_id}")
                return False

            cleaned_text = self._clean_caption_text(caption_text)
            content_hash = self.vector_store.generate_content_hash(cleaned_text)
            if self.vector_store.is_content_processed(content_hash):
                print(f"Text from {caption_path} has already been processed")
                set_file_status(
                    video_id=video_id,
                    column=self.csv_config["text_embed"],
                    csv_path=csv_filename,
                )
                return True

            summaries, full_chunks, chunk_mappings = self.text_processor.split_text(
                cleaned_text
            )
            summary_embeddings = self.embedding_model.encode_text(summaries)

            summary_ids = []
            summary_documents = []
            summary_embedding_list = []
            summary_metadatas = []

            full_chunk_ids = []
            full_chunk_documents = []
            full_metadatas = []

            for i, (summary, full_chunk, embedding, mapping) in enumerate(
                zip(summaries, full_chunks, summary_embeddings, chunk_mappings)
            ):
                summary_doc_id = f"summary_{video_id}_{uuid.uuid4().hex[:8]}_{i}"
                full_chunk_doc_id = f"full_{video_id}_{uuid.uuid4().hex[:8]}_{i}"

                summary_ids.append(summary_doc_id)
                summary_documents.append(summary)
                summary_embedding_list.append(embedding.cpu().tolist())

                summary_metadata = {
                    "content_type": "text_summary",
                    "source": caption_path,
                    "video_id": video_id,
                    "chunk_index": i,
                    "total_chunks": len(summaries),
                    "content_hash": content_hash,
                    "created_at": datetime.now().isoformat(),
                    "full_chunk_id": full_chunk_doc_id,  # Link to full chunk
                    "is_summary": True,
                    "summary_method": "llm_gemini",  # Track summarization method
                }
                summary_metadatas.append(summary_metadata)

                full_chunk_ids.append(full_chunk_doc_id)
                full_chunk_documents.append(full_chunk)

                full_metadata = {
                    "content_type": "text_full",
                    "source": caption_path,
                    "video_id": video_id,
                    "chunk_index": i,
                    "total_chunks": len(full_chunks),
                    "content_hash": content_hash,
                    "created_at": datetime.now().isoformat(),
                    "summary_id": summary_doc_id,  # Link to summary
                    "is_summary": False,
                    # Track chunk size
                    "chunk_tokens": len(full_chunk.split()),
                }
                full_metadatas.append(full_metadata)

            print(f"Storing {len(summary_documents)} summaries with embeddings...")
            self.vector_store.text_collection.add(
                documents=summary_documents,
                embeddings=summary_embedding_list,
                ids=summary_ids,
                metadatas=summary_metadatas,
            )

            print(f"Storing {len(full_chunk_documents)} full chunks...")
            self.vector_store.text_collection.add(
                documents=full_chunk_documents,
                # embeddings=None,  # No embeddings needed for full chunks
                embeddings=np.zeros((len(full_chunk_documents), 512)),
                ids=full_chunk_ids,
                metadatas=full_metadatas,
            )

            self.vector_store.mark_content_processed(content_hash, "text", caption_path)
            print(f"Successfully processed {caption_path}:")
            print(f"  - Created {len(summaries)} LLM-generated summaries")
            print(f"  - Created {len(full_chunks)} full context chunks")
            print(
                f"  - Average full chunk size: {
                    sum(len(chunk.split()) for chunk in full_chunks) // len(full_chunks)
                } tokens"
            )

            set_file_status(
                video_id=video_id,
                column=self.csv_config["text_embed"],
                csv_path=csv_filename,
            )

            return True

        except Exception as e:
            print(f"Error processing caption for video {video_id}: {e}")
            return False

    def _clean_caption_text(self, caption_text: str) -> str:
        """Clean caption text by removing SRT formatting"""

        transcript_lines = caption_text.splitlines()
        clean_text = []
        seen_text = set()

        for line in transcript_lines:
            if "-->" in line:
                continue

            # skip index numbers
            if line.strip().isdigit():
                continue

            text = line.strip()
            # skip empty lines
            if not text:
                continue

            if text not in seen_text:
                seen_text.add(text)
                clean_text.append(text)

        return " ".join(clean_text)

    def process_single_video(self, df: pd.DataFrame, video_id: str) -> bool:
        """Process a single video through the complete pipeline"""
        print(f"\n{'=' * 50}")
        print(f"Processing video: {video_id}")
        print(f"{'=' * 50}")

        csv_filename = os.path.join(
            self.file_config["data_path"],
            f"{self.channel_config['channel_name']}_videos.csv",
        )

        video_url = self.get_video_url_by_id(df, video_id)
        if not video_url:
            return False

        video_path = get_file_path(video_id, file_type="video")
        caption_path = get_file_path(video_id, file_type="caption")

        success = True

        # Step 1: Ensure caption exists
        if not caption_path or not get_status(
            video_id=video_id,
            column_name=self.csv_config["captions"],
            csv_filename=csv_filename,
        ):
            print(f"Downloading caption for {video_id}")
            if download_single_caption(video_id=video_id, video_url=video_url):
                caption_path = get_file_path(video_id, file_type="caption")
            else:
                print(f"Failed to download caption for {video_id}")
                success = False

        # Step 2: Create video embeddings
        if (
            video_path
            and caption_path
            and not get_status(
                video_id=video_id,
                column_name=self.csv_config["video_embed"],
                csv_filename=csv_filename,
            )
        ):
            print(f"Creating image embeddings for {video_id}")
            if not self.create_image_embeddings_from_video(
                video_id=video_id,
                video_path=video_path,
                caption_path=caption_path,
                csv_filename=csv_filename,
            ):
                success = False

        print("caption path:", caption_path)
        print(
            "embedding status:",
            get_status(
                video_id=video_id,
                column_name=self.csv_config["text_embed"],
                csv_filename=csv_filename,
            ),
        )
        # Step 3: Create text embeddings
        if caption_path and not get_status(
            video_id=video_id,
            column_name=self.csv_config["text_embed"],
            csv_filename=csv_filename,
        ):
            print(f"Creating text embeddings for {video_id}")
            if not self.create_text_embeddings_from_caption(
                video_id=video_id,
                caption_path=caption_path,
                csv_filename=csv_filename,
            ):
                success = False

        status = "SUCCESS" if success else "FAILED"
        print(f"Video {video_id} processing: {status}")
        return success

    def run_pipeline(self, df: pd.DataFrame, max_new_downloads: int = 10):
        print("Starting Pipeline...")

        csv_filepath = os.path.join(
            self.file_config["data_path"],
            f"{self.channel_config['channel_name']}_videos.csv",
        )
        if max_new_downloads > 0:
            print(f"\nDownloading up to {max_new_downloads} new videos...")
            # only download not downloaded videos
            filtered_df = df[df[self.csv_config["video"]] == False]
            download_videos(
                filtered_df, caption_only=False, max_new_downloads=max_new_downloads
            )

        all_videos = get_id_based_on_column(
            column_name=self.csv_config["video"],
            csv_path=csv_filepath,
            file_present=True,
        )
        downloaded_video_ids = get_filtered_ids(
            csv_path=csv_filepath,
        )

        if not downloaded_video_ids:
            print("No downloaded videos found.")
            return

        print(f"\nFound total of {len(all_videos)}  videos")
        print(
            f"\n{len(all_videos) - len(downloaded_video_ids)} have embeddings present"
        )
        print(f"\nProcessing {len(downloaded_video_ids)} downloaded videos")

        # for specific video
        if args.video_id:
            if args.video_id in downloaded_video_ids:
                self.process_single_video(df, args.video_id)
            else:
                print(f"Video {args.video_id} not found in downloaded videos")
            return

        successful_count = 0
        failed_count = 0

        for i, video_id in enumerate(downloaded_video_ids, 1):
            print(f"\n[{i}/{len(downloaded_video_ids)}] Processing {video_id}")

            if self.process_single_video(df, video_id):
                successful_count += 1
            else:
                failed_count += 1

        print(f"\n{'=' * 50}")
        print("Pipeline completed!")
        print(f"Successfully processed: {successful_count} videos")
        print(f"Failed: {failed_count} videos")
        print(f"{'=' * 50}")


def main():
    if not api_key:
        print("Error: YOUTUBE_API environment variable not set")
        return

    pipeline = Pipeline()
    channel_config = get_config(path="configs/channel_config.yaml")
    channel_name = channel_config["channel_name"]
    channel_id = channel_config["channel_id"]

    cfg = get_config()
    data_path = cfg["file_config"]["data_path"]
    video_status_file_path = os.path.join(
        os.getcwd(), cfg["file_config"]["video_status_file"]
    )
    file_config = cfg["file_config"]
    csv_path = os.path.join(file_config["data_path"], f"{channel_name}_videos.csv")

    if not os.path.exists(csv_path):
        print(f"Fetching videos for channel: {channel_name}")
        videos = get_all_videos(
            api_key, channel_name=channel_name, channel_id=channel_id
        )
        print(f"Total videos found: {len(videos)}")

        if videos:
            for i, video in enumerate(videos[:5]):
                print(f"Video {i + 1}:")
                print(f"  Title: {video['title']}")
                print(f"  URL: {video['video_url']}")
                print(f"  Published: {video['published_at']}")
                print("-" * 50)

            df = pd.DataFrame(videos)
            os.makedirs(data_path, exist_ok=True)
            final_csv_path = os.path.join(data_path, f"{channel_name}_videos.csv")
            df.to_csv(final_csv_path, index=False, encoding="utf-8")

            df_id = pd.DataFrame(
                [video["video_id"] for video in videos], columns=["video_id"]
            )
            df_id.to_csv(video_status_file_path, index=False, encoding="utf-8")
            print(df.head())
        else:
            print("No videos found!")
            return
    else:
        df = pd.read_csv(csv_path)

    pipeline.run_pipeline(df, max_new_downloads=args.num)


if __name__ == "__main__":
    main()
