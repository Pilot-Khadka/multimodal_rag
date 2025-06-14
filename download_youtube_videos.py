import os
import argparse
import subprocess
import pandas as pd
from typing import List, Dict
from googleapiclient.discovery import build

from utils.helper import set_file_status, get_file_path
from configs.settings import CsvConfig, FileConfig

api_key = os.environ.get("YOUTUBE_API")
csv_config = CsvConfig()
file_config = FileConfig()

parser = argparse.ArgumentParser(description="Download youtube videos")
parser.add_argument(
    "--num", type=int, required=True, help="number of videos to download"
)
args = parser.parse_args()


def get_all_videos(api_key, channel_name=None, channel_id=None) -> List[Dict]:
    youtube = build("youtube", "v3", developerKey=api_key)

    if channel_name and not channel_id:
        channel_request = youtube.channels().list(part="id", forUsername=channel_name)
        channel_response = channel_request.execute()

        if not channel_response.get("items"):
            print(f"No channel found with username: {channel_name}")
            return []

        channel_id = channel_response["items"][0]["id"]
        print(f"Found channel ID: {channel_id}")

    if not channel_id:
        print("Error: Either channel name or channel ID must be provided")
        return []

    channel_request = youtube.channels().list(part="contentDetails", id=channel_id)
    channel_response = channel_request.execute()
    uploads_playlist_id = channel_response["items"][0]["contentDetails"][
        "relatedPlaylists"
    ]["uploads"]

    videos = []
    next_page_token = None

    while True:
        playlist_request = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=next_page_token,
        )
        playlist_response = playlist_request.execute()

        video_ids = [
            item["contentDetails"]["videoId"] for item in playlist_response["items"]
        ]

        # durations for all videos in the current batch
        video_request = youtube.videos().list(
            part="contentDetails", id=",".join(video_ids)
        )
        video_response = video_request.execute()

        durations = {
            item["id"]: item["contentDetails"]["duration"]
            for item in video_response["items"]
        }

        for item in playlist_response["items"]:
            video_id = item["contentDetails"]["videoId"]
            duration = durations.get(video_id, "")
            # skip Shorts: videos shorter than 60 seconds (duration starts with 'PT' and ends in 'S')
            if (
                duration.startswith("PT")
                and "M" not in duration
                and ("H" not in duration and int(duration.strip("PTS")) < 60)
            ):
                continue

            video_data = {
                "title": item["snippet"]["title"],
                "video_id": video_id,
                "published_at": item["snippet"]["publishedAt"],
                "video_url": f"https://youtube.com/watch?v={video_id}",
            }
            videos.append(video_data)

        next_page_token = playlist_response.get("nextPageToken")
        if not next_page_token:
            break

    return videos


def load_video_ids_from_csv(csv_path: str, column_name: str):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if column_name in df.columns:
            return set(pd.read_csv(csv_path)[column_name])
    return set()


def download_single_caption(video_id: str, video_url: str) -> bool:
    existing_caption = get_file_path(video_id, "caption")
    if existing_caption:
        print(f"Caption already exists for {video_id}: {existing_caption}")
        return True

    path = os.getcwd()
    video_path = os.path.join(path, file_config.video_path)
    caption_path = os.path.join(path, file_config.caption_path)
    csv_path = os.path.join(path, file_config.video_status_file)
    os.makedirs(caption_path, exist_ok=True)

    print(f"Downloading caption for video ID: {video_id}")
    print(f"URL: {video_url}")

    try:
        subprocess.run(
            [
                "yt-dlp",
                "--skip-download",
                "--write-sub",
                "--sub-lang",
                "en",
                "--convert-subs",
                "srt",
                "--output",
                os.path.join(caption_path, f"{video_id}.%(ext)s"),
                video_url,
            ],
            check=True,
        )

        srt_file = f"{video_id}.en.srt"
        srt_path_in_caption = os.path.join(caption_path, srt_file)
        srt_path_in_video = os.path.join(video_path, srt_file)
        final_srt_path = os.path.join(caption_path, f"{video_id}.srt")

        if os.path.exists(srt_path_in_caption):
            os.rename(srt_path_in_caption, final_srt_path)
        elif os.path.exists(srt_path_in_video):
            os.rename(srt_path_in_video, final_srt_path)
        else:
            print(f"Caption file not found after download for {video_id}")
            return False

        set_file_status(csv_path=csv_path, video_id=video_id, column=csv_config.caption)
        return True

    except subprocess.CalledProcessError as e:
        print(f"Failed to download caption for {video_id}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error downloading caption for {video_id}: {e}")
        return False


def download_videos(df: pd.DataFrame, caption_only=False, max_new_downloads=10):
    path = os.getcwd()
    video_path = os.path.join(path, file_config.video_path)
    caption_path = os.path.join(path, file_config.caption_path)
    csv_path = os.path.join(path, file_config.video_status_file)

    os.makedirs(video_path, exist_ok=True)
    os.makedirs(caption_path, exist_ok=True)

    df = df[~df[csv_config.video]]
    downloaded_count = 0
    for idx, row in df.iterrows():
        if downloaded_count >= max_new_downloads:
            break

        url = row["video_url"]
        video_id = url.split("v=")[-1].split("&")[0]
        video_id = str(video_id)

        print(f"Downloading: {url}")

        try:
            if caption_only:
                subprocess.run(
                    [
                        "yt-dlp",
                        "--quiet",
                        "--no-warnings",
                        "--skip-download",
                        "--write-sub",
                        "--sub-lang",
                        "en",
                        "--convert-subs",
                        "srt",
                        "--output",
                        os.path.join(caption_path, f"{video_id}.%(ext)s"),
                        url,
                    ],
                    check=True,
                )
            else:
                subprocess.run(
                    [
                        "yt-dlp",
                        "--quiet",
                        "--no-warnings",
                        "-f",
                        "bv*[vcodec^=avc1][height<=480]+ba[acodec^=mp4a]/b[ext=mp4]",
                        "--merge-output-format",
                        "mp4",
                        "--write-auto-sub",
                        "--sub-lang",
                        "en",
                        "--convert-subs",
                        "srt",
                        "--output",
                        os.path.join(video_path, f"{video_id}.%(ext)s"),
                        url,
                    ],
                    check=True,
                )
                set_file_status(
                    csv_path=csv_path, video_id=video_id, column=csv_config.video
                )

            srt_file = f"{video_id}.en.srt"
            srt_path = os.path.join(video_path, srt_file)
            if os.path.exists(srt_path):
                os.rename(srt_path, os.path.join(caption_path, f"{video_id}.srt"))

            set_file_status(
                csv_path=csv_path, video_id=video_id, column=csv_config.caption
            )
            downloaded_count += 1

        except subprocess.CalledProcessError as e:
            print(f"Failed to download captions for {url}: {e}")


def download_single_video(df, video_id, caption_only=False):
    path = os.getcwd()
    video_path = os.path.join(path, file_config.video_path)
    caption_path = os.path.join(path, file_config.caption_path)
    csv_path = os.path.join(path, file_config.video_status_file)

    os.makedirs(video_path, exist_ok=True)
    os.makedirs(caption_path, exist_ok=True)

    df_status = pd.read_csv(csv_path)
    df_merged = pd.merge(df_status, df, on="video_id", how="inner")
    url = df_merged["video_url"].iloc[0]

    if caption_only:
        subprocess.run(
            [
                "yt-dlp",
                "--skip-download",
                "--write-sub",
                "--sub-lang",
                "en",
                "--convert-subs",
                "srt",
                "--output",
                os.path.join(caption_path, f"{video_id}.%(ext)s"),
                url,
            ],
            check=True,
        )
    else:
        subprocess.run(
            [
                "yt-dlp",
                "-f",
                "bv*[vcodec^=avc1][height<=480]+ba[acodec^=mp4a]/b[ext=mp4]",
                "--merge-output-format",
                "mp4",
                "--write-auto-sub",
                "--sub-lang",
                "en",
                "--convert-subs",
                "srt",
                "--output",
                os.path.join(video_path, f"{video_id}.%(ext)s"),
                url,
            ],
            check=True,
        )
        set_file_status(csv_path=csv_path, video_id=video_id, column=csv_config.video)

    srt_file = f"{video_id}.en.srt"
    srt_path = os.path.join(video_path, srt_file)
    if os.path.exists(srt_path):
        os.rename(srt_path, os.path.join(caption_path, f"{video_id}.srt"))

    set_file_status(csv_path=csv_path, video_id=video_id, column=csv_config.caption)


if __name__ == "__main__":
    channel_name = "1veritasium"
    if not os.path.exists(f"{channel_name}_videos.csv"):
        videos = get_all_videos(api_key, channel_name=channel_name)
        print(f"Total videos found: {len(videos)}")

        for i, video in enumerate(videos[:5]):
            print(f"Video {i + 1}:")
            print(f"  Title: {video['title']}")
            print(f"  URL: {video['video_url']}")
            print(f"  Published: {video['published_at']}")
            print("-" * 50)

        df = pd.DataFrame(videos)
        df.to_csv(f"{channel_name}_videos.csv", index=False, encoding="utf-8")

        df_id = pd.DataFrame(
            [video["video_id"] for video in videos], columns=["video_id"]
        )
        df_id.to_csv("video_status.csv", index=False, encoding="utf-8")
        print(df.head())
    else:
        df = pd.read_csv(f"{channel_name}_videos.csv")
        download_videos(df, caption_only=False, max_new_downloads=args.num)
