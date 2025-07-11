import os
import sys
import pandas as pd
from datetime import datetime
from typing import Optional, List, Tuple, Dict

from rich.text import Text
from rich.color import Color


def render_gradient_text(text_str, start_hex, end_hex):
    start_color = Color.parse(start_hex).triplet
    end_color = Color.parse(end_hex).triplet
    total_chars = len(text_str)

    gradient_text = Text()
    for i, char in enumerate(text_str):
        r = int(
            start_color[0] + (end_color[0] - start_color[0]) * i / (total_chars - 1)
        )
        g = int(
            start_color[1] + (end_color[1] - start_color[1]) * i / (total_chars - 1)
        )
        b = int(
            start_color[2] + (end_color[2] - start_color[2]) * i / (total_chars - 1)
        )
        hex_color = f"#{r:02X}{g:02X}{b:02X}"
        gradient_text.append(char, style=f"bold {hex_color}")

    return gradient_text


def print_frame_progress(current, total, bar_length=40):
    progress = int(bar_length * current / total)
    bar = "[" + "#" * progress + "-" * (bar_length - progress) + "]"
    sys.stdout.write(f"\r  {bar} Frame {current}/{total}")
    sys.stdout.flush()


def get_file_path(video_id: str, file_type: str = "video") -> Optional[str]:
    file_formats = {
        "video": (os.path.join("data", "videos"), ".mp4"),
        "caption": (os.path.join("data", "captions"), ".srt"),
    }

    if file_type not in file_formats:
        print("Invalid file type. Choose 'video' or 'caption'")
        return None

    relative_path, file_ext = file_formats[file_type]
    base_path = os.path.join(os.getcwd(), relative_path)

    if not os.path.exists(base_path):
        print(f"Directory not found: {base_path}")
        return None

    filename = f"{video_id}{file_ext}"
    full_path = os.path.join(base_path, filename)

    if os.path.exists(full_path):
        return full_path
    else:
        print(f"File not found: {filename} in {base_path}")
        return None


def get_all_files(file_type: str = "video") -> Optional[Tuple[List[str], List[str]]]:
    file_formats = {
        "video": (os.path.join("data", "videos"), ".mp4"),
        "caption": (os.path.join("data", "captions"), ".srt"),
    }

    if file_type not in file_formats:
        print("invalid. choose video or caption")
        return None

    relative_path, file_ext = file_formats[file_type]
    base_path = os.path.join(os.getcwd(), relative_path)

    if not os.path.exists(base_path):
        print(f"directory not found: {base_path}")
        return None

    all_files = os.listdir(base_path)
    matched_files = [f for f in all_files if f.endswith(file_ext)]

    if not matched_files:
        print(f"No {file_type} files with extension '{file_ext}' found in {base_path}")
        return None

    matched_files.sort()

    full_paths = [os.path.join(base_path, f) for f in matched_files]
    file_ids = [
        os.path.splitext(f)[0].replace(file_ext.replace(".", ""), "")
        for f in matched_files
    ]

    return full_paths, file_ids


def check_for_video_and_caption() -> Dict[str, Tuple[str, str]]:
    video_result = get_all_files(file_type="video")
    caption_result = get_all_files(file_type="caption")

    if not video_result or not caption_result:
        return {}

    video_paths, video_ids = video_result
    caption_paths, caption_ids = caption_result

    video_map = dict(zip(video_ids, video_paths))
    caption_map = dict(zip(caption_ids, caption_paths))

    matched_files = {}
    for video_id in video_map:
        if video_id in caption_map:
            matched_files[video_id] = (video_map[video_id], caption_map[video_id])
        else:
            print(f"Warning: No caption file found for video ID: {video_id}")

    for caption_id in caption_map:
        if caption_id not in video_map:
            print(f"Warning: No video file found for caption ID: {caption_id}")

    return matched_files


def set_file_status(video_id, column, csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} does not exist.")

    df = pd.read_csv(csv_path)

    if video_id not in df["video_id"].values:
        raise ValueError(f"Video ID {video_id} not found in {csv_path}.")

    if column not in df.columns:
        df[column] = False

    if "last_updated" not in df.columns:
        df["last_updated"] = ""

    df.loc[df["video_id"] == video_id, column] = True
    df.loc[df["video_id"] == video_id, "last_updated"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    df.to_csv(csv_path, index=False)


def get_filtered_ids(csv_filename):
    csv_path = os.path.join(os.getcwd(), csv_filename)
    if not os.path.exists(csv_path):
        print("No csv file exists")
        return []

    df = pd.read_csv(csv_path)

    required_columns = [
        "video_downloaded",
        "caption_downloaded",
        "video_embedding",
        "text_embedding",
    ]

    for col in required_columns + ["video_id"]:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            return []

    filtered_df = df[
        (df["video_downloaded"] == True)
        & (
            (df["caption_downloaded"] != True)
            | (df["video_embedding"] != True)
            | (df["text_embedding"] != True)
        )
    ]

    return filtered_df["video_id"].tolist()


def get_id_based_on_column(column_name, csv_filename, file_present=False):
    print("column name:", column_name)
    csv_path = os.path.join(os.getcwd(), csv_filename)
    if not os.path.exists(csv_path):
        print("No csv file exists")
        return

    df = pd.read_csv(csv_path)
    if column_name not in df.columns:
        df[column_name] = False
        df.to_csv(csv_path, index=False)

    if column_name == "video_downloaded":
        x = df[df["video_downloaded"] == file_present]
        return x["video_id"].tolist()

    filtered_files = df[
        (df["video_downloaded"] == True) & (df[column_name] == file_present)
    ]
    return filtered_files["video_id"].tolist()


def get_status(video_id, column_name, csv_filename):
    csv_path = os.path.join(os.getcwd(), csv_filename)
    if not os.path.exists(csv_path):
        print("No csv file exists")
        return None

    df = pd.read_csv(csv_path)
    if column_name not in df.columns:
        df[column_name] = False
    matching_row = df[df["video_id"] == video_id]

    if matching_row.empty:
        print(f"No entry found for video_id: {video_id}")
        return None

    return matching_row.iloc[0][column_name]


if __name__ == "__main_":
    get_all_files(video=True)
