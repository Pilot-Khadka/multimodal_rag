"""
usage:
    python transcript_processor.py --input INPUT_LOCATION

    -> vanilla parsing with dictionary
        -> Space: O(N); N-> number of frames
        -> Time: O(1)

    -> Frame range
        -> Space: O(n) ;n -> len of subtitles (n<<N)
        -> Time: O(log n) (binary search)
"""

import os
import re
from typing import Dict, Tuple, List, Optional

from utils.helper import set_file_status, get_id_based_on_column
from utils.general import get_config


class FrameRange:
    def __init__(self):
        self.ranges: List[Tuple[int, int, int]] = []

    def add_range(self, start_frame: int, end_frame: int, caption: str):
        self.ranges.append((start_frame, end_frame, caption))
        self.ranges.sort(key=lambda x: x[0])

    def get_caption(self, frame: int) -> Optional[str]:
        if not self.ranges:
            return None

        left, right = 0, len(self.ranges) - 1
        res = -1

        while left <= right:
            mid = left + (right - left) // 2
            start_frame = self.ranges[mid][0]

            if start_frame <= frame:
                res = mid
                left = mid + 1
            else:
                right = mid - 1

        if res >= 0:
            start, end, caption = self.ranges[res]
            if start <= frame <= end:
                return caption
        return None

    def __len__(self):
        return len(self.ranges)


def timestamp_to_frame(timestamp: str, fps: float) -> int:
    hours, minutes, seconds = re.split("[:]", timestamp)
    total_seconds = (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds.split(",")[0])
        + float(seconds.split(",")[1]) / 1000
    )
    return int(float(total_seconds) * fps)


def parse_srt_to_frame_map(input_path: str, fps: float) -> Dict[int, str]:
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    range_map = FrameRange()
    i = 0
    while i < len(lines):
        if lines[i].strip().isdigit():
            i += 1

        if i < len(lines) and "-->" in lines[i]:
            start_time, end_time = [t.strip() for t in lines[i].split("-->")]
            i += 1

            caption_lines = []
            while i < len(lines) and lines[i].strip():
                caption_lines.append(lines[i].strip())
                i += 1

            caption = " ".join(caption_lines)
            start_frame = timestamp_to_frame(start_time, fps)
            end_frame = timestamp_to_frame(end_time, fps)

            range_map.add_range(start_frame, end_frame, caption)
        i += 1

    return range_map


def clean_file(input_path: str, output_dir: str = None) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    transcript_lines = content.splitlines()
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

    filename = os.path.basename(input_path)

    if output_dir:
        output_file_path = os.path.join(
            output_dir, filename.replace(".srt", ".txt"))
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(" ".join(clean_text))

        print(f"Cleaned caption saved to {output_file_path}")
    return " ".join(clean_text)


def main():
    config = get_config()
    caption_dir = config["caption_path"]
    output_dir = config["cleaned_caption_path"]
    os.makedirs(output_dir, exist_ok=True)

    current_path = os.getcwd()
    caption_path = os.path.join(current_path, caption_dir)
    output_path = os.path.join(current_path, output_dir)

    csv_path = os.path.join(current_path, config["video_status_file"])

    filtered_files = get_id_based_on_column(column_name="captions_cleaned")
    all_captions = [f for f in os.listdir(caption_path) if f.endswith(".srt")]
    for caption in all_captions:
        caption_id = caption.split(".")[0]
        if caption_id not in filtered_files:
            continue
        current_caption = os.path.join(caption_path, caption)
        clean_file(input_path=current_caption, output_dir=output_path)
        set_file_status(
            csv_path=csv_path, video_id=caption_id, column="captions_cleaned"
        )
    print("Processing captions completed")


if __name__ == "__main__":
    main()
