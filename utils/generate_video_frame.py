import argparse
import os
import cv2
import pandas as pd
import shutil
from utils.helper import set_file_status, get_all_files
from utils.helper import print_frame_progress


class Frames:
    def __init__(self, frame):
        self.frame = frame


class Store:
    def __init__(self):
        self.frames = []


def is_scene_change(prev_frame, curr_frame, threshold=0.5):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    hist_prev = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
    hist_curr = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])

    cv2.normalize(hist_prev, hist_prev)
    cv2.normalize(hist_curr, hist_curr)

    score = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)

    return score < threshold, score


def store_frames(csv_path):
    os.makedirs(csv_path, exist_ok=True)


def is_folder_empty(folder_path):
    return not any(os.scandir(folder_path))


def remove_empty_video_folders(csv_path):
    home_path = os.path.dirname(os.getcwd())
    frames_root = os.path.join(home_path, "data", "video_frames")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print("csv not found")
        return

    to_remove = []

    for video_id in df["video_id"]:
        folder_path = os.path.join(frames_root, video_id)
        if os.path.exists(folder_path) and is_folder_empty(folder_path):
            print(f"empty folder : {video_id} — removing folder and entry.")
            shutil.rmtree(folder_path)
            to_remove.append(video_id)

    df = df[~df["video_id"].isin(to_remove)]
    df.to_csv(csv_path, index=False)


def create_video_embeddings(video_path, column_name):
    current_path = os.getcwd()
    video_id = os.path.basename(video_path).split(".")[0]

    print(f"\nProcessing: {video_id}", flush=True)
    save_folder = os.path.join(current_path, "data", "video_frames", video_id)
    os.makedirs(save_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, prev_frame = cap.read()

    if not success:
        print(f"Warning: Couldn't read first frame of {video_id}, skipping.")
        cap.release()
        return

    frame_idx = 1
    saved_idx = 0

    while True:
        success, curr_frame = cap.read()
        if not success:
            print(
                f"Warning: Failed to read frame {frame_idx} of {
                    video_id
                }, skipping frame."
            )
            frame_idx += 1
            continue

        try:
            is_change, score = is_scene_change(
                prev_frame, curr_frame, threshold=0.5)
        except Exception as e:
            print(
                f"\n[Error] Scene change detection failed at frame {frame_idx} of '{
                    video_id
                }': {e}",
                flush=True,
            )
            frame_idx += 1
            print_frame_progress(frame_idx, total_frames)
            continue

        if is_change:
            filename = f"scene_{saved_idx:04d}_frame{frame_idx}.jpg"
            file_path = os.path.join(save_folder, filename)
            cv2.imwrite(file_path, curr_frame)
            saved_idx += 1

        prev_frame = curr_frame
        print_frame_progress(frame_idx, total_frames)
        frame_idx += 1

        # may go beyond totla frames, so
        if frame_idx > total_frames:
            break

    cap.release()
    set_file_status(video_id=video_id, column=column_name)


def main(args):
    column_name = "frames_extracted"
    current_path = os.getcwd()
    csv_path = os.path.join(current_path, args.csv_path)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print("No csv found")

    if column_name not in df.columns:
        df[column_name] = False

    filtered_files = df[(df["downloaded"] == True) &
                        (df[column_name] == False)]
    filtered_ids = filtered_files["video_id"].tolist()
    all_video_path, _ = get_all_files(video=True)

    for video_path in all_video_path:
        video_id = os.path.basename(video_path).split(".")[0]
        if video_id not in filtered_ids:
            continue
        create_video_embeddings(video_path, column_name)


def get_csv_path():
    current_path = os.getcwd()
    home_path = os.path.dirname(current_path)
    csv_path = os.path.join(home_path, "processed_video_frames.csv")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare histogram")

    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        default=0.5,
        help="histogram threshold",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="data/video_frames",
        help="histogram threshold",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=False,
        default="video_status.csv",
        help="histogram threshold",
    )
    args = parser.parse_args()

    main(args)
    # csv_path = get_csv_path()
    # remove_empty_video_folders(csv_path)
