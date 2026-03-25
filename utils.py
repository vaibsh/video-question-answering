import torch
import cv2
from PIL import Image

def extract_frames(video_path, preprocess, max_frames=16):
    cap = cv2.VideoCapture(video_path)

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        raise ValueError(f"Video {video_path} has no frames")

    step = max(1, total_frames // max_frames)

    count = 0
    while len(frames) < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = preprocess(frame)
            frames.append(frame)

        count += 1

    cap.release()

    while len(frames) < max_frames:
        frames.append(frames[-1])

    frames = torch.stack(frames)
    return frames


def collate_fn(batch):
    frames, input_ids, attention_mask, labels = zip(*batch)

    frames = torch.stack(frames)
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    labels = torch.stack(labels)

    return frames, input_ids, attention_mask, labels