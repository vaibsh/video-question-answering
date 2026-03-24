import cv2
from PIL import Image

def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frames = []

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError(f"No frames in video: {video_path}")

    step = max(total_frames // num_frames, 1)

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()

        if not ret:
            break

        # 🔥 IMPORTANT FIXES
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR → RGB
        frame = cv2.resize(frame, (224, 224))           # resize
        frame = Image.fromarray(frame)                  # numpy → PIL

        frames.append(frame)

    cap.release()

    # 🚨 Safety check
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from {video_path}")

    return frames

import torch

def collate_fn(batch, pad_token_id):
    frames, questions, answers = zip(*batch)

    # Stack frames
    frames = torch.stack(frames)

    # Keep questions as list (strings)
    questions = list(questions)

    # Pad answers
    lengths = [len(a) for a in answers]
    max_len = max(lengths)

    padded_answers = torch.full(
                    (len(answers), max_len),
                    pad_token_id,
                    dtype=torch.long
                    )

    for i, a in enumerate(answers):
        padded_answers[i, :len(a)] = a

    return frames, questions, padded_answers