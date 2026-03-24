import torch
from torch.utils.data import Dataset
import json
import os
import cv2
from utils import extract_frames

class VideoQADataset(Dataset):
    def __init__(self, json_path, video_dir, tokenizer, preprocess, max_frames=16):
        self.data = json.load(open(json_path))
        self.video_dir = video_dir
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.max_frames = max_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        video_path = os.path.join(self.video_dir, f"video{item['video_id']}.mp4")

        # Load frames
        frames = extract_frames(video_path)

        # Limit frames
        frames = frames[:self.max_frames]

        # Handle empty video
        if len(frames) == 0:
            raise ValueError(f"No frames found in {video_path}")

        # Pad frames if fewer than max_frames
        while len(frames) < self.max_frames:
            frames.append(frames[-1])

        # Apply CLIP preprocessing correctly
        processed_frames = []
        for f in frames:
            f = self.preprocess(f)
            processed_frames.append(f)

        frames = torch.stack(processed_frames)

        question = item['question']

        # Tokenize answer
        answer_tokens = torch.tensor(
            self.tokenizer.encode(item['answer']),
            dtype=torch.long
        )

        return frames, question, answer_tokens