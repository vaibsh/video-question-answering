import torch
from torch.utils.data import Dataset
import json
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

        video_path = f"{self.video_dir}/video{item['video_id']}.mp4"
        frames = extract_frames(video_path, self.preprocess, self.max_frames)

        question = item["question"]
        answer = item["answer"]

        # Prompt (NO answer included here)
        prompt = f"Question: {question} Answer:"

        # Full sequence (prompt + answer)
        full_text = prompt + " " + answer + self.tokenizer.eos_token

        encoded = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=40,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Create labels
        labels = input_ids.clone()

        # Mask prompt tokens
        prompt_ids = self.tokenizer(
            prompt,
            truncation=True,
            max_length=40,
            return_tensors="pt"
        )["input_ids"].squeeze(0)

        labels[:len(prompt_ids)] = -100

        return frames, input_ids, attention_mask, labels