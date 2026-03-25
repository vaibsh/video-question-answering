import torch
import argparse

from model import VideoQAModel
from dataset import VideoQADataset
from train import train
from utils import collate_fn
from config import Config
from transformers import AutoTokenizer
import clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--question", type=str, default=None)

    args = parser.parse_args()

    config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if args.mode == "train":

        # ✅ Train dataset
        train_dataset = VideoQADataset(
            json_path=config.TRAIN_JSON,
            video_dir=config.VIDEO_DIR,
            tokenizer=tokenizer,
            preprocess=preprocess
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=collate_fn
        )

        # ✅ Validation dataset
        val_dataset = VideoQADataset(
            json_path=config.VAL_JSON,
            video_dir=config.VIDEO_DIR,
            tokenizer=tokenizer,
            preprocess=preprocess
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn
        )

        model = VideoQAModel().to(device)
        model.decoder.config.pad_token_id = tokenizer.pad_token_id
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        train(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            num_epochs=5
        )

        torch.save(model.state_dict(), config.MODEL_PATH)

    elif args.mode == "infer":
        model = VideoQAModel().to(device)
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
        model.decoder.config.pad_token_id = tokenizer.pad_token_id
        model.eval()

        from inference import run_inference

        run_inference(
            model,
            args.video_path,
            args.question,
            device,
            tokenizer,
            preprocess
        )


if __name__ == "__main__":
    main()