import torch
import json
from tqdm import tqdm
from config import Config
from dataset import VideoQADataset
from model import VideoQAModel
from transformers import AutoTokenizer
import clip
from torch.utils.data import DataLoader


# ✅ MUST be top-level (fixes multiprocessing error)
def inference_collate(batch):
    frames, input_ids, attention_mask = zip(*batch)

    frames = torch.stack(frames)
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)

    return frames, input_ids, attention_mask


def main():
    print("Running inference...")

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = VideoQAModel().to(device)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.decoder.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    # Dataset (🔥 FIX)
    test_dataset = VideoQADataset(
        json_path=config.TEST_JSON,
        video_dir=config.VIDEO_DIR,
        tokenizer=tokenizer,
        preprocess=preprocess,
        max_frames=config.MAX_FRAMES,
        is_inference=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=inference_collate,
        num_workers=config.NUM_WORKERS,  # set to 0 if issues
        pin_memory=True
    )

    results = []

    for frames, input_ids, attention_mask in tqdm(test_loader, desc="Inference"):
        frames = frames.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model.generate(
                frames,
                input_ids,
                attention_mask,
                max_new_tokens=20
            )

        for i in range(outputs.size(0)):
            decoded = tokenizer.decode(outputs[i], skip_special_tokens=True)

            answer = decoded.split("Answer:")[-1].strip()

            question_text = tokenizer.decode(
                input_ids[i],
                skip_special_tokens=True
            ).split("Answer:")[0].replace("Question:", "").strip()

            results.append({
                "question": question_text,
                "answer": answer
            })

    with open(config.OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Inference done! Saved to {config.OUTPUT_JSON}")


if __name__ == "__main__":
    main()