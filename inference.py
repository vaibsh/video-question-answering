import torch
from utils import extract_frames

def run_inference(model, video_path, question, device, tokenizer, preprocess):
    model.eval()

    frames = extract_frames(video_path, preprocess)
    frames = frames.unsqueeze(0).to(device)

    prompt = f"Question: {question} Answer:"

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        output = model.generate(frames, input_ids, attention_mask)

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("Answer:")[-1].strip()

    print("\nQuestion:", question)
    print("Answer:", answer)