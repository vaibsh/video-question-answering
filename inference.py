import torch
import cv2
from PIL import Image

def load_video(video_path, preprocess, max_frames=8):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = preprocess(frame)
        frames.append(frame)

    cap.release()

    return torch.stack(frames)


def run_inference(model, video_path, question, device, tokenizer, preprocess):
    model.eval()

    frames = load_video(video_path, preprocess)
    frames = frames.unsqueeze(0).to(device)

    with torch.no_grad():
        output_tokens = model.generate(frames, [question], tokenizer)

    answer = tokenizer.decode(output_tokens[0].cpu().numpy(),
                              skip_special_tokens=True)

    print("\nQuestion:", question)
    print("Answer:", answer)