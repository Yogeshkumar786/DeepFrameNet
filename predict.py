import torch
import cv2
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from model import MultiModalModel
import os

def predict_video(video_path):
    # --- 1. Preprocess video frames ---
    def extract_frames(video_path, interval=30, max_frames=10):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % interval == 0:
                frame = cv2.resize(frame, (224, 224))
                frame = frame.transpose(2, 0, 1)  # CHW format
                frames.append(frame)
            frame_count += 1
        cap.release()
        # Pad if not enough frames
        while len(frames) < max_frames:
            frames.append(np.zeros((3, 224, 224)))
        return np.stack(frames)

    # --- 2. Prepare dummy text input ---
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    dummy_text = "This is a video sample."
    encoding = tokenizer.encode_plus(
        dummy_text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # --- 3. Load model and weights ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model = RobertaModel.from_pretrained("model/text_model")
    model = MultiModalModel(text_model, num_labels=2).to(device)

    # Load weights safely (ignore loss_fn weights)
    state_dict = torch.load("model/model.pth", map_location=device)
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("loss_fn")}
    model.load_state_dict(filtered_state_dict)
    model.eval()

    # --- 4. Run prediction ---
    with torch.no_grad():
        video_frames = extract_frames(video_path)
        video_tensor = torch.tensor(video_frames).unsqueeze(0).float().to(device)

        outputs = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device),
            video=video_tensor
        )

        probs = torch.softmax(outputs['logits'], dim=1)
        prediction = torch.argmax(probs).item()
        confidence = probs[0][prediction].item()

    return "Fake" if prediction == 1 else "Real", confidence

# --- 5. Run prediction ---
if __name__ == "__main__":
    result, confidence = predict_video("MY.mp4")
    print(f"Prediction: {result} (Confidence: {confidence:.2f})")
