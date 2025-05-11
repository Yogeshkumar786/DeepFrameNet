import os
import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import torch.nn as nn
from model import MultiModalModel

class MultiModalDataset(Dataset):
    def __init__(self, metadata, preprocessed_video_dir, tokenizer, max_len, max_frames=10):
        self.metadata = metadata
        self.video_dir = preprocessed_video_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_frames = max_frames

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        entry = self.metadata[index]
        # 'file' is like 'real/video1.mp4' or 'fake/video2.mp4'
        label = 1 if entry['n_fakes'] > 0 else 0
        video_path = entry['file']
        class_folder = os.path.dirname(video_path)  # 'real' or 'fake'
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        # Dummy text for demonstration
        label_text = "This is a fake video." if label == 1 else "This is a real video."
        text = f"{label_text} Filename: {video_name}."
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        frames_dir = os.path.join(self.video_dir, class_folder, video_name)
        video_features = self.load_video_frames(frames_dir)
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'video': torch.tensor(video_features, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def load_video_frames(self, dir_path):
        frames = []
        for file in sorted(os.listdir(dir_path))[:self.max_frames]:
            img = cv2.imread(os.path.join(dir_path, file))
            img = cv2.resize(img, (224, 224))
            img = img.transpose((2, 0, 1))  # CHW
            frames.append(img)
        while len(frames) < self.max_frames:
            frames.append(np.zeros((3, 224, 224)))  # Padding
        return np.stack(frames)

def load_metadata(path):
    with open(path, 'r') as f:
        return json.load(f)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # metadata = load_metadata("dataset/metadata.json")
    metadata = load_metadata("metadata.json")
    train_data, val_data = train_test_split(metadata, test_size=0.2, random_state=42)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train_set = MultiModalDataset(train_data, "preprocessed_videos", tokenizer, 128)
    val_set = MultiModalDataset(val_data, "preprocessed_videos", tokenizer, 128)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)
    labels = [1 if entry['n_fakes'] > 0 else 0 for entry in train_data]
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    weights = torch.tensor(weights, dtype=torch.float).to(device)
    text_model = RobertaModel.from_pretrained("roberta-base")
    model = MultiModalModel(text_model, num_labels=2, class_weights=weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    for epoch in range(3):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                video=batch['video'].to(device),
                labels=batch['labels'].to(device)
            )
            outputs['loss'].backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete.")
    # Save model + tokenizer
    os.makedirs('model', exist_ok=True)
    torch.save(model.state_dict(), 'model/model.pth')
    text_model.save_pretrained('model/text_model')
    tokenizer.save_pretrained('tokenizer')
    print("Model saved to model/model.pth")

if __name__ == "__main__":
    train()
