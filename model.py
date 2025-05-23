# import torch
# import torch.nn as nn
# from torchvision import models
# from transformers import RobertaModel
# from torch.cuda.amp import autocast

# class CrossModalAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads=8, num_layers=2):
#         super(CrossModalAttention, self).__init__()
#         self.attn_layers = nn.ModuleList([
#             nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
#             for _ in range(num_layers)
#         ])
#         self.norm_layers = nn.ModuleList([
#             nn.LayerNorm(embed_dim) for _ in range(num_layers)
#         ])

#     def forward(self, query, key, value):
#         for attn, norm in zip(self.attn_layers, self.norm_layers):
#             attn_output, _ = attn(query, key, value)
#             query = norm(query + attn_output)
#         return query

# class MultiModalModel(nn.Module):
#     def __init__(self, text_model, num_labels, class_weights=None):
#         super(MultiModalModel, self).__init__()
#         self.text_model = text_model

#         self.video_cnn = models.resnet18(pretrained=True)
#         self.video_cnn.fc = nn.Linear(self.video_cnn.fc.in_features, 768)

#         self.cross_modal_attn = CrossModalAttention(768, num_heads=8, num_layers=2)
#         self.classifier = nn.Linear(768 * 2, num_labels)
#         self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
#         self.dropout = nn.Dropout(p=0.3)

#     def forward(self, input_ids, attention_mask, video=None, labels=None):
#         with autocast():
#             outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
#             text_features_embedding = outputs.last_hidden_state

#             if video is not None:
#                 batch_size, num_frames, channels, height, width = video.size()
#                 video = video.view(batch_size * num_frames, channels, height, width)
#                 video_features = self.video_cnn(video)
#                 video_features = video_features.view(batch_size, num_frames, -1)
#                 video_features = torch.mean(video_features, dim=1).unsqueeze(1).expand(-1, text_features_embedding.size(1), -1)
#             else:
#                 video_features = torch.zeros(text_features_embedding.size(0), text_features_embedding.size(1), 768).to(text_features_embedding.device)

#             video_attn = self.cross_modal_attn(video_features, text_features_embedding, text_features_embedding)

#             combined_features = torch.cat((text_features_embedding[:, 0, :], video_attn[:, 0, :]), dim=1)
#             combined_features = self.dropout(combined_features)
#             logits = self.classifier(combined_features)

#         if labels is not None:
#             loss = self.loss_fn(logits, labels)
#             return {"loss": loss, "logits": logits}
#         else:
#             return {"logits": logits}


import torch
import torch.nn as nn
from torchvision import models
from transformers import RobertaModel
from torch.cuda.amp import autocast

class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_layers=2):
        super(CrossModalAttention, self).__init__()
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])

    def forward(self, query, key, value):
        for attn, norm in zip(self.attn_layers, self.norm_layers):
            attn_output, _ = attn(query, key, value)
            query = norm(query + attn_output)
        return query

class MultiModalModel(nn.Module):
    def __init__(self, text_model, num_labels, class_weights=None):
        super(MultiModalModel, self).__init__()
        self.text_model = text_model
        self.video_cnn = models.resnet18(pretrained=True)
        self.video_cnn.fc = nn.Linear(self.video_cnn.fc.in_features, 768)
        self.cross_modal_attn = CrossModalAttention(768, num_heads=8, num_layers=2)
        self.classifier = nn.Linear(768 * 2, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask, video=None, labels=None):
        with autocast():
            outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_features_embedding = outputs.last_hidden_state
            if video is not None:
                batch_size, num_frames, channels, height, width = video.size()
                video = video.view(batch_size * num_frames, channels, height, width)
                video_features = self.video_cnn(video)
                video_features = video_features.view(batch_size, num_frames, -1)
                video_features = torch.mean(video_features, dim=1).unsqueeze(1).expand(-1, text_features_embedding.size(1), -1)
            else:
                video_features = torch.zeros(text_features_embedding.size(0), text_features_embedding.size(1), 768).to(text_features_embedding.device)
            video_attn = self.cross_modal_attn(video_features, text_features_embedding, text_features_embedding)
            combined_features = torch.cat((text_features_embedding[:, 0, :], video_attn[:, 0, :]), dim=1)
            combined_features = self.dropout(combined_features)
            logits = self.classifier(combined_features)
            if labels is not None:
                loss = self.loss_fn(logits, labels)
                return {"loss": loss, "logits": logits}
            else:
                return {"logits": logits}
