import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification

class ViViTModel(nn.Module):
    def __init__(self):
        super(ViViTModel, self).__init__()
        self.vivit = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base")
        self.vivit.classifier = nn.Identity()
        self.fc = nn.Linear(self.vivit.config.hidden_size, 3)  # Predicting Pleasure, Arousal, Dominance (PAD) each
    
    def forward(self, x):
        outputs = self.vivit(pixel_values=x, output_hidden_states=False)
        hidden_state = outputs.logits 
        return self.fc(hidden_state)