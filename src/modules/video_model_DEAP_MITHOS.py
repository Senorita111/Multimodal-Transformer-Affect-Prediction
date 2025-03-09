import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification
import yaml

# Load config
with open("video_config_DEAP_MITHOS.yaml", "r") as f:
    config = yaml.safe_load(f)

class ViViTModelMITHOS(nn.Module):
    def __init__(self, pretrained_model_path):
        super(ViViTModelMITHOS, self).__init__()
        self.vivit = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base")
        self.vivit.classifier = nn.Identity()
        self.fc = nn.Linear(self.vivit.config.hidden_size, 3)
        
        # Load pretrained DEAP model
        self.load_pretrained_weights(pretrained_model_path)
    
    def load_pretrained_weights(self, model_path):
        """Loads the pretrained DEAP model."""
        self.load_state_dict(torch.load(model_path), strict=False)
        print(f"Loaded pretrained DEAP model from {model_path}")
        
    def forward(self, x):
        outputs = self.vivit(pixel_values=x, output_hidden_states=False)
        return self.fc(outputs.logits)
