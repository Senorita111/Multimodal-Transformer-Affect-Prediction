import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
import yaml

# Load configuration
with open("audio_config.yaml", "r") as f:
    config = yaml.safe_load(f)

class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x.squeeze()

class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.arousal_head = RegressionHead(config)
        self.valence_head = RegressionHead(config)
        self.dominance_head = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state.mean(dim=1)
        arousal_output = self.arousal_head(hidden_states)
        valence_output = self.valence_head(hidden_states)
        dominance_output = self.dominance_head(hidden_states)
        return arousal_output, valence_output, dominance_output
