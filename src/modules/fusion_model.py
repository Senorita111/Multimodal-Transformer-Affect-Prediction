import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, VideoMAEForVideoClassification
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.init_weights()
    
    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = torch.mean(outputs[0], dim=1)
        return hidden_states

class ViViTModel(nn.Module):
    def __init__(self):
        super(ViViTModel, self).__init__()
        self.vivit = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base")
        self.vivit.classifier = nn.Identity()
        self.fc = nn.Linear(self.vivit.config.hidden_size, 3)
    
    def forward(self, x):
        outputs = self.vivit(pixel_values=x, output_hidden_states=True)
        return outputs.hidden_states[-1]

class AudioVisualTransformer(nn.Module):
    def __init__(self, num_classes, d_model=32, nhead=4, num_encoder_layers=3, dim_feedforward=32):
        super(AudioVisualTransformer, self).__init__()
        self.video_feature_extractor = ViViTModel()
        self.video_feature_extractor.load_state_dict(torch.load('best_video_model_mithos.pth')) # Load the Pretrained and fine-tuned model
        self.video_feature_extractor.fc = nn.Identity()


        model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.audio_feature_extractor = EmotionModel.from_pretrained(model_name)

        self.video_projection = nn.Linear(768, d_model)
        self.audio_projection = nn.Linear(1024, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.pleasure_head = nn.Linear(d_model, 1)
        self.arousal_head = nn.Linear(d_model, 1)
        self.dominance_head = nn.Linear(d_model, 1)
    
    def forward(self, audio_signal, video):
        video = video.float().permute(0, 2, 1, 3, 4).contiguous()
        audio = self.processor(audio_signal, sampling_rate=16000)['input_values'][0]
        audio = torch.from_numpy(audio).float().to(video.device).reshape(1, -1)

        video_features = self.video_feature_extractor(video)
        audio_features = self.audio_feature_extractor(audio)

        combined_features = self.video_projection(video_features) + self.audio_projection(audio_features)
        transformed_features = self.transformer_encoder(combined_features.permute(1, 0, 2))
        pooled_features = transformed_features.mean(dim=0)
        
        return self.pleasure_head(pooled_features), self.arousal_head(pooled_features), self.dominance_head(pooled_features)
