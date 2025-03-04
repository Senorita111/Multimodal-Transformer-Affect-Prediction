import pandas as pd
import torch
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, file_paths, labels, processor, sample_rate=16000, max_length=400000):
        self.samples = []
        self.labels = []
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_length = max_length
        
        for idx, path in enumerate(file_paths):
            audio, rate = librosa.load(path, sr=sample_rate)
            # Truncate or pad audio
            if len(audio) > max_length:
                audio = audio[:max_length]
            elif len(audio) < max_length:
                audio = np.pad(audio, (0, max_length - len(audio)), 'constant')
            self.samples.append(audio)
            self.labels.append(labels[idx])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio = self.samples[idx]
        label = self.labels[idx]
        audio_tensor = torch.tensor(audio, dtype=torch.float)
        label_tensor = torch.tensor(label, dtype=torch.float)
        return audio_tensor, label_tensor