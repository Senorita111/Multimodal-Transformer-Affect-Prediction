import torch
import librosa
import numpy as np
import random
import pickle
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import yaml

# Load configuration
with open("audio_config.yaml", "r") as f:
    config = yaml.safe_load(f)

class EmotionDataset(Dataset):
    def __init__(self, file_paths, labels, processor, sample_rate=16000, max_length=400000):
        self.samples = []
        self.labels = []
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_length = max_length
        
        for idx, path in enumerate(file_paths):
            audio, rate = librosa.load(path, sr=sample_rate)
            # Truncate or pad the audio
            if len(audio) > max_length:
                start = random.randint(0, len(audio) - max_length)
                audio = audio[start:start + max_length]
            elif len(audio) < max_length:
                audio = np.pad(audio, (0, max_length - len(audio)), 'constant')
            self.samples.append(audio)
            self.labels.append(labels[idx])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio = self.samples[idx]
        label = self.labels[idx]
        audio_tensor = self.processor(audio, sampling_rate=self.sample_rate)['input_values'][0]
        return torch.tensor(audio_tensor, dtype=torch.float), torch.tensor(label, dtype=torch.float)

def get_dataloaders(batch_size):
    """Loads preprocessed audio data and splits it into train, validation, and test sets."""
    with open(config['data']['audio_paths'], 'rb') as f:
        file_paths = pickle.load(f)
    with open(config['data']['labels_path'], 'rb') as f:
        labels = pickle.load(f)
    
    processor = Wav2Vec2Processor.from_pretrained(config['model']['wav2vec_model'])
    dataset = EmotionDataset(file_paths, labels, processor)
    
    dataset_length = len(dataset)
    train_val_count = int(config['training']['train_split'] * dataset_length)
    test_count = dataset_length - train_val_count
    train_val_dataset = Subset(dataset, list(range(train_val_count)))
    test_dataset = Subset(dataset, list(range(train_val_count, dataset_length)))
    train_count = int(config['training']['val_split'] * len(train_val_dataset))
    val_count = len(train_val_dataset) - train_count
    train_dataset, val_dataset = random_split(train_val_dataset, [train_count, val_count])
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    )