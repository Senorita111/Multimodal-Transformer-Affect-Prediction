import torch
import pickle
from torch.utils.data import Dataset, DataLoader, random_split, Subset

class AudioVisualDataset(Dataset):
    def __init__(self, audio_data, video_data, labels):
        self.audio_data = audio_data
        self.video_data = video_data
        self.labels_data = labels

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        return self.audio_data[idx], self.video_data[idx], self.labels_data[idx]

def load_data(video_path, audio_path, labels_p_path, labels_a_path, labels_d_path):
    """Loads audio, video, and labels from preprocessed pickle files."""
    with open(video_path, 'rb') as f:
        video_tensor = torch.from_numpy(pickle.load(f)).permute(0, 3, 1, 2)
        sequence_length = 16
        num_videos = video_tensor.shape[0] // sequence_length
        video_tensor = video_tensor[:num_videos * sequence_length].view(num_videos, sequence_length, video_tensor.shape[1], video_tensor.shape[2], video_tensor.shape[3])
    
    with open(audio_path, 'rb') as f:
        audio_tensor = torch.from_numpy(pickle.load(f))
    
    with open(labels_p_path, 'rb') as f:
        labels_pleasure = torch.from_numpy(pickle.load(f)).float()
    with open(labels_a_path, 'rb') as f:
        labels_arousal = torch.from_numpy(pickle.load(f)).float()
    with open(labels_d_path, 'rb') as f:
        labels_dominance = torch.from_numpy(pickle.load(f)).float()
    
    labels_tensor = torch.stack((labels_pleasure, labels_arousal, labels_dominance), dim=1)
    return AudioVisualDataset(audio_tensor, video_tensor, labels_tensor)

def get_dataloaders(dataset, batch_size=1):
    """Splits dataset into train, validation, and test sets and returns DataLoaders."""
    dataset_length = len(dataset)
    test_split = int(0.9 * dataset_length)
    train_val_indices = list(range(0, test_split))
    test_indices = list(range(test_split, dataset_length))
    train_val_dataset = Subset(dataset, train_val_indices)
    test_dataset = Subset(dataset, test_indices)
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
    
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True))