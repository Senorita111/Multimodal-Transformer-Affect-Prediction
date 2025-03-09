import torch
import pickle
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import yaml

# Load configuration
with open("video_config_DEAP_MITHOS.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_mithos_data():
    """Loads and preprocesses the MITHOS dataset."""
    with open(config['data']['video_path'], 'rb') as f:
        video_data = torch.from_numpy(pickle.load(f)).float()
    
    with open(config['data']['labels_p_path'], 'rb') as f:
        labels_p = torch.from_numpy(pickle.load(f)).float()
    with open(config['data']['labels_a_path'], 'rb') as f:
        labels_a = torch.from_numpy(pickle.load(f)).float()
    with open(config['data']['labels_d_path'], 'rb') as f:
        labels_d = torch.from_numpy(pickle.load(f)).float()
    
    labels = torch.cat((labels_p.unsqueeze(1), labels_a.unsqueeze(1), labels_d.unsqueeze(1)), dim=1)
    
    # Reshape data to match ViViT input format
    sequence_length = config['training']['sequence_length']
    num_videos = video_data.shape[0] // sequence_length
    video_data = video_data[:num_videos * sequence_length].view(num_videos, sequence_length, video_data.shape[1], video_data.shape[2], video_data.shape[3])
    labels = labels[:num_videos * sequence_length].view(num_videos, sequence_length, 3).mean(dim=1)
    
    return video_data, labels

class VideoDataset(Dataset):
    def __init__(self, video_clips, labels, num_frames=16):
        self.video_clips = video_clips
        self.labels = labels
        self.num_frames = num_frames
    
    def __len__(self):
        return len(self.video_clips)
    
    def __getitem__(self, idx):
        return self.video_clips[idx][:self.num_frames], self.labels[idx]

def get_dataloaders(batch_size):
    """Splits dataset into train, validation, and test sets and returns DataLoaders."""
    video_data, labels = load_mithos_data()
    dataset = VideoDataset(video_data, labels)
    
    dataset_length = len(dataset)
    test_split = int(config['training']['train_split'] * dataset_length)
    train_val_dataset = Subset(dataset, range(0, test_split))
    test_dataset = Subset(dataset, range(test_split, dataset_length))
    
    train_size = int(config['training']['val_split'] * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    )
