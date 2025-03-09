import torch
import pickle
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import yaml

# Load configuration
with open("video_config_DEAP.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_deap_data():
    """Loads and preprocesses the DEAP dataset."""
    with open(config['data']['video_path1'], 'rb') as f:
        data1 = torch.from_numpy(pickle.load(f)).float()
    with open(config['data']['video_path2'], 'rb') as f:
        data2 = torch.from_numpy(pickle.load(f)).float()
    
    final_data = torch.cat((data1, data2), dim=0)
    final_data = final_data.permute(0, 3, 1, 2)  # Convert to (num_samples, channels, height, width)
    
    with open(config['data']['labels_p_path'], 'rb') as f:
        labels_p = torch.from_numpy(pickle.load(f)).float()
    with open(config['data']['labels_a_path'], 'rb') as f:
        labels_a = torch.from_numpy(pickle.load(f)).float()
    with open(config['data']['labels_d_path'], 'rb') as f:
        labels_d = torch.from_numpy(pickle.load(f)).float()
    
    labels = torch.cat((labels_p.unsqueeze(1), labels_a.unsqueeze(1), labels_d.unsqueeze(1)), dim=1)
    
    # Reshape the data to match ViViT input format
    sequence_length = config['training']['sequence_length']
    num_videos = final_data.shape[0] // sequence_length
    final_data = final_data[:num_videos * sequence_length]
    final_data = final_data.view(num_videos, sequence_length, final_data.shape[1], final_data.shape[2], final_data.shape[3])
    labels = labels[:num_videos * sequence_length].view(num_videos, sequence_length, 3).mean(dim=1)
    
    return final_data, labels

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
    final_data, labels = load_deap_data()
    dataset = VideoDataset(final_data, labels)
    
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
