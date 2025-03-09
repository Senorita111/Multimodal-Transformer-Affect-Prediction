import torch
import torch.nn as nn
from torch.optim import Adam
from audio_data_loader import get_dataloaders
from audio_model import EmotionModel
import yaml

# Load configuration
with open("audio_config.yaml", "r") as f:
    config = yaml.safe_load(f)

def train_model(model, train_loader, val_loader, device):
    """Trains the audio model."""
    optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    
    model.to(device)
    for epoch in range(config['training']['num_epochs']):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            optimizer.zero_grad()
            arousal_output, valence_output, dominance_output = model(inputs)
            loss = (criterion(arousal_output, targets[:, 0]) +
                    criterion(valence_output, targets[:, 1]) +
                    criterion(dominance_output, targets[:, 2]))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                arousal_output, valence_output, dominance_output = model(inputs)
                val_loss += (criterion(arousal_output, targets[:, 0]) +
                             criterion(valence_output, targets[:, 1]) +
                             criterion(dominance_output, targets[:, 2])).item()
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config['training']['model_save_path'])
        else:
            patience_counter += 1
        
        if patience_counter >= config['training']['patience']:
            print("Early stopping triggered")
            break

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, _ = get_dataloaders(config['training']['batch_size'])
    model = EmotionModel.from_pretrained(config['model']['wav2vec_model'])
    train_model(model, train_loader, val_loader, device)

if __name__ == "__main__":
    main()
