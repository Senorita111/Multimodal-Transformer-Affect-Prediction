import torch
import torch.nn as nn
from torch.optim import Adam
from data_loader_mithos import get_dataloaders
from model_mithos import ViViTModelMITHOS
import yaml

# Load configuration
with open("video_config_DEAP_MITHOS.yaml", "r") as f:
    config = yaml.safe_load(f)

def train_model(model, train_loader, val_loader, device):
    """Fine-tunes the pretrained DEAP model on MITHOS dataset."""

    # Fine-tuning layers
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    # Find the correct module for the model’s backbone
    base_model = model.vivit.videomae if hasattr(model.vivit, 'videomae') else model.vivit

    # Unfreeze the last encoder layers (for example, unfreeze the last transformer block)
    for layer in base_model.encoder.layer[-1:]:  # Unfreeze the last 2 encoder layers
        for param in layer.parameters():
            param.requires_grad = True

    optimizer = Adam([
            {'params': base_model.encoder.layer[-1:].parameters(), 'lr': 1e-4},  # Lower learning rate for pre-trained layers
            {'params': model.fc.parameters(), 'lr': 1e-3}  # Higher learning rate for the new FC layer
    ])

    # optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
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
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
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
    model = ViViTModelMITHOS(config['training']['pretrained_deap_model_path'])
    train_model(model, train_loader, val_loader, device)

if __name__ == "__main__":
    main()
