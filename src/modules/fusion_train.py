import torch
import torch.nn as nn
from torch.optim import Adam
from fusion_dataloader import load_data, get_dataloaders
from fusion_model import AudioVisualTransformer
import numpy as np
from sklearn.metrics import mean_absolute_error
import yaml

# Load config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

    

def compute_metrics(outputs, labels):
    outputs, labels = np.array(outputs), np.array(labels)
    mae = mean_absolute_error(labels, outputs)
    pcc = np.corrcoef(labels, outputs)[0, 1] if len(outputs) > 1 else 0
    return mae, pcc

def train_model(model, train_loader, val_loader, epochs=config["training"]["num_epochs"], patience=config["training"]["patience"]):
    device, best_val_loss, patience_counter = torch.device("cuda" if torch.cuda.is_available() else "cpu"), float('inf'), 0
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.MSELoss()
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for audio_batch, video_batch, labels_batch in train_loader:
            audio_batch, video_batch, labels_batch = audio_batch.to(device), video_batch.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            pleasure_output, arousal_output, dominance_output = model(audio_batch, video_batch)
            loss = criterion(pleasure_output, labels_batch[:, 0].unsqueeze(1)) + \
                   criterion(arousal_output, labels_batch[:, 1].unsqueeze(1)) + \
                   criterion(dominance_output, labels_batch[:, 2].unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for audio_batch, video_batch, labels_batch in val_loader:
                audio_batch, video_batch, labels_batch = audio_batch.to(device), video_batch.to(device), labels_batch.to(device)
                pleasure_output, arousal_output, dominance_output = model(audio_batch, video_batch)
                loss = criterion(pleasure_output, labels_batch[:, 0].unsqueeze(1)) + \
                       criterion(arousal_output, labels_batch[:, 1].unsqueeze(1)) + \
                       criterion(dominance_output, labels_batch[:, 2].unsqueeze(1))
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            torch.save(model.state_dict(), config["training"]["model_save_path"])
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

def main():
    # dataset = load_data("data/processed/Fusion_Video_MITHOS.pkl", "data/processed/Fusion_Audio_MITHOS.pkl",
    #                     "data/processed/Fusion_Labels_P_MITHOS.pkl", "data/processed/Fusion_Labels_A_MITHOS.pkl",
    #                     "data/processed/Fusion_Labels_D_MITHOS.pkl")

    dataset = load_data(
                        config["data"]["video_path"], 
                        config["data"]["audio_path"],
                        config["data"]["labels_p_path"], 
                        config["data"]["labels_a_path"],
                        config["data"]["labels_d_path"]
                        )


    train_loader, val_loader, test_loader = get_dataloaders(dataset)
    model = AudioVisualTransformer(num_classes=1)
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main()