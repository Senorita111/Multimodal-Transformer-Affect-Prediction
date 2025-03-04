import torch
import torchvision
import pickle
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
import os
import torch.optim as optim
from sklearn.model_selection import KFold
from transformers import VideoMAEForVideoClassification
import sys

output_log=open("ViViT_DM_PAD.txt",'w')
sys.stdout = output_log


# Setup
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()
torch.backends.cudnn.enabled = False
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


with open('Cropped_Face3_2_MITHOS_PAD.pkl', 'rb') as file:
    final_data = torch.from_numpy(pickle.load(file)).float()
    print("Loaded Data:", type(final_data),final_data.shape)

with open('Cleaned_Labels3_2_MITHOS_All_P.pkl', 'rb') as file:
    labels_p = torch.from_numpy(pickle.load(file)).float()
    print("Loaded Pleasure Labels:", type(labels_p),labels_p.shape)

with open('Cleaned_Labels3_2_MITHOS_All_A.pkl', 'rb') as file:
    labels_a = torch.from_numpy(pickle.load(file)).float()
    print("Loaded Arousal Labels:", type(labels_a),labels_a.shape)

with open('Cleaned_Labels3_2_MITHOS_All_D.pkl', 'rb') as file:
    labels_d = torch.from_numpy(pickle.load(file)).float()
    print("Loaded Dominance Labels:", type(labels_d),labels_d.shape)



final_data = final_data.permute(0, 3, 1, 2)  # Convert to (num_samples, channels, height, width)


labels = torch.cat((labels_p.unsqueeze(1), labels_a.unsqueeze(1), labels_d.unsqueeze(1)), dim=1)
print("Loaded and concatenated labels", labels.shape)  # The shape should now be (num_samples, 3)


# Reshape final_data to (num_videos, sequence_length, channels, height, width)
sequence_length = 16 # Adjust this based on your needs
num_videos = final_data.shape[0] // sequence_length

final_data = final_data[:num_videos * sequence_length]  # Trim extra frames if necessary
final_data = final_data.view(num_videos, sequence_length, final_data.shape[1], final_data.shape[2], final_data.shape[3])

print("Reshaped Input data", type(final_data), final_data.shape)


labels = labels[:num_videos * sequence_length]
labels = labels.view(num_videos, sequence_length, 3)
labels = labels.mean(dim=1)
print("Reshaped labels:", labels.shape) 

# Define custom VideoDataset
class VideoDataset(Dataset):
    def __init__(self, video_clips, labels, num_frames=16, transform=None):
        self.video_clips = video_clips
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.video_clips)

    def __getitem__(self, idx):
        frames = self.video_clips[idx]
        label = self.labels[idx]

        # Sample frames if necessary
        if frames.size(0) > self.num_frames:
            frames = frames[:self.num_frames]
        
        return frames, label


# Prepare dataset for K-Fold validation
dataset = VideoDataset(final_data, labels)
print("Dataset loaded")


# Parameters
num_epochs = 40
num_folds = 3
batch_size = 8  # Adjust batch size to fit GPU memory
kfold = KFold(n_splits=num_folds, shuffle=True)


for unfrozen_layers in range(1,7):
    print("===================================================================================")
    print("Computing for number of unfrozen layers:",unfrozen_layers)

    # K-Fold Cross-Validation
    results = 0
    results1 = 0
    results2 = 0
    resultsmae = 0
    resultspcc = 0
    resultsclass = 0

    results_p,results_a,results_d = 0,0,0
    results1_p,results1_a,results1_d = 0,0,0
    results2_p,results2_a,results2_d = 0,0,0
    resultsmae_p,resultsmae_a, resultsmae_d = 0,0,0
    resultspcc_p,resultspcc_a,resultspcc_d = 0,0,0
    resultsclass_p,resultsclass_a, resultsclass_d = 0,0,0



    train_fraction = 0.9
    validation_fraction = 0.2
    test_fraction = 0.1

    # Calculate the number of samples for train, validation, and test sets
    dataset_length = len(dataset)
    test_split = int(0.9 * dataset_length)

    # Split the data without shuffling
    train_val_indices = list(range(0, test_split))
    test_indices = list(range(test_split, dataset_length))

    train_val_dataset = Subset(dataset, train_val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size

    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])


    # Define the ViViT model
    class ViViTModel(nn.Module):
        def __init__(self):
            super(ViViTModel, self).__init__()
            self.vivit = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base")
            
            self.vivit.classifier = nn.Identity()  
            self.fc = nn.Linear(self.vivit.config.hidden_size, 3) # Adjusted in_features to match ViViT output

        def forward(self, x):
            outputs = self.vivit(pixel_values=x, output_hidden_states=False)  # Assuming the logits output is the final result
            hidden_state = outputs.logits 
            output = self.fc(hidden_state)
            return output


    for fold in range(num_folds):
        print(f'-----------------------FOLD {fold}-----------------------')

        best_val_loss = float('inf')

        # Create data loaders for train, validation, and test sets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        # Initialize your ViViT model
        pretrained_vivit = ViViTModel().to(device)

        pretrained_vivit.load_state_dict(torch.load('Final_ViViT_D_PAD.pth'))
        print("Model loaded from 'Final_ViViT_D_PAD.pth'.")

        # Fine-tuning layers
        for param in pretrained_vivit.parameters():
            param.requires_grad = False

        for param in pretrained_vivit.fc.parameters():
            param.requires_grad = True

        # Find the correct module for the model’s backbone
        base_model = pretrained_vivit.vivit.videomae if hasattr(pretrained_vivit.vivit, 'videomae') else pretrained_vivit.vivit

        # Unfreeze the last encoder layers (for example, unfreeze the last transformer block)
        for layer in base_model.encoder.layer[-unfrozen_layers:]:  # Unfreeze the last 2 encoder layers
            for param in layer.parameters():
                param.requires_grad = True

        optimizer = optim.Adam([
            {'params': base_model.encoder.layer[-unfrozen_layers:].parameters(), 'lr': 1e-4},  # Lower learning rate for pre-trained layers
            {'params': pretrained_vivit.fc.parameters(), 'lr': 1e-3}  # Higher learning rate for the new FC layer
        ])

        criterion = nn.MSELoss()

        # Training loop
        print(">>>>>>>>>>>>Training started")
        for epoch in range(num_epochs):
            pretrained_vivit.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_correct_p = 0
            train_correct_a = 0
            train_correct_d = 0
            train_total = 0

            for inputs, targets in train_loader:
                if inputs.size(0) == batch_size:
                    inputs, targets = inputs.to(device), targets.to(device).float()
                    
                    optimizer.zero_grad()
                    outputs = pretrained_vivit(inputs) # Forward pass
                    loss = criterion(outputs, targets)  # Compute loss
                    loss.backward()  # Backpropagate
                    optimizer.step()  # Update weights

                    train_loss += loss.item()
                    
                    predicted = torch.round(outputs)
                    train_correct_p += (predicted[:, 0] == targets[:, 0]).sum().item()
                    train_correct_a += (predicted[:, 1] == targets[:, 1]).sum().item()
                    train_correct_d += (predicted[:, 2] == targets[:, 2]).sum().item()
                    train_total += targets.size(0)

            # Calculate individual training accuracies
            train_accuracy_p = 100 * train_correct_p / train_total
            train_accuracy_a = 100 * train_correct_a / train_total
            train_accuracy_d = 100 * train_correct_d / train_total

            # Print training accuracies
            # print(f'Epoch {epoch+1}: Train Acc - Pleasure: {train_accuracy_p:.2f}%, Arousal: {train_accuracy_a:.2f}%, Dominance: {train_accuracy_d:.2f}%')

            
            # Validation phase
            pretrained_vivit.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_correct_class = 0
            val_loss = 0.0
            val_total = 0

            # Initialize counters for individual accuracies
            val_correct_p = 0
            val_correct_a = 0
            val_correct_d = 0

            all_outputs = []
            all_targets = []

            with torch.no_grad():
                for inputs, targets in val_loader:
                    if inputs.size(0) == batch_size:
                        inputs, targets = inputs.to(device), targets.to(device).float()
                        outputs = pretrained_vivit(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        
                        
                        predicted_classes = torch.where(outputs >= 5.0, 1, 0)
                        target_classes = torch.where(targets >= 5.0, 1, 0)
                        val_correct_class += (predicted_classes == target_classes).sum().item()

                        all_outputs.append(outputs.cpu())
                        all_targets.append(targets.cpu())

                        val_correct_p += (predicted[:, 0] == targets[:, 0]).sum().item()
                        val_correct_a += (predicted[:, 1] == targets[:, 1]).sum().item()
                        val_correct_d += (predicted[:, 2] == targets[:, 2]).sum().item()
                        val_total += targets.size(0)

            '''
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total
            class_accuracy = 100 * val_correct_class / val_total'''
            
            all_outputs = torch.cat(all_outputs, dim=0)  # Shape: (num_samples, 3)
            all_targets = torch.cat(all_targets, dim=0)  # Shape: (num_samples, 3)
            val_loss /= len(val_loader)

            accuracy_p = 100 * val_correct_p / val_total
            accuracy_a = 100 * val_correct_a / val_total
            accuracy_d = 100 * val_correct_d / val_total

            # Calculate MAE for each dimension
            mae_p = torch.mean(torch.abs(all_outputs[:, 0] - all_targets[:, 0]))
            mae_a = torch.mean(torch.abs(all_outputs[:, 1] - all_targets[:, 1]))
            mae_d = torch.mean(torch.abs(all_outputs[:, 2] - all_targets[:, 2]))

            # Calculate PCC for each dimension
            pcc_p = torch.corrcoef(torch.stack([all_outputs[:, 0], all_targets[:, 0]]))[0, 1]
            pcc_a = torch.corrcoef(torch.stack([all_outputs[:, 1], all_targets[:, 1]]))[0, 1]
            pcc_d = torch.corrcoef(torch.stack([all_outputs[:, 2], all_targets[:, 2]]))[0, 1]

            # Print validation results
            print(f'Epoch {epoch+1}: Val Loss: {val_loss:.4f}')
            # print(f'Epoch {epoch+1}: Val Loss: {val_loss:.4f}, Accuracy - Pleasure: {accuracy_p:.2f}%, Arousal: {accuracy_a:.2f}%, Dominance: {accuracy_d:.2f}%')
            # print(f'Validation MAE - Pleasure: {mae_p:.4f}, Arousal: {mae_a:.4f}, Dominance: {mae_d:.4f}')
            # print(f'Validation PCC - Pleasure: {pcc_p:.4f}, Arousal: {pcc_a:.4f}, Dominance: {pcc_d:.4f}')

            # print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%, PCC: {pcc:.4f}, MAE: {mae:.4f}, Accuracy (Class-based): {class_accuracy:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trials = 0
            else:
                trials += 1
                if trials >= 5:
                    print(f'Epoch {epoch+1}: Train Acc - Pleasure: {train_accuracy_p:.2f}%, Arousal: {train_accuracy_a:.2f}%, Dominance: {train_accuracy_d:.2f}%')

                    # print("Early stopping triggered")
                    break

        print(">>>>>>>>>>>>Testing")

        # torch.save(pretrained_vivit.state_dict(), 'Video_ViViT_Updated.pth')

        correct_predictions, correct_predictions_range2, correct_predictions_range1, test_total = 0, 0, 0, 0
        all_outputs = []
        all_targets = []
        test_correct_class = 0
        test_correct_p = 0
        test_correct_a = 0
        test_correct_d = 0
        test_total = 0
        test_correct_p_range1=0
        test_correct_a_range1=0
        test_correct_d_range1=0
        test_correct_p_range2=0
        test_correct_a_range2=0
        test_correct_d_range2=0
        test_correct_p_class=0
        test_correct_a_class=0
        test_correct_d_class=0

        with torch.no_grad():
            for inputs, targets in test_loader:
                if inputs.size(0) == batch_size:
                    inputs, targets = inputs.to(device), targets.to(device).float()
                    outputs = pretrained_vivit(inputs) #.squeeze(1)
                    predicted = torch.round(outputs)


                    # correct_predictions += (predicted == targets).sum().item()
                    # correct_predictions_range2 += ((predicted >= (targets - 2)) & (predicted <= (targets + 2))).sum().item()
                    # correct_predictions_range1 += ((predicted >= (targets - 1)) & (predicted <= (targets + 1))).sum().item()
                    test_total += targets.size(0)

                    predicted_classes = torch.where(outputs >= 5.0, 1, 0)
                    target_classes = torch.where(targets >= 5.0, 1, 0)
                    test_correct_class += (predicted_classes == target_classes).sum().item()

                    test_correct_p += (predicted[:, 0] == targets[:, 0]).sum().item()
                    test_correct_a += (predicted[:, 1] == targets[:, 1]).sum().item()
                    test_correct_d += (predicted[:, 2] == targets[:, 2]).sum().item()

                    test_correct_p_range1 += ((predicted[:, 0] >= (targets[:, 0] - 1)) & (predicted[:, 0] <= (targets[:, 0] + 1))).sum().item()
                    test_correct_a_range1 += ((predicted[:, 1] >= (targets[:, 1] - 1)) & (predicted[:, 1] <= (targets[:, 1] + 1))).sum().item()
                    test_correct_d_range1 += ((predicted[:, 2] >= (targets[:, 2] - 1)) & (predicted[:, 2] <= (targets[:, 2] + 1))).sum().item()
                    
                    test_correct_p_range2 += ((predicted[:, 0] >= (targets[:, 0] - 2)) & (predicted[:, 0] <= (targets[:, 0] + 2))).sum().item()
                    test_correct_a_range2 += ((predicted[:, 1] >= (targets[:, 1] - 2)) & (predicted[:, 1] <= (targets[:, 1] + 2))).sum().item()
                    test_correct_d_range2 += ((predicted[:, 2] >= (targets[:, 2] - 2)) & (predicted[:, 2] <= (targets[:, 2] + 2))).sum().item()
                    

                    all_outputs.append(outputs.cpu())
                    all_targets.append(targets.cpu())

                    # Classify predicted and target values into two classes (0-4 and 5-9)
                    predicted_class_p = torch.where(predicted[:, 0] <= 4, 0, 1)  # Class 0: 0-4, Class 1: 5-9 for Pleasure
                    predicted_class_a = torch.where(predicted[:, 1] <= 4, 0, 1)  # Class 0: 0-4, Class 1: 5-9 for Arousal
                    predicted_class_d = torch.where(predicted[:, 2] <= 4, 0, 1)  # Class 0: 0-4, Class 1: 5-9 for Dominance

                    target_class_p = torch.where(targets[:, 0] <= 4, 0, 1)
                    target_class_a = torch.where(targets[:, 1] <= 4, 0, 1)
                    target_class_d = torch.where(targets[:, 2] <= 4, 0, 1)

                    # Calculate class-based accuracies (correct if both predicted and target are in the same class)
                    test_correct_p_class += (predicted_class_p == target_class_p).sum().item()
                    test_correct_a_class += (predicted_class_a == target_class_a).sum().item()
                    test_correct_d_class += (predicted_class_d == target_class_d).sum().item()
                    test_total += targets.size(0)

        
        all_outputs = torch.cat(all_outputs, dim=0)  # Shape: (num_samples, 3)
        all_targets = torch.cat(all_targets, dim=0) 

        # Calculate individual accuracies
        test_accuracy_p = 100 * test_correct_p / test_total
        test_accuracy_a = 100 * test_correct_a / test_total
        test_accuracy_d = 100 * test_correct_d / test_total

        test_accuracy_p_range1 = 100 * test_correct_p_range1 / test_total
        test_accuracy_a_range1 = 100 * test_correct_a_range1 / test_total
        test_accuracy_d_range1 = 100 * test_correct_d_range1 / test_total

        test_accuracy_p_range2 = 100 * test_correct_p_range2 / test_total
        results2_p+=test_accuracy_p_range2
        test_accuracy_a_range2 = 100 * test_correct_a_range2 / test_total
        results2_a+=test_accuracy_a_range2
        test_accuracy_d_range2 = 100 * test_correct_d_range2 / test_total
        results2_d+=test_accuracy_d_range2

        test_accuracy_p_class = 100 * test_correct_p_class / test_total
        resultsclass_p+=test_accuracy_p_class
        test_accuracy_a_class = 100 * test_correct_a_class / test_total
        resultsclass_a+=test_accuracy_a_class
        test_accuracy_d_class = 100 * test_correct_d_class / test_total
        resultsclass_d+=test_accuracy_d_class

        # Calculate MAE for each dimension
        mae_p = torch.mean(torch.abs(all_outputs[:, 0] - all_targets[:, 0]))
        resultsmae_p+=mae_p
        mae_a = torch.mean(torch.abs(all_outputs[:, 1] - all_targets[:, 1]))
        resultsmae_a+=mae_a
        mae_d = torch.mean(torch.abs(all_outputs[:, 2] - all_targets[:, 2]))
        resultsmae_d+=mae_d

        # Calculate PCC for each dimension
        pcc_p = torch.corrcoef(torch.stack([all_outputs[:, 0], all_targets[:, 0]]))[0, 1]
        resultspcc_p+=pcc_p
        pcc_a = torch.corrcoef(torch.stack([all_outputs[:, 1], all_targets[:, 1]]))[0, 1]
        resultspcc_a+=pcc_a
        pcc_d = torch.corrcoef(torch.stack([all_outputs[:, 2], all_targets[:, 2]]))[0, 1]
        resultspcc_d+=pcc_d

        test_accuracy = (test_accuracy_p+test_accuracy_a+test_accuracy_d)/3
        results+=test_accuracy
        test_accuracy_range1 = (test_accuracy_p_range1+test_accuracy_a_range1+test_accuracy_d_range1)/3
        test_accuracy_range2 = (test_accuracy_p_range2+test_accuracy_a_range2+test_accuracy_d_range2)/3
        results2+=test_accuracy_range2
        test_accuracy_class = (test_accuracy_p_class+test_accuracy_a_class+test_accuracy_d_class)/3
        resultsclass+=test_accuracy_class
        test_mae = (mae_p+mae_a+mae_d)/3
        resultsmae+=test_mae
        test_pcc = (pcc_p+pcc_a+pcc_d)/3
        resultspcc+=test_pcc

        # Print test results
        print(f'Test Accuracy - (P): {test_accuracy_p:.2f}%, (A): {test_accuracy_a:.2f}%, (D): {test_accuracy_d:.2f}%, (Total): {test_accuracy:.2f}%')
        print(f'Test Accuracy Range 1- (P): {test_accuracy_p_range1:.2f}%, (A): {test_accuracy_a_range1:.2f}%, (D): {test_accuracy_d_range1:.2f}%, (Total): {test_accuracy_range1:.2f}%')
        print(f'Test Accuracy Range 2- (P): {test_accuracy_p_range2:.2f}%, (A): {test_accuracy_a_range2:.2f}%, (D): {test_accuracy_d_range2:.2f}%, (Total): {test_accuracy_range2:.2f}%')
        print(f'Validation Class-based Accuracy - (P): {test_accuracy_p_class:.2f}%, (A): {test_accuracy_a_class:.2f}%, (D): {test_accuracy_d_class:.2f}%, (Total): {test_accuracy_class:.2f}%')

        print(f'Test MAE - (P): {mae_p:.4f}, (A): {mae_a:.4f}, (D): {mae_d:.4f}, (Total): {test_mae:.2f}')
        print(f'Test PCC - (P): {pcc_p:.4f}, (A): {pcc_a:.4f}, (D): {pcc_d:.4f}, (Total): {test_pcc:.2f}')
        
        

    # Print fold results
    print(f'\n\nK-FOLD CROSS VALIDATION RESULTS FOR {num_folds} FOLDS')

    print('--------------------------------')
    print(f'Average Accuracy Range 2:  (P):{results2_p/num_folds} %, (A): {results2_a/num_folds}, (D): {results2_d/num_folds}, (Total): {results2/num_folds}')
    print('--------------------------------')
    print(f'Average Accuracy Class Based:  (P):{resultsclass_p/num_folds} %, (A): {resultsclass_a/num_folds}, (D): {resultsclass_d/num_folds}, (Total): {resultsclass/num_folds}')
    print('--------------------------------')
    print(f'Average MAE: (P):{resultsmae_p/num_folds} %, (A): {resultsmae_a/num_folds}, (D): {resultsmae_d/num_folds}, (Total): {resultsmae/num_folds}')
    print('--------------------------------')
    print(f'Average Accuracy PCC (P):{resultspcc_p/num_folds} %, (A): {resultspcc_a/num_folds}, (D): {resultspcc_d/num_folds}, (Total): {resultspcc/num_folds}')


output_log.close()