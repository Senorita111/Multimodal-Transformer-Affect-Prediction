


import numpy as np
import torch
import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import DataLoader, Dataset,random_split, Subset
import librosa
import soundfile as sf
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import random
import pandas as pd
from scipy.stats import pearsonr
import torch.nn.functional as F
import os 


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'



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
        audio_tensor = torch.tensor(audio_tensor, dtype=torch.float) 
        label_tensor = torch.tensor(label, dtype=torch.float) 
        return audio_tensor, label_tensor

class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x.squeeze()  # Output a single value per sample

class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.arousal_head = RegressionHead(config)
        self.valence_head = RegressionHead(config)
        self.dominance_head = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state.mean(dim=1)
        arousal_output = self.arousal_head(hidden_states)
        valence_output = self.valence_head(hidden_states)
        dominance_output = self.dominance_head(hidden_states)
        return arousal_output, valence_output, dominance_output
# Function to calculate MAE
def calculate_mae(predictions, targets):
    return F.l1_loss(predictions, targets).item()

# Function to calculate PCC
def calculate_pcc(predictions, targets):
    # Ensure predictions and targets are detached from the computation graph and converted to CPU
    predictions = predictions.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()
    # print("!!!!!!!!!!!!!!!!!!!",predictions,targets)
    # Handle the case where there is only one data point (e.g., batch size of 1)
    if len(predictions) == 1:
        return 0.0  # Pearson correlation is not defined for a single data point
    
    return pearsonr(predictions, targets)[0]  

def calculate_accuracy_regression(predictions, targets, tolerance=0.7):
    with torch.no_grad():
        correct = torch.abs(predictions - targets) < tolerance
        return correct.float().mean()
def calculate_accuracy_reg_range1(predictions, targets, tolerance=1.5):
    with torch.no_grad():
        correct = torch.abs(predictions - targets) < tolerance
        return correct.float().mean()
def calculate_accuracy_reg_range2(predictions, targets, tolerance=2.5):
    with torch.no_grad():
        correct = torch.abs(predictions - targets) < tolerance
        return correct.float().mean()


def calculate_class_accuracy(predictions, targets):
    with torch.no_grad():
        # Classify predictions and targets into classes 0-4 and 5-9
        pred_class = (predictions >= 5).int()
        target_class = (targets >= 5).int()
        correct = (pred_class == target_class).float()
        return correct.mean()



def collate_fn(batch):
    # Find the max length in this batch
    max_len = max([len(x[0]) for x in batch])
    
    # Pad sequences and stack them
    inputs = torch.stack([torch.cat([x[0], torch.zeros(max_len - len(x[0]))]) for x in batch])
    targets = torch.stack([x[1] for x in batch])
    
    return inputs, targets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name)
model = model.to(device)



file_paths=[]
labels=[]

file_path='Timestamps_All'
all_videos=[]
labels=[]
df = pd.read_csv(file_path, sep='\t')
print("------------------")
# Display the first few rows of the dataframe
for i in range(len(df)):
    participant= df.iloc[i]['Participant_Number'][-2:]
    if participant in ['02','04','06','08','10','11','12','13','14','15','16','18','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40']:
        ts=df.iloc[i]['Timestamp_No']
        audio_file_name='CroppedAudios/Participant'+participant+'_TS_'+str(ts)+'.wav'
        file_paths.append(audio_file_name)

        labels.append([df.iloc[i]['ArousalAverage'],df.iloc[i]['PleasureAverage'],df.iloc[i]['DominanceAverage']])

print("-------------Data Loaded-------------")
dataset = EmotionDataset(file_paths, labels, processor)
'''
# Split dataset
total_count = len(dataset)
train_count = int(0.7 * total_count)
val_count = int(0.2 * total_count)
batch_size = 1
test_count = total_count - train_count - val_count
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_count, val_count, test_count])



# Update your DataLoader to use the collate function
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
'''



# Determine split sizes
total_count = len(dataset)
train_val_count = int(0.9 * total_count)  # First 90% for training and validation
test_count = total_count - train_val_count  # Last 10% for testing

# Create subsets for train+val and test
train_val_dataset = Subset(dataset, list(range(train_val_count)))  # First 90% (indices 0 to train_val_count-1)
test_dataset = Subset(dataset, list(range(train_val_count, total_count)))  # Last 10% (indices train_val_count to end)

# Now split the train_val_dataset into train and validation sets
train_count = int(0.7 * len(train_val_dataset))  # 70% of the 90% for training
val_count = len(train_val_dataset) - train_count  # 30% of the 90% for validation

# Random split for training and validation datasets (shuffled)
train_dataset, val_dataset = random_split(train_val_dataset, [train_count, val_count])

# DataLoader for train and validation (shuffled) and test (not shuffled)
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print("-------------Loaders set-------------")

optimizer = Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss().to(device)

# Early stopping parameters
patience = 10
best_val_loss = float('inf')
patience_counter = 0
print("-------------Training Started-------------")
num_epochs = 100

arousal_preds = []
arousal_targets_list = []
valence_preds = []
valence_targets_list = []
dominance_preds = []
dominance_targets_list = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_mae = 0
    total_pcc = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        arousal_targets = targets[:, 0].float() 
        valence_targets = targets[:, 1].float() 
        dominance_targets = targets[:, 2].float() 
        
        optimizer.zero_grad()
        arousal_output, valence_output, dominance_output = model(inputs)

        loss_a = criterion(arousal_output, arousal_targets)
        loss_v = criterion(valence_output, valence_targets)
        loss_d = criterion(dominance_output, dominance_targets)
        loss = loss_a + loss_v + loss_d
        loss.backward()
        optimizer.step()


        total_loss += loss.item()
        total_mae += (calculate_mae(arousal_output, arousal_targets) +
                      calculate_mae(valence_output, valence_targets) +
                      calculate_mae(dominance_output, dominance_targets)) / 3
        # total_pcc += (calculate_pcc(arousal_output, arousal_targets) +
        #               calculate_pcc(valence_output, valence_targets) +
        #               calculate_pcc(dominance_output, dominance_targets)) / 3
        
        arousal_preds.extend(arousal_output.detach().cpu().numpy().flatten())
        arousal_targets_list.extend(arousal_targets.detach().cpu().numpy().flatten())
        valence_preds.extend(valence_output.detach().cpu().numpy().flatten())
        valence_targets_list.extend(valence_targets.detach().cpu().numpy().flatten())
        dominance_preds.extend(dominance_output.detach().cpu().numpy().flatten())
        dominance_targets_list.extend(dominance_targets.detach().cpu().numpy().flatten())
        


    train_loss = total_loss / len(train_loader)
    train_mae = total_mae / len(train_loader)
    
    # Calculate PCC using all accumulated predictions and targets
    train_pcc_arousal = calculate_pcc(torch.tensor(arousal_preds), torch.tensor(arousal_targets_list))
    train_pcc_valence = calculate_pcc(torch.tensor(valence_preds), torch.tensor(valence_targets_list))
    train_pcc_dominance = calculate_pcc(torch.tensor(dominance_preds), torch.tensor(dominance_targets_list))
    train_pcc = (train_pcc_arousal + train_pcc_valence + train_pcc_dominance) / 3


    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_mae = 0
        val_pcc = 0
        # Lists to accumulate predictions and targets for PCC calculation
        arousal_preds = []
        arousal_targets_list = []
        valence_preds = []
        valence_targets_list = []
        dominance_preds = []
        dominance_targets_list = []

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            arousal_targets = targets[:, 0]
            valence_targets = targets[:, 1]
            dominance_targets = targets[:, 2]
            
            arousal_output, valence_output, dominance_output = model(inputs)
            
            loss_a = criterion(arousal_output, arousal_targets)
            loss_v = criterion(valence_output, valence_targets)
            loss_d = criterion(dominance_output, dominance_targets)
            loss = loss_a + loss_v + loss_d
            val_loss += loss.item()
            
            val_mae += (calculate_mae(arousal_output, arousal_targets) +
                        calculate_mae(valence_output, valence_targets) +
                        calculate_mae(dominance_output, dominance_targets)) / 3
            # val_pcc += (calculate_pcc(arousal_output, arousal_targets) +
            #             calculate_pcc(valence_output, valence_targets) +
            #             calculate_pcc(dominance_output, dominance_targets)) / 3
        
            # Accumulate predictions and targets for PCC calculation
            arousal_preds.extend(arousal_output.detach().cpu().numpy().flatten())
            arousal_targets_list.extend(arousal_targets.detach().cpu().numpy().flatten())
            valence_preds.extend(valence_output.detach().cpu().numpy().flatten())
            valence_targets_list.extend(valence_targets.detach().cpu().numpy().flatten())
            dominance_preds.extend(dominance_output.detach().cpu().numpy().flatten())
            dominance_targets_list.extend(dominance_targets.detach().cpu().numpy().flatten())
        
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        # val_pcc /= len(val_loader)
        # Calculate PCC using all accumulated predictions and targets
        val_pcc_arousal = calculate_pcc(torch.tensor(arousal_preds), torch.tensor(arousal_targets_list))
        val_pcc_valence = calculate_pcc(torch.tensor(valence_preds), torch.tensor(valence_targets_list))
        val_pcc_dominance = calculate_pcc(torch.tensor(dominance_preds), torch.tensor(dominance_targets_list))
        val_pcc = (val_pcc_arousal + val_pcc_valence + val_pcc_dominance) / 3
        
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train PCC: {train_pcc:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val PCC: {val_pcc_valence:.4f} {val_pcc_arousal:.4f}{val_pcc_dominance:.4f}')
    #print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train PCC: {train_pcc:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val PCC: {val_pcc:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    
    # print("Saving the model")
    # torch.save(model, 'emotion_classification_model.pt')
print("-------------Training Done-------------")



test_mae = 0
mae_p=0
mae_a=0
mae_d=0
test_pcc = 0
test_accuracy = 0
test_accuracy_range1=0
test_accuracy_range2=0
test_acc_range2_p=0
test_acc_range2_a=0
test_acc_range2_d=0
arousal_preds = []
arousal_targets_list = []
valence_preds = []
valence_targets_list = []
dominance_preds = []
dominance_targets_list = []
test_class_accuracy_p = 0
test_class_accuracy_a = 0
test_class_accuracy_d = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        arousal_targets = targets[:, 0]
        valence_targets = targets[:, 1]
        dominance_targets = targets[:, 2]
        
        arousal_output, valence_output, dominance_output = model(inputs)

        mae_p+=calculate_mae(valence_output, valence_targets) 
        mae_a+=calculate_mae(arousal_output, arousal_targets) 
        mae_d+=calculate_mae(dominance_output, dominance_targets)
        
        test_mae += (calculate_mae(arousal_output, arousal_targets) +
                     calculate_mae(valence_output, valence_targets) +
                     calculate_mae(dominance_output, dominance_targets)) / 3
        
        # test_pcc += (calculate_pcc(arousal_output, arousal_targets) +
        #              calculate_pcc(valence_output, valence_targets) +
        #              calculate_pcc(dominance_output, dominance_targets)) / 3
        # Accumulate predictions and targets for PCC calculation
        arousal_preds.extend(arousal_output.detach().cpu().numpy().flatten())
        arousal_targets_list.extend(arousal_targets.detach().cpu().numpy().flatten())
        valence_preds.extend(valence_output.detach().cpu().numpy().flatten())
        valence_targets_list.extend(valence_targets.detach().cpu().numpy().flatten())
        dominance_preds.extend(dominance_output.detach().cpu().numpy().flatten())
        dominance_targets_list.extend(dominance_targets.detach().cpu().numpy().flatten())


        
        test_accuracy += (calculate_accuracy_regression(arousal_output, arousal_targets) +
                          calculate_accuracy_regression(valence_output, valence_targets) +
                          calculate_accuracy_regression(dominance_output, dominance_targets)) / 3
        
        test_accuracy_range1 += (calculate_accuracy_reg_range1(arousal_output, arousal_targets) +
                          calculate_accuracy_reg_range1(valence_output, valence_targets) +
                          calculate_accuracy_reg_range1(dominance_output, dominance_targets)) / 3
        
        test_acc_range2_p += calculate_accuracy_reg_range2(valence_output, valence_targets) 
        test_acc_range2_a += calculate_accuracy_reg_range2(arousal_output, arousal_targets)
        test_acc_range2_d += calculate_accuracy_reg_range2(dominance_output, dominance_targets)

        test_accuracy_range2 += (calculate_accuracy_reg_range2(arousal_output, arousal_targets) +
                          calculate_accuracy_reg_range2(valence_output, valence_targets) +
                          calculate_accuracy_reg_range2(dominance_output, dominance_targets)) / 3

        test_class_accuracy_p += calculate_class_accuracy(valence_output, valence_targets)
        test_class_accuracy_a += calculate_class_accuracy(arousal_output, arousal_targets)
        test_class_accuracy_d += calculate_class_accuracy(dominance_output, dominance_targets)



    mae_p /= len(test_loader)
    mae_a /= len(test_loader)
    mae_d /= len(test_loader)

    test_acc_range2_p/= len(test_loader)
    test_acc_range2_a/= len(test_loader)
    test_acc_range2_d/= len(test_loader)

    test_class_accuracy_p /= len(test_loader)
    test_class_accuracy_a /= len(test_loader)
    test_class_accuracy_d /= len(test_loader)


    test_mae /= len(test_loader)
    # test_pcc /= len(test_loader)
    test_accuracy /= len(test_loader)
    test_accuracy_range1/= len(test_loader)
    test_accuracy_range2/= len(test_loader)
    # Calculate PCC using all accumulated predictions and targets
    test_pcc_arousal = calculate_pcc(torch.tensor(arousal_preds), torch.tensor(arousal_targets_list))
    test_pcc_valence = calculate_pcc(torch.tensor(valence_preds), torch.tensor(valence_targets_list))
    test_pcc_dominance = calculate_pcc(torch.tensor(dominance_preds), torch.tensor(dominance_targets_list))
    test_pcc = (val_pcc_arousal + val_pcc_valence + val_pcc_dominance) / 3
    print("----MAE------",mae_p,"---",mae_a,"---",mae_d)
    print("----PCC------",test_pcc_valence,"---",test_pcc_arousal,"---",test_pcc_dominance)
    print("----Test_acc_range------",test_acc_range2_p,"---",test_acc_range2_a,"---",test_acc_range2_d)
    print("----Test_acc_class------",test_class_accuracy_p,"---",test_class_accuracy_a,"---",test_class_accuracy_d)
    

print(f'Test MAE: {test_mae:.4f}, Test PCC: {test_pcc:.4f}, Test Accuracy: {test_accuracy*100:.4f},{test_accuracy_range1*100:.4f},{test_accuracy_range2*100:.4f}, Test PCC: {test_pcc:.4f} {test_pcc_valence:.4f} {test_pcc_arousal:.4f}{test_pcc_dominance:.4f}')








