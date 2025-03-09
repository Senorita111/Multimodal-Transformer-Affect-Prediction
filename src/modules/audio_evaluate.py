import torch
import numpy as np
from fusion_dataloader import load_data, get_dataloaders
from fusion_model import AudioVisualTransformer
from sklearn.metrics import mean_absolute_error
import yaml

# Load config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def compute_metrics(outputs, labels):
    outputs, labels = np.array(outputs), np.array(labels)
    mae = mean_absolute_error(labels, outputs)
    pcc = np.corrcoef(labels, outputs)[0, 1] if len(outputs) > 1 else 0
    class_outputs = np.where(outputs >= 5, 1, 0)
    class_labels = np.where(labels >= 5, 1, 0)
    class_accuracy = (class_outputs == class_labels).sum() / len(labels) * 100
    range_2_accuracy = ((outputs >= (labels - 2)) & (outputs <= (labels + 2))).sum() / len(labels) * 100
    return mae, pcc, class_accuracy, range_2_accuracy

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_loss = 0
    criterion = torch.nn.MSELoss()
    test_metrics = {"pleasure": [], "arousal": [], "dominance": []}
    
    with torch.no_grad():
        for audio_batch, video_batch, labels_batch in test_loader:
            audio_batch, video_batch, labels_batch = audio_batch.to(device), video_batch.to(device), labels_batch.to(device)
            pleasure_output, arousal_output, dominance_output = model(audio_batch, video_batch)
            loss = criterion(pleasure_output, labels_batch[:, 0].unsqueeze(1)) + \
                   criterion(arousal_output, labels_batch[:, 1].unsqueeze(1)) + \
                   criterion(dominance_output, labels_batch[:, 2].unsqueeze(1))
            total_loss += loss.item()
            test_metrics["pleasure"].append((pleasure_output.item(), labels_batch[:, 0].item()))
            test_metrics["arousal"].append((arousal_output.item(), labels_batch[:, 1].item()))
            test_metrics["dominance"].append((dominance_output.item(), labels_batch[:, 2].item()))
    
    final_metrics = {}
    for key, values in test_metrics.items():
        outputs, labels = zip(*values)
        mae, pcc, class_acc, range_2_acc = compute_metrics(outputs, labels)
        final_metrics[f"{key}_mae"] = mae
        final_metrics[f"{key}_pcc"] = pcc
        final_metrics[f"{key}_class_acc"] = class_acc
        final_metrics[f"{key}_range_2_acc"] = range_2_acc
    
    print(f"Test Metrics: {final_metrics}")
    print(f'Test Loss: {total_loss/len(test_loader):.4f}')
    return final_metrics

def main():
    dataset = load_data(
                        config["data"]["video_path"], 
                        config["data"]["audio_path"],
                        config["data"]["labels_p_path"], 
                        config["data"]["labels_a_path"],
                        config["data"]["labels_d_path"]
                        )
    _, _, test_loader = get_dataloaders(dataset)
    model = AudioVisualTransformer(num_classes=1)
    model.load_state_dict(torch.load(config["evaluation"]["model_load_path"]))
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()