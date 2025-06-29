import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


class LSTM_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_layers, output_size):
        super(LSTM_MLP, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = self.relu(self.fc1(lstm_out))
        x = self.fc2(x)
        return x


def load_data_from_csv(file_path):
    df = pd.read_csv(file_path, encoding='latin1')
    data = df.iloc[:, :18].astype(np.float32).values
    #labels = df.iloc[:, 18].astype(np.int64).values

    # xpwang map the string into the integer
    label_map = {'quasistable': 0, 'nonstationary': 1, 'instability': 2}
    labels_str = df.iloc[:, 18]
    labels = labels_str.map(label_map).astype(np.int64).values
    
    return data, labels

def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def create_sequences(data, labels, window_size, target_offset):
    sequences = []
    label_seqs = []
    for i in range(len(data) - window_size - target_offset + 1):
        sequences.append(data[i:i + window_size])
        label_seqs.append(labels[i + window_size + target_offset - 1])
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(label_seqs, dtype=torch.long)

# Hyperparameter settings(Please be the same as the training parameters)
input_size = 18
hidden_size = 64
lstm_layers = 4
output_size = 3   # (quasistable->0, nonstationary->1, instability->2, Please replace the labels in the .csv file)
window_size = 20
target_offset = 5

model = LSTM_MLP(input_size, hidden_size, lstm_layers, output_size).to(device)
model.load_state_dict(torch.load('model_weights_n_11.pth'))   # (The model has been trained, please use the file directly)
model.eval()


# folder_path = r'D:/'
folder_path = r'./Data/Data with PHOENIX'
file_accuracies = {}
all_true = []
all_preds = []
all_probs = []


for file_name in os.listdir(folder_path):
    if not file_name.endswith('.csv'):
        continue

    file_path = os.path.join(folder_path, file_name)
    print(f"\nProcessing file: {file_name}")

    data, labels = load_data_from_csv(file_path)
    data = normalize_data(data)
    sequences, label_seqs = create_sequences(data, labels, window_size, target_offset)
    if len(sequences) == 0:
        print(f"Skipping {file_name} due to insufficient length.")
        continue

    sequences = sequences.to(device)
    label_seqs = label_seqs.to(device)

    preds = []
    trues = []
    probs = []

    with torch.no_grad():
        for i in range(len(sequences)):
            seq = sequences[i:i+1]
            pred = model(seq)
            pred_class = torch.argmax(pred, dim=1)
            preds.append(pred_class.item())
            trues.append(label_seqs[i].item())
            probs.append(pred.cpu().numpy())

    acc = accuracy_score(trues, preds)
    print(f"Accuracy for {file_name}: {acc:.4f}")
    file_accuracies[file_name] = acc

    all_true.extend(trues)
    all_preds.extend(preds)
    all_probs.extend(probs)


accuracy_df = pd.DataFrame(list(file_accuracies.items()), columns=["File", "Accuracy"])
accuracy_df.to_csv('per_file_accuracy.csv', index=False)


cm = confusion_matrix(all_true, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)  

plt.figure(figsize=(6, 5))
sns.heatmap(cm_normalized * 100, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (%) - All Files')
plt.tight_layout()
plt.savefig('confusion_matrix_all_percent.png')
plt.show()



print("\n=== Classification Report (All Files) ===")
print(classification_report(all_true, all_preds, target_names=['Class 0', 'Class 1', 'Class 2']))


