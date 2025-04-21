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



def load_data_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='latin1')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None, None
    data = df.iloc[:, :18].astype(np.float32).values  
    labels = df.iloc[:, 18].astype(np.int64).values  
    return data, labels



def normalize_data(data, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)
    else:
        data_normalized = scaler.transform(data)
    return data_normalized, scaler



def create_sequences(data, labels, window_size, target_offset):
    sequences = []
    label_seqs = []
    for i in range(len(data) - window_size - target_offset + 1):
        seq = data[i:i + window_size]
        label_seq = labels[i + window_size + target_offset - 1]
        sequences.append(seq)
        label_seqs.append(label_seq)
    return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(label_seqs), dtype=torch.long)



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



input_size = 18  
hidden_size = 64
lstm_layers = 4
output_size = 3  # (quasistable->0, nonstationary->1, instability->2, Please replace the labels in the .csv file)


model = LSTM_MLP(input_size, hidden_size, lstm_layers, output_size).to(device)


load_path = r"D:/DATA/model_weights_frozen2003.pth"  # The model has been trained, please use the file directly
model.load_state_dict(torch.load(load_path))
model.eval()
print(f'Model weights loaded from {load_path}')


file_path = r'D:/DATA'  
data, labels = load_data_from_csv(file_path)
if data is None or labels is None:
    raise ValueError("Data loading failed, please check the CSV file")


data, scaler = normalize_data(data)


window_size = 20  
target_offset = 5  
sequences, label_seqs = create_sequences(data, labels, window_size, target_offset)


sequences = sequences.to(device)
label_seqs = label_seqs.to(device)


all_preds = []
true_labels = []
all_probabilities = []  

with torch.no_grad():
    for i in range(len(sequences)):
        test_seq = sequences[i:i + 1]
        test_seq = test_seq.to(device)  
        prediction = model(test_seq)
        predicted_class = torch.argmax(prediction, dim=1)
        all_preds.append(predicted_class.item())
        true_labels.append(label_seqs[i].item())
        all_probabilities.append(prediction.cpu().numpy())  


all_probabilities = np.array(all_probabilities).reshape(-1, output_size)


accuracy = accuracy_score(true_labels, all_preds)
print(f"Accuracy: {accuracy:.4f}")


print("Sample Predictions and True Labels:")
for i in range(5):
    print(f"True Label: {true_labels[i]}, Predicted Label: {all_preds[i]}")


cm = confusion_matrix(true_labels, all_preds)
print("Confusion Matrix:")
print(cm)


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()


report = classification_report(true_labels, all_preds, target_names=['Class 0', 'Class 1', 'Class 2'])
print("Classification Report:")
print(report)


plt.figure(figsize=(10, 6))
plt.plot(true_labels, label='True Labels', marker='o', linestyle='-', color='b')
plt.plot(all_preds, label='Predicted Labels', marker='x', linestyle='--', color='r')
plt.xlabel('Sample Index')
plt.ylabel('Label')
plt.title('True vs Predicted Labels')
plt.legend()
plt.savefig('true_vs_predicted.png')
plt.show()


roc_auc_dict = {}
roc_x_dict = {}
roc_y_dict = {}
roc_data = []  
plt.figure(figsize=(10, 8))

for class_index in range(output_size):
    fpr, tpr, thresholds = roc_curve(true_labels, all_probabilities[:, class_index], pos_label=class_index)
    roc_auc = auc(fpr, tpr)
    roc_auc_dict[f'Class {class_index}'] = roc_auc
    roc_x_dict[class_index] = fpr
    roc_y_dict[class_index] = tpr

  
    plt.plot(fpr, tpr, label=f'Class {class_index} (AUROC = {roc_auc:.2f})')


plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.show()


max_length = max(len(roc_x_dict[class_index]) for class_index in range(output_size))

roc_curve_df = pd.DataFrame()

for class_index in range(output_size):
    fpr_values_padded = np.pad(roc_x_dict[class_index], (0, max_length - len(roc_x_dict[class_index])),
                               constant_values=np.nan)
    tpr_values_padded = np.pad(roc_y_dict[class_index], (0, max_length - len(roc_y_dict[class_index])),
                               constant_values=np.nan)

    roc_curve_df[f'Class {class_index} FPR'] = fpr_values_padded
    roc_curve_df[f'Class {class_index} TPR'] = tpr_values_padded

roc_csv_path = 'roc_curve_data.csv'
roc_curve_df.to_csv(roc_csv_path, index=False)
print(f'ROC curve data saved to {roc_csv_path}')


