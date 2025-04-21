import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import os
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')



def create_sequences(data, labels, window_size, target_offset):
    sequences = []
    label_seqs = []
    for i in range(len(data) - window_size - target_offset + 1):
        seq = data[i:i + window_size]
        label_seq = labels[i + window_size + target_offset - 1]  # Target data point position
        sequences.append(seq)
        label_seqs.append(label_seq)
    return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(label_seqs), dtype=torch.long)



def load_data_from_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")

    df = pd.read_csv(file_path, encoding='latin1')
    data = df.iloc[:, :18].values  
    labels = df.iloc[:, 18].values
    return data, labels



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



def freeze_lstm_layers(model, num_layers_to_freeze):
    for name, param in model.lstm.named_parameters():
        if 'weight_ih_l' in name or 'weight_hh_l' in name or 'bias_l' in name:
            layer_idx = int(name.split('_l')[-1])  # Extract layer index
            if layer_idx < num_layers_to_freeze:
                param.requires_grad = False
                print(f"Freezing parameter: {name}")
            else:
                print(f"Training parameter: {name}")



input_size = 18  
hidden_size = 64
lstm_layers = 4
output_size = 3  # (quasistable->0, nonstationary->1, instability->2, Please replace the labels in the .csv file)
epochs = 400
batch_size = 32
learning_rate = 0.00001
window_size = 20  
target_offset = 5  


folder_path = 'D:/DATA'  


model = LSTM_MLP(input_size, hidden_size, lstm_layers, output_size).to(device)

# Load pre-trained model weights
pretrained_model_path = 'model_weights_n_11.pth'  
if os.path.exists(pretrained_model_path):
    model.load_state_dict(torch.load(pretrained_model_path))
    print(f"Loaded pretrained model weights from {pretrained_model_path}.")
else:
    raise FileNotFoundError(f"Pretrained model file not found at {pretrained_model_path}.")

# Freeze the first two LSTM layers
freeze_lstm_layers(model, num_layers_to_freeze=2)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)


model.train()

for epoch in range(epochs):
    epoch_loss = 0.0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            data, labels = load_data_from_csv(file_path)

            
            scaler = StandardScaler()
            data = scaler.fit_transform(data)

            
            sequences, label_seqs = create_sequences(data, labels, window_size, target_offset)

            
            sequences = sequences.to(device)
            label_seqs = label_seqs.to(device)

            for i in range(0, len(sequences), batch_size):
                x_batch = sequences[i:i + batch_size]
                y_batch = label_seqs[i:i + batch_size]

                optimizer.zero_grad()
                output = model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')


new_model_path = 'model_weights_frozen.pth'
torch.save(model.state_dict(), new_model_path)
print(f"Model weights with frozen layers saved to {new_model_path}.")
