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
        label_seq = labels[i + window_size + target_offset - 1]  
        sequences.append(seq)
        label_seqs.append(label_seq)
    return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(label_seqs), dtype=torch.long)



def load_data_from_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")

    df = pd.read_csv(file_path, encoding='latin1')
    data = df.iloc[:, :18].values  
    # labels = df.iloc[:, 18].values

    # xpwang map the string into the integer
    label_map = {'quasistable': 0, 'nonstationary': 1, 'instability': 2}
    labels_str = df.iloc[:, 18]
    labels = labels_str.map(label_map).astype(np.int64).values
    
    return data, labels


# Define the LSTM and MLP combined model
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


# Hyperparameter settings
input_size = 18 
hidden_size = 64
lstm_layers = 4
output_size = 3  # (quasistable->0, nonstationary->1, instability->2, Please replace the labels in the .csv file)
epochs = 150
batch_size = 32
learning_rate = 0.00001
window_size = 20  
target_offset = 5  

# Load data
folder_path = './Data/Data with PHOENIX'  


model = LSTM_MLP(input_size, hidden_size, lstm_layers, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


train_losses = []

# xpwang revise
# plt.ion()  
# fig, ax = plt.subplots()
# line, = ax.plot([], [], marker='o')
# ax.set_xlim(0, epochs)
# ax.set_ylim(0, 5)  
# ax.set_xlabel('Epochs')
# ax.set_ylabel('Loss')
# ax.set_title('Training Loss Curve')
# plt.grid()

model.train()

for epoch in range(epochs):
    epoch_loss = 0.0
    num_batches = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            data, labels = load_data_from_csv(file_path)


            scaler = StandardScaler()
            data = scaler.fit_transform(data)


            sequences, label_seqs = create_sequences(data, labels, window_size, target_offset)


            sequences = sequences.to(device)
            label_seqs = label_seqs.to(device)


            num_batches = len(sequences) // batch_size

            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                x_batch = sequences[start:end]
                y_batch = label_seqs[start:end]

                optimizer.zero_grad()
                output = model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

    avg_loss = epoch_loss / num_batches if num_batches > 0 else epoch_loss
    train_losses.append(avg_loss)


    # line.set_xdata(range(1, epoch + 2))
    # line.set_ydata(train_losses)
    # ax.set_ylim(0, max(train_losses) * 1.1) 
    # plt.draw()
    # plt.pause(0.01)  

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')


# plt.ioff()
# plt.savefig('loss_curve.png')
# plt.show()

# 创建图像
fig, ax = plt.subplots()
ax.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('Training Loss Curve')
ax.grid(True)
ax.legend()


save_path = 'model_weights.pth'
torch.save(model.state_dict(), save_path)
print(f'Model weights saved to {save_path}')
