import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import torch.nn.functional as F


data_folder = 'D:/DATA'
learning_rate = 1e-5
num_epochs = 500
batch_size = 32
input_size = 13
output_size = 5
num_conditions = 3
loss_function_choice = 'SmoothL1'


class ConditionWeightedBPNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition_proj = nn.Sequential(
            nn.Linear(num_conditions, output_size),
            nn.Sigmoid()  
        )
        self.backbone = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )

    def forward(self, x, cond):
        main_output = self.backbone(x)
        weights = self.condition_proj(cond)
        return main_output * (1 + weights)



def load_data(folder):
    data = []
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder, file), encoding='ISO-8859-1')
            data.append(df)
    return pd.concat(data)


def train_model(model, criterion, optimizer, train_loader):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets, conditions in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs, conditions)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}')
        if torch.isnan(torch.tensor(avg_loss)):
            print("NaN loss detected, stopping training.")
            break

    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve (Condition Weight Multiplication)')
    plt.legend()
    plt.savefig('loss_curve_condition_mul.png')
    plt.show()


if __name__ == '__main__':
    dataset = load_data(data_folder)
    X = dataset.iloc[:, :13].values
    y = dataset.iloc[:, 13:18].values
    conditions = dataset.iloc[:, 18:21].values


    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    scaler_conditions = RobustScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    conditions_scaled = scaler_conditions.fit_transform(conditions)


    X_train, _, y_train, _, cond_train, _ = train_test_split(
        X_scaled, y_scaled, conditions_scaled, test_size=0.1, random_state=3407)


    train_loader = torch.utils.data.DataLoader(
        list(zip(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(cond_train, dtype=torch.float32))),
        batch_size=batch_size, shuffle=True
    )


    model = ConditionWeightedBPNN()


    if loss_function_choice == 'SmoothL1':
        criterion = nn.SmoothL1Loss()
    elif loss_function_choice == 'MSE':
        criterion = nn.MSELoss()
    elif loss_function_choice == 'MAE':
        criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    train_model(model, criterion, optimizer, train_loader)


    torch.save(model.state_dict(), 'CBN_BPNN_model_weights_cond_mul.pth')
