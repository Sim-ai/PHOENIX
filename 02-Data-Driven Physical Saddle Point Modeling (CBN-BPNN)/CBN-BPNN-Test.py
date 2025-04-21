import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch.nn.functional as F


test_folder = 'D:/DATA'
output_folder = 'D:/DATA'
input_size = 13
output_size = 5
num_conditions = 3


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



def load_all_test_data(folder):
    data = []
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder, file), encoding='ISO-8859-1')
            data.append(df)
    return pd.concat(data, ignore_index=True)


def predict(model, inputs, conditions):
    model.eval()
    with torch.no_grad():
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        cond_tensor = torch.tensor(conditions, dtype=torch.float32)
        return model(inputs_tensor, cond_tensor).numpy()


os.makedirs(output_folder, exist_ok=True)


dataset = load_all_test_data(test_folder)
X_test = dataset.iloc[:, :input_size].values
y_test = dataset.iloc[:, input_size:input_size + output_size].values
conditions_test = dataset.iloc[:, input_size + output_size:input_size + output_size + num_conditions].values


scaler_X = RobustScaler()
scaler_y = RobustScaler()
scaler_c = RobustScaler()
X_test_scaled = scaler_X.fit_transform(X_test)
y_test_scaled = scaler_y.fit_transform(y_test)
cond_scaled = scaler_c.fit_transform(conditions_test)


model = ConditionWeightedBPNN()
model.load_state_dict(torch.load('CBN_BPNN_model_weights_cond_mul3.pth'))


y_pred_scaled = predict(model, X_test_scaled, cond_scaled)


print("\n[Singl]:")
for i in range(output_size):
    y_true_col = y_test_scaled[:, i]
    y_pred_col = y_pred_scaled[:, i]
    mae = mean_absolute_error(y_true_col, y_pred_col)
    mse = mean_squared_error(y_true_col, y_pred_col)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_col, y_pred_col)
    n = len(y_true_col)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2) if n > 2 else r2

    print(f"Singl {i+1}: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, Adj R²={adj_r2:.4f}")


    plt.figure(figsize=(10, 4))
    plt.plot(y_true_col, label='True', color='blue')
    plt.plot(y_pred_col, label='Predicted', color='orange', linestyle='--')
    plt.title(f'Output {i + 1} - True vs Predicted (Normalized)')
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'output_{i + 1}_curve.png'))
    plt.close()


mae_total = mean_absolute_error(y_test_scaled, y_pred_scaled)
mse_total = mean_squared_error(y_test_scaled, y_pred_scaled)
rmse_total = np.sqrt(mse_total)
r2_total = r2_score(y_test_scaled, y_pred_scaled)
n_total = y_test_scaled.shape[0]
p_total = y_test_scaled.shape[1]
adj_r2_total = 1 - (1 - r2_total) * (n_total - 1) / (n_total - p_total - 1)

print("\n[All]:")
print(f'MAE={mae_total:.4f}, MSE={mse_total:.4f}, RMSE={rmse_total:.4f}, R²={r2_total:.4f}, Adj R²={adj_r2_total:.4f}')
