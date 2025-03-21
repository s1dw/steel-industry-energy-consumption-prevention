#%% 載入模組
import torch
from torch import nn
from torch.optim import Adam, AdamW
import torch.nn.functional as F 
import numpy as np
import os
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
def set_seed(seed=42):
    random.seed(seed)               # 設置 Python 的隨機種子
    np.random.seed(seed)            # 設置 NumPy 的隨機種子
    torch.manual_seed(seed)         # 設置 PyTorch 的隨機種子（CPU）
    torch.cuda.manual_seed(seed)    # 設置 PyTorch 的隨機種子（單個 GPU）
    torch.cuda.manual_seed_all(seed) # 設置 PyTorch 的隨機種子（多個 GPU）
    # 保證 CUDA 環境下的可復現性（但會影響運行效率）
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  
set_seed(0)
#%% 讀取與預處理數據
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 讀取數據
data = pd.read_csv("data\Steel_industry_data.csv", parse_dates=["date"])
end = int(len(data) * 1)
data = data[:end]
data["date"] = pd.to_datetime(data["date"], format="%d/%m/%Y %H:%M")
data["date_only"] = data["date"].dt.date  
count_per_day = data.groupby("date_only").size()

data['WeekStatus'] = data['WeekStatus'].map({'Weekday':0, 'Weekend':1})
data['Load_Type'] = data['Load_Type'].map({'Light_Load':0, 'Medium_Load':1, 'Maximum_Load':2})

# 提取時間特徵
data["month"] = data["date"].dt.month
data["Day_of_week"] = data["date"].dt.dayofweek

"""feature scaling"""
scaler = StandardScaler()
features = data[['month', 'Day_of_week', 'WeekStatus', 'NSM', 'Lagging_Current_Reactive.Power_kVarh', 
                 'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)', 'Lagging_Current_Power_Factor',
                 'Leading_Current_Power_Factor', 'Load_Type', 'Usage_kWh']].values
scaler.fit(features)
features_scaled = scaler.transform(features)
"""label scaling"""
yScaler = StandardScaler()
label = data[['Usage_kWh']].values
yScaler.fit(label)
label_scaled = yScaler.transform(label)

# 繪製數據分布
plt.figure(figsize=(16, 4)) 
plt.plot(data["NSM"][:500].values, label_scaled[:500].flatten())
plt.show()
#%% 建立滑動窗口函數
def sliding_window(data, seq_len, forecasting_len):
    X, xf, y = [], [], []
    for i in range(len(data) - seq_len - forecasting_len):
        X.append(features_scaled[i:i+seq_len])
        xf.append(features_scaled[i+seq_len:i+seq_len+forecasting_len, :4])
        y.append(label_scaled[i+seq_len:i+seq_len+forecasting_len])
    return np.asarray(X), np.asarray(xf), np.asarray(y)

seq_len = 5
forecasting_len = 1

# 產生數據
X, xf, y = sliding_window(data, seq_len, forecasting_len)
print(f"X shape: {X.shape}, xf shape: {xf.shape}, y shape: {y.shape}")

# 拆分訓練與測試集
X_train, X_temp, xf_train, xf_temp, y_train, y_temp = train_test_split(X, xf, y, test_size=0.3, random_state=0, shuffle=False)
X_val, X_test, xf_val, xf_test, y_val, y_test = train_test_split(X_temp, xf_temp, y_temp, test_size=0.5, random_state=0, shuffle=False)

print(f"Train Set - X: {X_train.shape}, xf: {xf_train.shape}, y: {y_train.shape}")
print(f"Validation Set - X: {X_val.shape}, xf: {xf_val.shape}, y: {y_val.shape}")
print(f"Test Set - X: {X_test.shape}, xf: {xf_test.shape}, y: {y_test.shape}")

#%% 定義 MQRNN 模型
class LocalMLP(nn.Module):
    def __init__(self, xf_feature_num, output_horizon, num_quantiles, hidden_dim=10):
        super(LocalMLP, self).__init__()
        input_dim = xf_feature_num * (1 + 1 + output_horizon)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),                       
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),                        
            nn.Linear(hidden_dim // 2, num_quantiles)
        )
    def forward(self, x):
        return self.mlp(x)

'''GlobalMLP 的隱藏層數量會嚴重影響預測結果'''
class GlobalMLP(nn.Module):
    def __init__(self, encoder_hidden_size, xf_feature_num, num_quantiles, hidden_dim=64):
        super(GlobalMLP, self).__init__()
        input_dim = encoder_hidden_size + xf_feature_num
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),                       
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),                        
            nn.Linear(hidden_dim // 2, num_quantiles)
        )
    def forward(self, x):
        return self.mlp(x)
class Decoder(nn.Module):
    def __init__(self, encoder_hidden_size, output_horizon, xf_feature_num, num_quantiles):
        super(Decoder, self).__init__()
        self.global_mlp = GlobalMLP(encoder_hidden_size, xf_feature_num, xf_feature_num)
        self.local_mlp = LocalMLP(xf_feature_num, output_horizon, num_quantiles)
    def forward(self, ht, xf):
        num_ts, output_horizon, _ = xf.shape
        ht = ht.unsqueeze(1).expand(num_ts, output_horizon, ht.size(1))  # 複製 output_horizon 倍
        inp = torch.cat([xf, ht], dim=2)
        contexts = self.global_mlp(inp)
        y = []
        for i in range(output_horizon):
            ca = contexts
            ci = contexts[:, i, :].unsqueeze(1)
            xfi = xf[:, i, :].unsqueeze(1)
            inp = torch.cat([xfi, ci, ca], dim=1).view(inp.size(0), 1, -1)
            y.append(self.local_mlp(inp))
        return torch.cat(y, dim=1)
class MQRNN(nn.Module):
    def __init__(self, output_horizon, num_quantiles, input_size,
                 encoder_hidden_size, encoder_n_layers=3, xf_feature_num=None):
        super(MQRNN, self).__init__()
        self.encoder = nn.LSTM(input_size, encoder_hidden_size, encoder_n_layers, batch_first=True)
        self.decoder = Decoder(encoder_hidden_size, output_horizon, xf_feature_num, num_quantiles)
        self.dense = nn.Linear(encoder_hidden_size, encoder_hidden_size)
    def forward(self, x, xf):
        _, (h, c) = self.encoder(x)
        ht = self.dense(h[-1, :, :])
        ht = F.relu(ht)
        return self.decoder(ht, xf)
#%% 訓練

n_layers = 3
encoder_hidden_size = 10
quantiles = [0.1, 0.5, 0.9]
num_quantiles = len(quantiles)
num_epoches = 100

lr = 1e-3
batch_size = 64
best_val_loss = float('inf')
best_model_state = None
patience = 10           # 早停機制的耐心參數
patience_counter = 0    # 記錄連續未改善的 epoch 數
train_losses = []       # 訓練損失
val_losses = []         # 驗證損失

model = MQRNN(
    forecasting_len, 
    num_quantiles, 
    X.shape[-1], 
    encoder_hidden_size, 
    n_layers,
    xf.shape[-1]
).to(device)

optimizer = AdamW(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# 轉換為 Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
xf_train = torch.tensor(xf_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
xf_val = torch.tensor(xf_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# 建立 DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, xf_train, y_train)
train_Loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
val_dataset = torch.utils.data.TensorDataset(X_val, xf_val, y_val)
val_Loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

for epoch in range(num_epoches):  
    model.train()  # 訓練模式
    losses = []    # 記錄每個批次的訓練損失
    
    # 訓練過程
    for x, xf, y in train_Loader:
        x, xf, y = x.to(device), xf.to(device), y.to(device)
        y = y.squeeze(2)
        ypred = model(x, xf)
        
        # 計算 quantile loss
        loss = torch.zeros_like(y)
        for q, rho in enumerate(quantiles):
            e = y - ypred[:, :, q]
            loss += torch.max(rho * e, (rho - 1) * e)
        loss = loss.mean()
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 訓練損失輸出
    train_loss = sum(losses) / len(losses)
    train_losses.append(train_loss)

    # 驗證過程
    model.eval()  # 評估模式
    val_losses_epoch = []  # 記錄驗證損失
    
    with torch.no_grad():  # 不計算梯度
        for x_val, xf_val, y_val in val_Loader:
            x_val, xf_val, y_val = x_val.to(device), xf_val.to(device), y_val.to(device)
            y_val = y_val.squeeze(2)
            ypred_val = model(x_val, xf_val)

            # 計算 quantile loss
            val_loss = torch.zeros_like(y_val)
            for q, rho in enumerate(quantiles):
                e = y_val - ypred_val[:, :, q]
                val_loss += torch.max(rho * e, (rho - 1) * e)
            val_loss = val_loss.mean()
            val_losses_epoch.append(val_loss.item())
    
    # 驗證損失輸出
    val_loss = sum(val_losses_epoch) / len(val_losses_epoch)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, lr:{scheduler.get_last_lr()}")
    
    # 早停機制
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()  # 保存最佳模型的參數
        patience_counter = 0  # 重置早停計數器
    else:
        patience_counter += 1
        print(f"Validation loss has not improved for {patience_counter} epochs.")
    
    # 如果驗證損失連續幾個 epoch 都沒改善，就停止訓練
    if patience_counter >= patience:
        print(f"Early stopping triggered after {patience} epochs without improvement.")
        break

    # 更新學習率 (退火排程)
    scheduler.step(val_loss)
    
#%% 訓練結束後，將最佳的模型參數加載回來
model.load_state_dict(best_model_state)
print("Best model selected based on validation loss.")

# 繪製訓練和驗證損失曲線
plt.plot(train_losses, color='r', label="Training Loss")
plt.plot(val_losses, color='b', label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

# 測試與繪製預測圖
X_test = torch.tensor(X_test, dtype=torch.float32)
xf_test = torch.tensor(xf_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
print(f"Test Set - X: {X_test.shape}, xf: {xf_test.shape}, y: {y_test.shape}")

#%% 測試與繪圖
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
xf_test_tensor = torch.tensor(xf_test, dtype=torch.float32).to(device)
with torch.no_grad():
    y_pred = model(X_test_tensor, xf_test_tensor).cpu().numpy()

y_pred = yScaler.inverse_transform(y_pred.squeeze(1))
y_test_real = yScaler.inverse_transform(y_test.squeeze(1))

start = 96 * 0
end = 70
x_range = range(len(y_test_real[start:end]))
plt.figure(figsize=(12, 5))
plt.plot(x_range, y_test_real[start:end], label="Actual")
plt.plot(x_range, y_pred[start:end, 1], label="Predicted")
plt.fill_between(x = x_range, y1 = y_pred[start:end, 0], y2 = y_pred[start:end, 2], alpha=0.5)
plt.legend()
plt.show()

a = y_test_real[start:end].flatten()
b = y_pred[start:end, 1].flatten()
mape = np.mean(np.abs(a - b) / (a))
print(f'MAPE: {mape:.2f}%')
