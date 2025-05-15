# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from torch.utils.data import DataLoader, random_split
from dataset import RetailRocketDataset, collate_fn
from model import TransformerWithMLP

# 로그 스케일 변환 함수
def log_transform(x):
    return np.log1p(x)

# 1. 데이터 불러오기
with open('retailrocket_dataset.pkl', 'rb') as f:
    obj = pickle.load(f)

sequences = obj['session_event_sequences']
features = obj['additional_features']
targets = torch.tensor(log_transform(obj['time_to_next_session'].values), dtype=torch.float32)

# 2. Dataset 준비
dataset = RetailRocketDataset(sequences, features, targets)

# 3. Train/Test 분리
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, _ = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# 4. 모델 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerWithMLP(num_tokens=3, additional_feature_dim=8).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

# 5. 학습 루프
n_epochs = 20
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0

    for batch_seq, batch_feat, batch_target in train_loader:
        batch_seq, batch_feat, batch_target = batch_seq.to(device), batch_feat.to(device), batch_target.to(device)

        optimizer.zero_grad()
        output = model(batch_seq, batch_feat)
        loss = criterion(output, batch_target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

# 6. 저장
torch.save(model.state_dict(), "transformer_with_mlp.pt")
print("✅ 모델 저장 완료!")


