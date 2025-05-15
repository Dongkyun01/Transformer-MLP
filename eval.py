# eval.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import DataLoader, random_split
from dataset import RetailRocketDataset, collate_fn
from model import TransformerWithMLP
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
# 로그 역변환
def inverse_log_transform(x):
    return np.expm1(x)

# 1. 데이터 불러오기
with open('retailrocket_dataset.pkl', 'rb') as f:
    obj = pickle.load(f)

sequences = obj['session_event_sequences']
features = obj['additional_features']
targets = torch.tensor(np.log1p(obj['time_to_next_session'].values), dtype=torch.float32)

dataset = RetailRocketDataset(sequences, features, targets)

# 2. Test Set 준비
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
_, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# 3. 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerWithMLP(num_tokens=3, additional_feature_dim=8).to(device)
model.load_state_dict(torch.load("transformer_with_mlp.pt"))
model.eval()

# 4. 예측
all_preds, all_targets, session_lengths = [], [], []

with torch.no_grad():
    for batch_seq, batch_feat, batch_target in test_loader:
        batch_seq, batch_feat, batch_target = batch_seq.to(device), batch_feat.to(device), batch_target.to(device)
        pred = model(batch_seq, batch_feat)

        all_preds.append(pred.cpu())
        all_targets.append(batch_target.cpu())
        session_lengths += [len(seq[seq > 0]) for seq in batch_seq.cpu()]

# 5. 역변환
preds = inverse_log_transform(torch.cat(all_preds).numpy())
targets = inverse_log_transform(torch.cat(all_targets).numpy())

# 6. 평가 지표
mse = np.mean((preds - targets)**2)
mae = np.mean(np.abs(preds - targets))
print(f"✅ Test MSE: {mse:.4f}")
print(f"✅ Test MAE: {mae:.4f}")

# 7. 세션 길이별 성능
lengths = np.array(session_lengths)
mae_short = np.mean(np.abs(preds[lengths < 5] - targets[lengths < 5]))
mae_long = np.mean(np.abs(preds[lengths >= 5] - targets[lengths >= 5]))

print(f"MAE (Short sessions <5): {mae_short:.4f}")
print(f"MAE (Long sessions >=5): {mae_long:.4f}")

# 8. 히스토그램
plt.figure(figsize=(8,4))
plt.hist(preds, bins=50, alpha=0.7, label='Predicted')
plt.hist(targets, bins=50, alpha=0.5, label='Actual')
plt.legend()
plt.title("Prediction Histogram")
plt.xlabel("Time to Next Session (seconds)")
plt.grid(True)
plt.tight_layout()
plt.savefig("prediction_histogram.png")
print("✅ 히스토그램 저장 완료")

