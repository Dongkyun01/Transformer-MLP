import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class RetailRocketDataset(Dataset):
    def __init__(self, sequences, features, targets):
        self.sequences = sequences         # pandas Series
        self.features = features           # pandas DataFrame or Series
        self.targets = targets             # torch.Tensor

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences.iloc[idx]     # OK (Series)
        feat = self.features.iloc[idx]     # OK (DataFrame or Series)
        target = self.targets[idx]         # Tensor이므로 iloc x

        return (
            torch.LongTensor(seq),
            torch.FloatTensor(feat),
            torch.tensor(target.clone().detach(), dtype=torch.float32)  # 또는 target.clone().detach()
        )

def collate_fn(batch):
    sequences, features, targets = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    features_tensor = torch.stack(features)
    targets_tensor = torch.stack(targets)
    return sequences_padded, features_tensor, targets_tensor
