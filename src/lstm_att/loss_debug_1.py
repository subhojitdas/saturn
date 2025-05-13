import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import pandas as pd


dataset_dir = '/Users/subhojit/datasets/amazon_review_polarity_csv'
df_train = pd.read_csv(dataset_dir + '/train.csv')
df_test = pd.read_csv(dataset_dir + '/test.csv')

review = df_train.iloc[:, 2].to_numpy()
test_review = df_test.iloc[:, 2].to_numpy()
all_review = np.concatenate((review, test_review))
c = sorted(list(set(''.join(all_review))))
chars = c + ['<SOS>', '<EOS>', '<PAD>']
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

x = df_train.iloc[:, 2].to_numpy()
y = df_train.iloc[:, 0].to_numpy()
n = int(0.9*len(x))
xtrain = x[:n]
ytrain = y[:n]
xval = x[n:]
yval = y[n:]

xtest = df_test.iloc[:, 2].to_numpy()
ytest = df_test.iloc[:, 0].to_numpy()

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[xi] for xi in s]
decode = lambda l: ''.join([itos[li] for li in l])
encode('asdasdsadas')
device = "cpu"

def pad_sequences(sequences):
    pad_index = stoi['<PAD>']
    max_len = np.max([len(s) for s in sequences])
    padded_seq = np.full((len(sequences), max_len), pad_index, dtype=np.int32)
    for i, seq in enumerate(sequences):
        padded_seq[i, :len(seq)] = seq
    return padded_seq


def get_batch(batch_size, split='train'):
    data = xtrain if split == 'train' else xval
    target = ytrain if split == 'train' else yval
    idx = np.random.randint(0, len(data), (batch_size,))
    x_sample = [encode(s) for s in data[idx]]
    y_sample = target[idx]
    xpadded = pad_sequences(x_sample)
    xb, yb = xpadded, y_sample
    yb = torch.from_numpy(yb)
    yb = yb - 1
    # yb = torch.nn.functional.one_hot(yb - 1, num_classes=2)
    xb = torch.from_numpy(xb)
    x = xb.to(device, dtype=torch.long)
    y = yb.to(device, dtype=torch.long)
    return x, y


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.embedding(x)            # (B, T, D)
        output, (hn, cn) = self.lstm(x)  # output: (B, T, H)
        last_hidden = output[:, -1, :]   # (B, H)  <-- last time step
        logits = self.fc(last_hidden)    # (B, C)
        return logits