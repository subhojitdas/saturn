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


df_filtered = df_train[df_train.iloc[:, 2].str.len() < 256]
x_filtered = df_filtered.iloc[:, 2].to_numpy()
y_filtered = df_filtered.iloc[:, 0].to_numpy()

n = int(0.9*len(x_filtered))
xtrain = x_filtered[:n]
ytrain = y_filtered[:n]
xval = x_filtered[n:]
yval = y_filtered[n:]

xtest = df_test.iloc[:, 2].to_numpy()
ytest = df_test.iloc[:, 0].to_numpy()

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[xi] for xi in s]
decode = lambda l: ''.join([itos[li] for li in l])

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
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
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.embedding(x)            # (B, T, D)
        output, (hn, cn) = self.lstm(x)  # output: (B, T, H)
        last_hidden = output[:, -1, :]   # (B, H)  <-- last time step
        logits = self.fc(last_hidden)    # (B, C)
        return logits


class MLPClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(emb_dim * max_seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)  # (B, T, D)
        return self.net(x)


embedding_dim = 32
hidden_size = 256
output_size = 2
batch_size = 64
seq_len = 10
learning_rate = 1e-2
max_iter = 5000
eval_interval = 500

# one batch overfitting

# xb, yb = get_batch(10)
# xb, yb = xb.to(device), yb.to(device)
model = LSTMClassifier(vocab_size, hidden_size, output_size, embedding_dim)
# model = MLPClassifier(vocab_size, embedding_dim, hidden_size, output_size, xb.shape[1])
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

# 1-batch overfitting loop
# for step in range(1000):
#     # print("Input:", xb.shape, xb.dtype, xb.device)
#     # print("Target:", yb.shape, yb.dtype, yb.device)
#     logits = model(xb)
#     loss = F.cross_entropy(logits, yb)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print(f"Step {step}, loss = {loss.item():.4f}")
#     if step % 100 == 0:
#         with torch.no_grad():
#             print("Logits:", logits[:2])
#             preds = torch.argmax(logits, dim=1)
#             print("Preds:", preds.tolist())
#             print("Targets:", yb.tolist())

for step in range(max_iter):
    xb, yb = get_batch(batch_size)
    logits = model(xb)
    loss = F.cross_entropy(logits, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        print("Confidence range:", probs.max(dim=1).values[:10])
        preds = torch.argmax(logits, dim=1)
        print("Preds: ", preds.tolist())
        print("Targets: ", yb.tolist())

    if step % eval_interval == 0:
        print(f"step {step}: train loss {loss:.4f}")