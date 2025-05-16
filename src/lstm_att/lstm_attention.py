import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn.functional as F

# data preparation
dataset_dir = '/Users/subhojit/datasets/sms_spam_collection'
df = pd.read_csv(dataset_dir + "/SMSSpamCollection", sep='\t', header=None, names=['label', 'text'])

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
texts = df['text'].tolist()
labels = df['label'].tolist()

chars = sorted(set(''.join(texts)))
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi['<PAD>'] = 0
vocab_size = len(stoi)
encode = lambda s: [stoi[c] for c in s if c in stoi]

xtrain, xval, ytrain, yval = train_test_split(texts, labels, test_size=0.2, random_state=1894)

def pad_sequences(sequences, max_len=256):
    padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
    lengths = torch.zeros(len(sequences), dtype=torch.long)
    for i, seq in enumerate(sequences):
        seq = seq[:max_len]
        padded[i, :len(seq)] = torch.tensor(seq)
        lengths[i] = len(seq)
    return padded, lengths

def get_batch(batch_size, split='train'):
    x = xtrain if split == 'train' else xval
    y = ytrain if split == 'train' else yval
    idx = torch.randint(0, len(x), (batch_size,))
    xb = [encode(x[i]) for i in idx]
    yb = [y[i] for i in idx]
    xb, lengths = pad_sequences(xb)
    return xb, torch.tensor(yb, dtype=torch.long), lengths


class LSTMAndAdditiveAttention(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, embedding_dim, attention_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.attention = DRawAdditiveAttention(hidden_size, attention_size=attention_dim)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        x_embed = self.embedding(x)
        out, _ = self.lstm(x_embed)
        mask = (x != 0).int()
        context = self.attention(out, mask)
        logits = self.fc(context)
        return logits, context


class DRawAdditiveAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, lstm_out, Wa, va, mask=None):
        B, T, H = lstm_out.shape

        flat_h = lstm_out.reshape(B*T, H)
        zt = torch.matmul(flat_h, Wa.T)
        at = torch.tanh(zt)
        scores = torch.matmul(at, va).reshape(B, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=1)
        weighted = lstm_out * attention_weights.unsqueeze(-1) # (B, T, H)
        context = torch.sum(weighted, dim=1) # (B, H)

        # Save for backward
        ctx.save_for_backward(lstm_out, Wa, va, at, attention_weights, mask)

        return context

    @staticmethod
    def backward(ctx, dcontext):
        lstm_out, Wa, va, at, attn_weights, mask = ctx.saved_tensors
        B, T, H = lstm_out.shape

        dattention = lstm_out * dcontext.unsqueeze(1)
        dattention_weights = dattention.sum(dim=2)  # (B, T)

        dscores = torch.zeros_like(dattention_weights)
        for b in range(B):
            a = attn_weights[b]
            da = dattention_weights[b]
            dscores[b] = a * (da - (a * da).sum())

        dat = dscores.reshape(B * T, 1) @ va.T  # (B*T, A)
        dzt = dat * (1 - at ** 2)  # tanh grad

        flat_h = lstm_out.reshape(B * T, H)
        dWa = dzt.T @ flat_h  # (A, H)
        dva = at.T @ dscores.reshape(B * T, 1)  # (A, 1)

        dht_from_att = (dzt @ Wa).reshape(B, T, H)
        dht_from_context = attn_weights.unsqueeze(-1) * dcontext.unsqueeze(1)
        dlstm_out = dht_from_att + dht_from_context  # (B, T, H)

        return dlstm_out, dWa, dva, None


class DRawAdditiveAttention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super().__init__()
        self.Wa = nn.Parameter(torch.randn(attention_size, hidden_size) * 0.1)
        self.va = nn.Parameter(torch.randn(attention_size, 1) * 0.1)

    def forward(self, lstm_out, mask=None):
        return DRawAdditiveAttentionFunction.apply(lstm_out, self.Wa, self.va, mask)


embedding_dim = 32
hidden_size = 256
output_size = 2
batch_size = 256
attention_dim = 64
seq_len = 10
learning_rate = 1e-2
max_iter = 5000
eval_interval = 500
device = "mps"

def one_batch_overfit():
    model = LSTMAndAdditiveAttention(vocab_size, hidden_size, output_size, embedding_dim, attention_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    xb, yb, lengths = get_batch(batch_size, split='train')
    xb, yb, lengths = xb.to(device), yb.to(device), lengths.to(device)

    for step in range(100):
        logits, context = model(xb, lengths)
        loss = F.cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step}, loss = {loss.item():.4f}")


def lets_train():
    model = LSTMAndAdditiveAttention(vocab_size, hidden_size, output_size, embedding_dim, attention_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iter):
        xb, yb, lengths = get_batch(batch_size, split='train')
        xb, yb, lengths = xb.to(device), yb.to(device), lengths.to(device)

        logits, context = model(xb, lengths)
        loss = F.cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            print(f"Step {step}, loss = {loss.item():.4f}")

    return model


# model = lets_train()


# one_batch_overfit()
