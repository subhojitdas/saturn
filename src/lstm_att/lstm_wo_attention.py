import numpy as np
import torch.nn as nn
import torch


class LSTMWithoutAttention(nn.Module):

    def __init__(self, vocab_size, hidden_size, output_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        x_embed = self.embedding(x)
        out, _ = self.lstm(x_embed)

        batch_size = x.size(0)
        last_hidden = torch.zeros(batch_size, out.size(2), device=x.device)
        for i in range(batch_size):
            last_hidden[i] = out[i, lengths[i] - 1]

        return self.fc(last_hidden)

class DrawAttention:

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.Wa = np.random.randn(hidden_size, hidden_size) * 0.1
        self.va = np.random.randn(hidden_size, 1)

        self.dWa = np.zeros_like(self.Wa)
        self.dva = np.zeros_like(self.va)

        self.lstm_out = None
        self.zt = None
        self.at = None
        self.scores = None
        self.attention_weights = None

    def forward(self, lstm_out):
        self.lstm_out = lstm_out
        B, T, H = lstm_out.shape

        zt = lstm_out @ self.Wa.T
        at = np.tanh(zt)
        scores = at @ self.va
        attention_weights = torch.softmax(scores, dim=1)

        context = np.sum(lstm_out * attention_weights.unsqueeze(-1), axis=1)

        # Cache for backward
        self.zt = zt
        self.at = at
        self.scores = scores
        self.attention_weights = attention_weights

        return context, attention_weights

    def backward(self, dcontext):
        pass




