import numpy as np
import torch.nn as nn
import torch


class LSTMWithAttention(nn.Module):

    def __init__(self, vocab_size, hidden_size, output_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        logits = self.fc(last_hidden)
        return logits

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




