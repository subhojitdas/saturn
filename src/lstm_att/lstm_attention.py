import torch.nn as nn
import torch


class LSTMAndAdditiveAttention(nn.Module):

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


class DrawAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = torch.randn(hidden_size, hidden_size)