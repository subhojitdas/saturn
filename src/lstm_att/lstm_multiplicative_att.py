import torch
import torch.nn.functional as F
import torch.nn as nn


class LSTMAndMultiplicativeAttentionEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)

    def forward(self, x, lengths):
        x_embed = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(x_embed, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (hn, cn) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out, (hn, cn)


class LSTMAndMultiplicativeAttentionDecoder(nn.Module):

    def __init__(self, hidden_size, output_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim + hidden_size, hidden_size=hidden_size, batch_first=True)
        self.attention = MultiplicativeAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, y, encoder_outputs, hidden, lengths, mask=None):
        y_embed = self.embedding(y) # (B, T, D)
        outputs = []
        h, c = hidden

        for t in range(y.size(1)):
            context, aw = self.attention(h[-1], encoder_outputs, mask)
            self.aw = aw
            lstm_input = torch.cat((y_embed[:, t:t+1, :], context.unsqueeze(1)), dim=2) # (B, 1, D + H)
            out, (h,c) = self.lstm(lstm_input, (h, c))
            logits = self.fc(out.squeeze(1)) # (B, vocab_size)
            outputs.append(logits)

        return torch.stack(outputs, dim=1) # (B, T, vocab_size)


class MultiplicativeAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        # decoder_hidden = (B, H)
        # encoder_outputs = (B, T, H)
        B, T, H = encoder_outputs.shape

        proj_encoder = encoder_outputs @ self.Wa # (B, T, H) * (H, H)
        decoder_exp = decoder_hidden.unsqueeze(1) #  (B, 1, H)
        scores = (decoder_exp * proj_encoder).sum(dim=2) # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=1) # (B, T)
        context = torch.sum(encoder_outputs * attention_weights.unsqueeze(-1), dim=1) # (B, H)
        return context, attention_weights


class EncoderDecoderLSTMWithMA(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, embedding_dim):
        super().__init__()
        self.encoder = LSTMAndMultiplicativeAttentionEncoder(vocab_size, hidden_size, embedding_dim)
        self.decoder = LSTMAndMultiplicativeAttentionDecoder(hidden_size, output_size, embedding_dim)

    def forward(self, x, x_lengths, y, y_lengths):
        encoder_outputs, hidden = self.encoder(x, x_lengths)
        mask = (x != 0).int()
        logits = self.decoder(y, encoder_outputs, hidden, y_lengths, mask=mask)
        return logits


