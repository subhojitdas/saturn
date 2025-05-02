import numpy as np


class LSTMCellBatch:

    def __init__(self, embedding_dim, hidden_size):
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.concat_size = embedding_dim + hidden_size

        self.Wf = np.random.randn(self.hidden_size, self.concat_size) * 0.1
        self.bf = np.zeros((self.hidden_size, 1))
        self.Wi = np.random.randn(self.hidden_size, self.concat_size) * 0.1
        self.bi = np.zeros((self.hidden_size, 1))
        self.Wo = np.random.randn(self.hidden_size, self.concat_size) * 0.1
        self.bo = np.zeros((self.hidden_size, 1))
        self.Wc = np.random.randn(self.hidden_size, self.concat_size) * 0.1
        self.bc = np.zeros((self.hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, xt, hprev, cprev):
        concat = np.vstack((hprev, xt)) # (concat_size, batch_size)

        ft = self.sigmoid(self.Wf @ concat + self.bf)
        it = self.sigmoid(self.Wi @ concat + self.bi)
        ot = self.sigmoid(self.Wo @ concat + self.bo)
        chatt = np.tanh(self.Wc @ concat + self.bc)

        ct = ft * cprev + it * chatt
        ht = ot * np.tanh(ct)
        self.cache = (xt, hprev, cprev, ft, it, ot, chatt, ct, concat)

        return ht, ct

    def backward(self, dh_next, dc_next):
        xt, hprev, cprev, ft, it, ot, chatt, ct, concat = self.cache
        tanh_ct = np.tanh(ct)

        dot = dh_next * tanh_ct
        dct = dh_next * ot * (1 - tanh_ct ** 2) + dc_next

        dft = dct * cprev
        dit = dct * chatt
        dchatt = dct * it
        dc_prev = dct * ft

        dzf = dft * ft * (1 - ft)
        dzi = dit * it * (1 - it)
        dzo = dot * ot * (1 - ot)
        dzc = dchatt * (1 - chatt ** 2)

        dWf = dzf @ concat.T
        dWi = dzi @ concat.T
        dWo = dzo @ concat.T
        dWc = dzc @ concat.T

        dbf = np.sum(dzf, axis=1, keepdims=True)
        dbi = np.sum(dzi, axis=1, keepdims=True)
        dbo = np.sum(dzo, axis=1, keepdims=True)
        dbc = np.sum(dzc, axis=1, keepdims=True)

        dconcat = (
                self.Wf.T @ dzf + self.Wi.T @ dzi + self.Wo.T @ dzo + self.Wc.T @ dzc
        )
        dh_prev = dconcat[:self.hidden_size, :]
        dxt = dconcat[self.hidden_size:, :]

        grads = {
            'dWf': dWf, 'dbf': dbf,
            'dWi': dWi, 'dbi': dbi,
            'dWo': dWo, 'dbo': dbo,
            'dWc': dWc, 'dbc': dbc,
        }

        return dxt, dh_prev, dc_prev, grads


class LSTMLayerBatch:
    def __init__(self, input_size, hidden_size, output_size, embedding_dim):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.embedding = np.random.randn(self.embedding_dim, self.input_size) * 0.1
        self.lstm_batch_cell = LSTMCellBatch(self.embedding_dim, hidden_size)

        self.Wy = np.random.randn(output_size, hidden_size) * 0.1
        self.by = np.zeros((output_size, 1))
        self.cache = []

    def forward(self, inputs, h0, c0):
        # inputs => (seq_len, batch_size, input_size)
        # h0 => (hidden_size, batch_size)
        # c0 => (hidden_size, input_size)
        seq_len, batch_size, _ = inputs.shape
        self.cache = []
        # h0 = np.random.randn(self.hidden_size, self.hidden_size) * 0.1
        # c0 = np.random.randn(self.hidden_size, self.hidden_size) * 0.1
        outputs = []
        ht, ct = h0, c0

        for t in range(seq_len):
            input = inputs[t] # (batch_size, input_size)
            xt = input.T # (input_size, batch_size)

            ht, ct = self.lstm_batch_cell.forward(xt, ht, ct)
            yt = self.Wy @ ht + self.by
            outputs.append((yt, ht, ct))
            self.cache.append((xt, ht, ct, self.lstm_batch_cell.cache))
        return outputs

    def backward(self, dy_list):
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros_like(self.cache[0][1])
        dc_next = np.zeros_like(self.cache[0][2])

        grads = {
            'dWf': 0, 'dbf': 0,
            'dWi': 0, 'dbi': 0,
            'dWo': 0, 'dbo': 0,
            'dWc': 0, 'dbc': 0,
        }

        for t in reversed(range(len(self.cache))):
            xt, ht, ct, lstm_cache = self.cache[t]
            dy = dy_list[t]
            dWy += dy @ ht.T
            dby += dy.sum(axis=1, keepdims=True)
            dh = self.Wy.T @ dy + dh_next
            self.lstm_batch_cell.cache = lstm_cache
            dx, dh_next, dc_next, g = self.lstm_batch_cell.backward(dh, dc_next)
            for k in grads:
                grads[k] += g[k]

        return grads, dWy, dby

    def clip_gradients(self, grads, max_norm=5.0):
        total_norm = sum(np.sum(g ** 2) for g in grads.values())
        total_norm = np.sqrt(total_norm)
        if total_norm > max_norm:
            for k in grads:
                grads[k] *= max_norm / (total_norm + 1e-6)
        return grads

    def update_parameters(self, grads, dWy, dby, lr):
        clipped_grads = self.clip_gradients(grads)
        self.lstm_batch_cell.Wf -= lr * clipped_grads['dWf']
        self.lstm_batch_cell.bf -= lr * clipped_grads['dbf']
        self.lstm_batch_cell.Wi -= lr * clipped_grads['dWi']
        self.lstm_batch_cell.bi -= lr * clipped_grads['dbi']
        self.lstm_batch_cell.Wo -= lr * clipped_grads['dWo']
        self.lstm_batch_cell.bo -= lr * clipped_grads['dbo']
        self.lstm_batch_cell.Wc -= lr * clipped_grads['dWc']
        self.lstm_batch_cell.bc -= lr * clipped_grads['dbc']
        self.Wy += -lr * dWy
        self.by += -lr * dby

