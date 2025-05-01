import numpy as np


class LSTMCellBatch:

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.concat_size = input_size + hidden_size

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
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm_batch_cell = LSTMCellBatch(input_size, hidden_size)

        self.Wy = np.random.randn(output_size, hidden_size) * 0.1
        self.by = np.zeros((output_size, 1))
        self.cache = []

    def forward(self, inputs, h0, c0):
        # inputs => (seq_len, batch_size, input_size)
        # h0 => (hidden_size, batch_size)
        # c0 => (hidden_size, input_size)
        seq_len, batch_size, _ = inputs.shape




