import numpy as np
from xgboost.dask import dconcat


class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.concat_size = input_size + hidden_size

        self.Wf = np.random.randn(self.hidden_size, self.concat_size) * 0.1
        self.bf = np.zeros((hidden_size, 1))
        self.Wi = np.random.randn(self.hidden_size, self.concat_size) * 0.1
        self.bi = np.zeros((hidden_size, 1))
        self.Wo = np.random.randn(self.hidden_size, self.concat_size) * 0.1
        self.bo = np.zeros((hidden_size, 1))
        # self.Wg = np.random.randn(self.hidden_size, self.concat_size) * 0.1
        # self.bg = np.zeros((hidden_size, 1))
        self.Wc = np.random.randn(self.hidden_size, self.concat_size) * 0.1
        self.bc = np.zeros((hidden_size, 1))

    def forward(self, xt, hprev, cprev):
        concat = np.vstack((hprev, xt))

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

        tanhct = np.tanh(ct)
        dot = dh_next * tanhct
        dct = dh_next * ot * (1 - tanhct**2) + dc_next

        dft = dct * cprev
        dit = dct * chatt
        dchatt = dct * it
        dc_prev = dct * ft

        dzf = dft * ft * (1 - ft)
        dzi = dit * it * (1 - it)
        dzo = dot * ot * (1 - ot)
        dzchatt = dchatt * (1 - chatt**2)

        dWf = dzf @ concat.T
        dWi = dzi @ concat.T
        dWo = dzo @ concat.T
        dWchatt = dzchatt @ concat.T

        dbf = dzf.sum(axis=1, keepdims=True)
        dbi = dzi.sum(axis=1, keepdims=True)
        dbo = dzo.sum(axis=1, keepdims=True)
        dbc = dzchatt.sum(axis=1, keepdims=True)

        dconcat = self.Wf.T @ dzf + self.Wi.T @ dzi + self.Wo.T @ dzo + self.Wc.T @ dzchatt

        dh_prev = dconcat[:self.hidden_size, :]
        dxt = dconcat[self.hidden_size:, :]

        grads = {
            'dWf': dWf, 'dbf': dbf,
            'dWi': dWi, 'dbi': dbi,
            'dWo': dWo, 'dbo': dbo,
            'dWc': dWchatt, 'dbc': dbc,
        }

        return dxt, dh_prev, dc_prev, grads

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class LSTMLayer:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm_cell = LSTMCell(input_size, hidden_size)

        self.Wy = np.random.randn(self.output_size, self.hidden_size) * 0.1
        self.by = np.zeros((self.output_size, 1))

        self.cache = []

    def forward(self, inputs, h0, c0):
        self.cache = []
        h, c = h0, c0
        outputs = []

        for xt in inputs:
            h, c = self.lstm_cell.forward(xt, h, c)
            yt = self.Wy @ h + self.by
            outputs.append((yt, h, c))
            self.cache.append((xt, h, c, self.lstm_cell.cache))
        return outputs

    def backward(self, dy_list):
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros((self.hidden_size, 1))
        dc_next = np.zeros((self.hidden_size, 1))

        grads = {
            'dWf': 0, 'dbf': 0,
            'dWi': 0, 'dbi': 0,
            'dWo': 0, 'dbo': 0,
            'dWc': 0, 'dbc': 0,
        }

        for t in reversed(range(len(self.cache))):
            xt, h, c, lstm_cache = self.cache[t]
            dy = dy_list[t]
            dWy += dy @ h.T
            dby += dy
            dh = self.Wy.T @ dy + dh_next
            self.lstm_cell.cache = lstm_cache
            dx, dh_next, dc_next, g = self.lstm_cell.backward(dh, dc_next)

            for k in grads:
                grads[k] += g[k]

        return grads, dWy, dby

    def update_parameters(self, grads, dWy, dby, lr):
        self.lstm_cell.Wf += -lr * grads['dWf']
        self.lstm_cell.bf += -lr * grads['dbf']
        self.lstm_cell.Wi += -lr * grads['dWi']
        self.lstm_cell.bi += -lr * grads['dbi']
        self.lstm_cell.Wo += -lr * grads['dWo']
        self.lstm_cell.bo += -lr * grads['dbo']
        self.lstm_cell.Wc += -lr * grads['dWc']
        self.lstm_cell.bc += -lr * grads['dbc']

        self.Wy += -lr * dWy
        self.by += -lr * dby









