import numpy as np


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
        self.Wg = np.random.randn(self.hidden_size, self.concat_size) * 0.1
        self.bg = np.zeros((hidden_size, 1))
        self.Wc = np.random.randn(self.hidden_size, self.concat_size) * 0.1
        self.bc = np.zeros((hidden_size, 1))

    def forward(self, xt, hprev, cprev):
        concat = np.hstack((hprev, xt))

        ft = self.sigmoid(self.Wf @ concat + self.bf)
        it = self.sigmoid(self.Wi @ concat + self.bi)
        ot = self.sigmoid(self.Wo @ concat + self.bo)
        chatt = np.tanh(self.Wc @ concat + self.bg)

        ct = ft * cprev + it * chatt
        ht = ot * np.tanh(ct)
        self.cache = (xt, hprev, cprev, ft, it, ot, chatt, ct, concat)
        return ht, ct

    def backward(self, ht, ct):
        


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
