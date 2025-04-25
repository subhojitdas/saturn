import numpy as np


class Linear:

    def __init__(self, fan_in, fan_out, have_bias=True, std_dev=1e-2):
        self.weights = np.random.randn(fan_in, fan_out) * std_dev
        self.have_bias = have_bias
        if have_bias:
            self.bias = np.zeros(fan_out)
        else:
            self.bias = None

    def forward(self, x):
        self.x = x
        self.out = np.dot(x, self.weights, out=x) + self.bias
        return self.out

    def backward(self, dout):
        dweights = np.dot(self.x.T, dout)
        dx = np.dot(dout, self.weights.T)
        if self.have_bias:
            dbias = np.sum(dout, axis=0)
        return dx, dweights, dbias

    def parameters(self):
        return [self.weights] + [self.bias] if self.have_bias else [self.weights]


class ReLU:

    def forward(self, x):
        self.x = x
        self.out = np.maximum(0, x)
        return self.out

    def backward(self, dout):
        dx = dout * (self.x > 0)
        return dx

    def parameters(self):
        return []

