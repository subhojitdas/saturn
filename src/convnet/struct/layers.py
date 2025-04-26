import numpy as np


class Linear:

    def __init__(self, fan_in, fan_out, have_bias=True, std_dev=1e-2, weight_decay=0.0):
        self.weights = np.random.randn(fan_in, fan_out) * std_dev
        self.have_bias = have_bias
        self.weight_decay = weight_decay
        if have_bias:
            self.bias = np.zeros(fan_out)
        else:
            self.bias = None

    def forward(self, x):
        self.x = x
        if self.have_bias:
            self.out = np.dot(x, self.weights) + self.bias
        else:
            self.out = np.dot(x, self.weights)
        return self.out

    def backward(self, dout):
        self.dweights = np.dot(self.x.T, dout)
        if self.weight_decay > 0:
            self.dweights += 2 * self.weight_decay * self.weights
        dx = np.dot(dout, self.weights.T)
        if self.have_bias:
            self.dbias = np.sum(dout, axis=0)
        return dx

    def update_param(self, lr):
        self.weights += -lr * self.dweights
        if self.have_bias:
            self.bias += -lr * self.dbias

    def update_parameters(self, lr):
        self.weights += -lr * self.dweights
        if self.have_bias:
            self.bias += -lr * self.dbias

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

    def update_param(self, lr):
        pass


class BatchNorm1d:

    def __init__(self, fan_in, eps=1e-5, momentum=0.9):
        self.epsilon = eps
        self.momentum = momentum
        self.training = True
        self.gamma = np.ones((1, fan_in), dtype=np.float32)
        self.beta = np.zeros((1, fan_in), dtype=np.float32)
        self.running_mean = np.zeros(fan_in)
        self.running_var = np.ones(fan_in)

    def forward(self, x, training=True):
        self.n = x.shape[0]
        if self.training:
            bnmeani = np.mean(x, axis=0, keepdims=True)
            bnvari = np.var(x, axis=0, keepdims=True)
        else:
            bnmeani = self.running_mean
            bnvari = self.running_var
        bndiff = (x - bnmeani)
        bndiff2 = bndiff**2
        bnvar = 1/(self.n - 1)*bndiff2.sum(0, keepdims=True)
        self.bnvar_inv = (bnvar + self.epsilon)**-0.5
        self.xhat = bndiff * self.bnvar_inv
        self.out = self.gamma * self.xhat + self.beta
        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * bnmeani
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * bnvari
        return self.out

    def backward(self, out_grad):
        self.dgamma = (out_grad * self.xhat).sum(0, keepdims=True)
        self.dbeta = out_grad.sum(0, keepdims=True)
        dhprebn = self.gamma*self.bnvar_inv/self.n * (self.n*out_grad - out_grad.sum(0) - self.n/(self.n-1)*self.xhat*(out_grad*self.xhat).sum(0))
        return dhprebn

    def update_param(self, lr):
        self.gamma += -lr * self.dgamma
        self.beta += -lr * self.dbeta

    def parameters(self):
        return [self.gamma, self.beta]


class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_param(self, lr):
        for layer in self.layers:
            layer.update_param(lr)
