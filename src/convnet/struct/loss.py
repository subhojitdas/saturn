import numpy as np

from convnet.struct.layers import Linear


class SoftmaxCrossEntropyLoss:

    def forward(self, logits, target):
        self.logits = logits
        self.target = target
        epsilon = 1e-12
        num_examples = logits.shape[0]
        shifted_scores = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_log_probs = -np.log(probs[np.arange(num_examples), target] + epsilon)
        loss = np.mean(correct_log_probs)
        return loss

    def softmax_numpy(self, x, axis=1):
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def backward(self):
        probs = self.softmax_numpy(self.logits)
        probs[np.arange(self.logits.shape[0]), self.target] -= 1
        dlogits = probs / self.logits.shape[0]
        return dlogits

    def l2_regularization(self, model, weight_decay):
        reg = 0.0
        for layer in model.layers:
            if isinstance(layer, Linear):
                w = layer.weights
                reg += np.sum(w ** 2)
        return reg * weight_decay

