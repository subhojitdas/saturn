import numpy as np

class MSELoss:

    def forward(self, outputs, targets):
        loss = 0.0
        for yhat, y in zip(outputs, targets):
            loss += np.sum(yhat - y)**2
        return loss / len(outputs)

    def backward(self, outputs, targets):
        doutputs = []
        for yhat, y in zip(outputs, targets):
            doutputs.append(2 * (yhat - y) / len(outputs))
        return doutputs


class CrossEntropyLoss:
    def forward(self, logits, target):
        self.logits = logits
        self.target = target
        epsilon = 1e-12
        loss = 0.0

        for yhat, y in zip(logits, target):
            shifted_scores = logits - np.max(yhat)
            exp_logits = np.exp(shifted_scores)
            probs = exp_logits / np.sum(exp_logits)
            loss += -np.log(probs[target, 0] + epsilon)
        return loss / len(logits)

    def softmax_numpy(self, x, axis=1):
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def backward(self):
        probs = self.softmax_numpy(self.logits)
        probs[np.arange(self.logits.shape[0]), self.target] -= 1
        dlogits = probs / self.logits.shape[0]
        return dlogits