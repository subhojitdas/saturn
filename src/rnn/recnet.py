import numpy as np

class VanillaRNN:

    def __init__(self, input_size, hidden_size, output_size, seed=42):
        np.random.seed(seed)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        self.Whh = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.Wxh = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)
        self.bh = np.zeros((hidden_size, 1))

        self.Why = np.random.randn(output_size, hidden_size) / np.sqrt(hidden_size)
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        ht = np.zeros((self.hidden_size, 1))
        self.inputs = inputs
        self.outputs = []
        self.hidden = []

        for xt in inputs:
            activation = self.Whh @ ht + self.Wxh @ xt + self.bh
            if np.isnan(activation).any() or np.isinf(activation).any():
                raise ValueError(f"Bad activation detected: {activation}")

            ht = np.tanh(activation)
            y = self.Why @ ht + self.by

            self.outputs.append(y)
            self.hidden.append(ht)

        return self.outputs

    def compute_loss(self, outputs, targets):
        loss = 0.0
        for y_hat, target in zip(outputs, targets):
            exp_logits = np.exp(y_hat - np.max(y_hat))
            probs = exp_logits / np.sum(exp_logits)
            loss += -np.log(probs[target, 0] + 1e-8)
        return loss / len(outputs)

    def backward(self, targets):
        seq_len = len(self.inputs)

        dWhh = np.zeros_like(self.Whh)
        dWxh = np.zeros_like(self.Wxh)
        dbh = np.zeros_like(self.bh)
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(seq_len)):
            xt = self.inputs[t]
            ht = self.hidden[t]
            hprev = self.hidden[t - 1] if t > 0 else np.zeros_like(ht)
            yhat = self.outputs[t]

            exp_logits = np.exp(yhat - np.max(yhat))
            probs = exp_logits / np.sum(exp_logits)

            dy = probs
            dy[targets[t]] -= 1

            dWhy += dy @ ht.T
            dby += dy

            dh = self.Why.T @ dy + dh_next
            da = (1 - ht ** 2) * dh  # tanh derivative

            dWxh += da @ xt.T
            dWhh += da @ hprev.T
            dbh += da

            dh_next = self.Whh.T @ da

            # numerical stability check
            if np.isnan(dh_next).any() or np.isinf(dh_next).any():
                raise ValueError(f"Bad gradient detected at time {t}")

        # 🔥 Gradient Clipping
        for grad in [dWhh, dWxh, dWhy, dbh, dby]:
            np.clip(grad, -5, 5, out=grad)

        self.grads = {
            'Whh': dWhh,
            'bh': dbh,
            'Wxh': dWxh,
            'by': dby,
            'Why': dWhy,
        }
        return self.grads

    def update_parameters(self, lr=0.01):
        self.Whh += -lr * self.grads['Whh']
        self.Wxh += -lr * self.grads['Wxh']
        self.bh += -lr * self.grads['bh']

        self.by += -lr * self.grads['by']
        self.Why += -lr * self.grads['Why']

    def train_step(self, inputs, targets, lr=0.01):
        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs, targets)
        self.backward(targets)
        self.update_parameters(lr)
        return loss




