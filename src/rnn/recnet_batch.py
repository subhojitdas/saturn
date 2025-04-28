import numpy as np

class VanillaBatchRNN:

    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))

        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        # inputs shape: (seq_len, batch_size, input_size, 1)
        batch_size = inputs.shape[1]
        seq_len = inputs.shape[0]
        ht = np.zeros((batch_size, self.hidden_size))  # (batch_size, hidden_size)
        self.inputs = inputs
        self.outputs = []
        self.hidden = []

        for t in range(seq_len):
            xt = inputs[t]  # (batch_size, input_size, 1)
            xt = xt.squeeze(-1)  # (batch_size, input_size)

            activation = xt @ self.Wxh.T + ht @ self.Whh.T + self.bh.T  # (batch_size, hidden_size)
            ht = np.tanh(activation)  # (batch_size, hidden_size)
            y = ht @ self.Why.T + self.by.T  # (batch_size, output_size)

            self.outputs.append(y)
            self.hidden.append(ht)

        return np.array(self.outputs)

    def backward(self, targets):
        # targets is of shape (batch_size, seq_len)
        batch_size = targets.shape[0]
        seq_len = targets.shape[1]

        dWhh = np.zeros_like(self.Whh)
        dWxh = np.zeros_like(self.Wxh)
        dbh = np.zeros_like(self.bh)
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros((batch_size, self.hidden_size))

        for t in reversed(range(seq_len)):
            xt = self.inputs[:, t, :]  # Shape: (batch_size, input_size)
            ht = self.hidden[t]  # Shape: (batch_size, hidden_size)
            hprev = self.hidden[t - 1] if t > 0 else np.zeros_like(ht)  # Shape: (batch_size, hidden_size)
            yhat = self.outputs[t]  # Shape: (batch_size, output_size)

            # Calculate softmax loss gradient
            exp_logits = np.exp(yhat - np.max(yhat, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            dy = probs
            batch_targets = targets[:, t]  # Shape: (batch_size,)
            dy[np.arange(batch_size), batch_targets] -= 1  # Subtract 1 for the true class

            # Calculate gradients
            dWhy += np.dot(dy.T, ht)  # Shape: (output_size, hidden_size)
            dby += np.sum(dy, axis=0, keepdims=True)  # Shape: (output_size,)

            dh = np.dot(dy, self.Why) + dh_next  # Shape: (batch_size, hidden_size)
            da = (1 - ht ** 2) * dh  # Derivative of tanh

            dWxh += np.dot(da.T, xt)  # Shape: (hidden_size, input_size)
            dWhh += np.dot(da.T, hprev)  # Shape: (hidden_size, hidden_size)
            dbh += np.sum(da, axis=0, keepdims=True)  # Shape: (hidden_size,)

            dh_next = np.dot(da, self.Whh)  # Shape: (batch_size, hidden_size)

        self.grads = {
            'Whh': dWhh,
            'Wxh': dWxh,
            'bh': dbh,
            'by': dby,
            'Why': dWhy,
        }
        return self.grads

    def train_step(self, inputs, targets):
        # inputs and targets are now batches
        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs, targets)
        self.backward(targets)
        lr = 0.1  # Learning rate
        self.update_parameters(lr)
        return loss

    def compute_loss(self, outputs, targets):
        loss = 0.0
        for y_hat, target in zip(outputs, targets):
            exp_logits = np.exp(y_hat - np.max(y_hat))
            probs = exp_logits / np.sum(exp_logits)
            loss += -np.log(probs[target, 0] + 1e-8)
        return loss / len(outputs)

    def update_parameters(self, lr=0.01):
        self.Whh += -lr * self.grads['Whh']
        self.Wxh += -lr * self.grads['Wxh']
        self.bh += -lr * self.grads['bh']

        self.by += -lr * self.grads['by']
        self.Why += -lr * self.grads['Why']