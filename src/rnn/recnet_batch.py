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
        # targets is of shape (seq_len, batch_size)
        batch_size = targets.shape[1]
        seq_len = targets.shape[0]

        dWhh = np.zeros_like(self.Whh)
        dWxh = np.zeros_like(self.Wxh)
        dbh = np.zeros_like(self.bh)
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros((batch_size, self.hidden_size))

        for t in reversed(range(seq_len)):
            xt = self.inputs[t].squeeze(-1)
            ht = self.hidden[t]
            hprev = self.hidden[t - 1] if t > 0 else np.zeros_like(ht)
            yhat = self.outputs[t]

            exp_logits = np.exp(yhat - np.max(yhat, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            dy = probs
            batch_targets = targets[t, :]
            dy[np.arange(batch_size), batch_targets] -= 1

            dWhy += dy.T @ ht
            dby += np.sum(dy, axis=0, keepdims=True).T

            dh = dy @ self.Why + dh_next
            da = (1 - ht ** 2) * dh

            dWxh += da.T @ xt
            dWhh += da.T @ hprev
            dbh += np.sum(da, axis=0, keepdims=True).T

            dh_next = da @ self.Whh

        self.grads = {
            'Whh': dWhh,
            'Wxh': dWxh,
            'bh': dbh,
            'by': dby,
            'Why': dWhy,
        }
        return self.grads

    def train_step(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs, targets)
        self.backward(targets)
        lr = 0.1
        self.update_parameters(lr)
        return loss

    def compute_loss(self, outputs, targets):
        # outputs => (seq_len, batch_size, output_class)
        # target => (seq_len, batch_size) and contains number of the char index i.e. THE class
        seq_len, batch_size, output_class = outputs.shape
        outputs = outputs.reshape(seq_len * batch_size, output_class)
        targets = targets.reshape(seq_len * batch_size)

        exp_logits = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        correct_class_prob = probs[np.arange(seq_len*batch_size), targets]
        loss = -np.mean(np.log(correct_class_prob + 1e-8))
        return loss

    def update_parameters(self, lr=0.01):
        self.Whh += -lr * self.grads['Whh']
        self.Wxh += -lr * self.grads['Wxh']
        self.bh += -lr * self.grads['bh']

        self.by += -lr * self.grads['by']
        self.Why += -lr * self.grads['Why']