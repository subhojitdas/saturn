import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def dedup_list(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def generate_training_data(tokens, window_size=2):
    training_data = []
    for centre_idx, centre_word in enumerate(tokens):
        for context_idx in range(-window_size, window_size + 1):
            idx = centre_idx + context_idx
            if idx < 0 or idx >= len(tokens) or context_idx == 0:
                continue
            training_data.append((centre_word, tokens[idx]))
    return training_data

def one_hot_encoding(word, vocab_size, stoi):
    vec = np.zeros(vocab_size)
    vec[stoi[word]] = 1
    return vec

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum(axis=0)

#Visualization with t-SNE
def visualize_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=1000)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        x, y = reduced[i, :]
        plt.scatter(x, y)
        plt.annotate(label, (x + 0.01, y + 0.01), fontsize=12)
    plt.grid(True)
    plt.show()

def get_embedding(word, stoi, embeddings):
    return embeddings[stoi[word]]

tiny_shakespeare = 'dataset/tiny_shakespeare.txt'
with open(tiny_shakespeare) as f:
    corpus = f.read()

tokens = corpus.lower().split()
vocab = dedup_list(tokens)
vocab_size = len(vocab)
stoi = {w:i for i,w in enumerate(vocab)}
itos = {i:w for w,i in stoi.items()}

#hyper parameters
window_size = 2
embedding_dim = 18
epochs = 10000
learning_rate = 1e-2

#parameters
W1 = np.random.randn(vocab_size, embedding_dim)
W2 = np.random.randn(embedding_dim, vocab_size)

training_data = generate_training_data(tokens, window_size=window_size)

for epoch in range(epochs):
    loss = 0.0
    for center_word, context_word in training_data:
        x = one_hot_encoding(center_word).reshape(-1, 1)
        y_true = one_hot_encoding(context_word).reshape(-1, 1)
        #forward pass
        h = W1.T @ x
        logits = W2.T @ h
        y_pred = softmax(logits)

        loss -= np.sum(y_true * np.log(y_pred))
        #backward pass
        e = y_pred - y_true
        dW2 = np.dot(h, e.T)
        dW1 = np.dot(x, np.dot(W2, e).T)

        # Update weights
        W1 -= learning_rate * dW1
        W2 -= learning_rate * dW2
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")