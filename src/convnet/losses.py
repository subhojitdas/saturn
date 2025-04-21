import numpy as np


def svm_loss(scores, y):
    num_examples = scores.shape[0]
    # print("num_examples =", num_examples)
    # print("np.arange(num_examples) =", np.arange(num_examples))
    corect_class_scores = scores[np.arange(num_examples), y].reshape(-1, 1)
    margin = np.maximum(0, scores - corect_class_scores + 1)
    margin[np.arange(num_examples), y] = 0
    loss = margin.sum() / num_examples
    return loss


def softmax_loss(scores, y):
    epsilon = 1e-12
    num_examples = scores.shape[0]
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_log_probs = -np.log(probs[np.arange(num_examples), y] + epsilon)
    loss = np.mean(correct_log_probs)
    return loss


def softmax_numpy(x, axis=1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)