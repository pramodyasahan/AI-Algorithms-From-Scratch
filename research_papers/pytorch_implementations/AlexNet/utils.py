import numpy as np


def cross_entropy_loss(y_true, y_pred):
    """
    Compute the cross-entropy loss.
    """
    y_pred = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def accuracy(y_true, y_pred):
    """
    Compute the accuracy.
    """
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))


def softmax(x):
    """
    Compute the softmax.
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
