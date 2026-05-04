"""
Loss functions with forward pass and gradients.
"""

import numpy as np


class MSE:
    """Mean Squared Error: 1/n Σ(y_true - y_pred)²"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, y_true, y_pred):
        """Compute MSE loss."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        self.cache = (y_true, y_pred)
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self):
        """Gradient: dL/dy_pred = -2(y_true - y_pred) / n"""
        y_true, y_pred = self.cache
        n = y_true.size
        return -2 * (y_true - y_pred) / n


class BinaryCrossEntropy:
    """Binary Cross-Entropy: -1/n Σ[y log(ŷ) + (1-y)log(1-ŷ)]"""
    
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon
        self.cache = None
    
    def forward(self, y_true, y_pred):
        """Compute binary cross-entropy loss."""
        y_true = np.array(y_true)
        y_pred = np.clip(np.array(y_pred), self.epsilon, 1 - self.epsilon)
        self.cache = (y_true, y_pred)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(self):
        """Gradient: dL/dy_pred = -(y/ŷ - (1-y)/(1-ŷ)) / n"""
        y_true, y_pred = self.cache
        n = y_true.size
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / n


class CategoricalCrossEntropy:
    """Categorical Cross-Entropy: -Σ y_true * log(y_pred)"""
    
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon
        self.cache = None
    
    def forward(self, y_true, y_pred):
        """Compute categorical cross-entropy loss."""
        y_true = np.array(y_true)
        y_pred = np.clip(np.array(y_pred), self.epsilon, 1.0)
        self.cache = (y_true, y_pred)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))
    
    def backward(self):
        """Gradient: dL/dy_pred = -y_true / y_pred / n (combined with softmax gives simpler form)"""
        y_true, y_pred = self.cache
        n = y_true.shape[0]
        return -y_true / y_pred / n
