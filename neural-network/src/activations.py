"""
Activation functions with forward pass and derivatives.
"""

import numpy as np


class Sigmoid:
    """Sigmoid activation: σ(x) = 1 / (1 + e^(-x))"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """Forward pass."""
        x = np.array(x)
        self.cache = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self.cache
    
    def derivative(self, x=None):
        """Derivative: σ'(x) = σ(x) * (1 - σ(x))"""
        s = self.cache if self.cache is not None else self.forward(x)
        return s * (1 - s)


class Tanh:
    """Tanh activation: tanh(x)"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """Forward pass."""
        x = np.array(x)
        self.cache = np.tanh(x)
        return self.cache
    
    def derivative(self, x=None):
        """Derivative: tanh'(x) = 1 - tanh²(x)"""
        t = self.cache if self.cache is not None else np.tanh(np.array(x))
        return 1 - t ** 2


class ReLU:
    """ReLU activation: max(0, x)"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """Forward pass."""
        x = np.array(x)
        self.cache = np.maximum(0, x)
        return self.cache
    
    def derivative(self, x=None):
        """Derivative: 1 if x > 0 else 0"""
        arr = self.cache if self.cache is not None else np.array(x)
        return (arr > 0).astype(float)


class LeakyReLU:
    """Leaky ReLU: max(αx, x) where α=0.01"""
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.cache = None
    
    def forward(self, x):
        """Forward pass."""
        x = np.array(x)
        self.cache = np.where(x > 0, x, self.alpha * x)
        return self.cache
    
    def derivative(self, x=None):
        """Derivative: 1 if x > 0 else α"""
        arr = self.cache if self.cache is not None else np.array(x)
        return np.where(arr > 0, 1.0, self.alpha)


class Softmax:
    """Softmax activation: e^x_i / Σe^x_j (for output layer)"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """Forward pass with numerical stability."""
        x = np.array(x)
        # Numerical stability: subtract max
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        self.cache = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.cache
    
    def derivative(self, x=None):
        """Derivative: Softmax derivative is handled in cross-entropy loss.
        Return Jacobian approximation for standalone use."""
        s = self.cache if self.cache is not None else self.forward(x)
        # For simplicity, return diagonal of Jacobian
        return s * (1 - s)


class Linear:
    """Linear activation (identity): f(x) = x"""
    
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """Forward pass."""
        x = np.array(x)
        self.cache = x
        return self.cache
    
    def derivative(self, x=None):
        """Derivative: 1"""
        shape = self.cache.shape if self.cache is not None else np.array(x).shape
        return np.ones(shape)
