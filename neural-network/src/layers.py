"""
Layer implementations for neural network.
"""

import numpy as np


class DenseLayer:
    """Fully connected (dense) layer."""

    def __init__(self, input_size, output_size, activation=None, init_method='he'):
        """
        Initialize dense layer.

        Args:
            input_size: Number of input neurons
            output_size: Number of output neurons
            activation: Activation function object
            init_method: Weight initialization method ('zeros', 'random', 'xavier', 'he')
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.init_method = init_method

        # Initialize weights and biases
        self.weights = self._initialize_weights(init_method)
        self.biases = np.zeros((1, output_size))

        # Cache for backprop
        self.cache = None
        self.gradients = None

    def _initialize_weights(self, method):
        """Initialize weights using specified method."""
        if method == 'zeros':
            return np.zeros((self.input_size, self.output_size))
        elif method == 'random':
            return np.random.randn(self.input_size, self.output_size) * 0.01
        elif method == 'xavier':
            # Xavier/Glorot initialization
            limit = np.sqrt(6 / (self.input_size + self.output_size))
            return np.random.uniform(-limit, limit, (self.input_size, self.output_size))
        elif method == 'he':
            # He initialization (good for ReLU)
            return np.random.randn(self.input_size, self.output_size) * np.sqrt(2 / self.input_size)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def forward(self, X):
        """
        Forward pass: Z = XW + b, then A = activation(Z)

        Args:
            X: Input matrix of shape (batch_size, input_size)

        Returns:
            Activated output
        """
        X = np.array(X)
        # Linear transformation
        Z = np.dot(X, self.weights) + self.biases

        # Apply activation if provided
        if self.activation is not None:
            A = self.activation.forward(Z)
        else:
            A = Z

        # Cache for backprop
        self.cache = (X, Z, A)

        return A

    def backward(self, dA):
        """
        Backward pass: Compute gradients using chain rule.

        Args:
            dA: Gradient of loss w.r.t. output (batch_size, output_size)

        Returns:
            Gradient w.r.t. input
        """
        X, Z, A = self.cache
        m = X.shape[0]

        # Gradient through activation
        if self.activation is not None:
            dZ = dA * self.activation.derivative(Z)
        else:
            dZ = dA

        # Gradients for weights and biases
        dW = np.dot(X.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m

        # Gradient w.r.t. input
        dX = np.dot(dZ, self.weights.T)

        # Store gradients
        self.gradients = {'weights': dW, 'biases': db}

        return dX

    def update(self, optimizer, layer_name):
        """
        Update weights using optimizer.
        
        Args:
            optimizer: Optimizer object with update method
            layer_name: Name/key for this layer in optimizer state
        """
        self.weights = optimizer.update(f"{layer_name}_weights", self.weights, self.gradients['weights'])
        self.biases = optimizer.update(f"{layer_name}_biases", self.biases, self.gradients['biases'])