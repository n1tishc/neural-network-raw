"""
Tests for DenseLayer.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from layers import DenseLayer
from activations import ReLU


def test_dense_layer_forward():
    """Test DenseLayer forward pass."""
    layer = DenseLayer(3, 2, activation=None, init_method='zeros')
    
    X = np.array([[1, 2, 3]])
    y = layer.forward(X)
    
    # With zero weights and biases, output should be zero
    assert y.shape == (1, 2), f"Wrong output shape: {y.shape}"
    assert np.allclose(y, 0), f"Zero init failed"
    
    print("✓ DenseLayer forward test passed")


def test_dense_layer_forward_with_activation():
    """Test DenseLayer forward with activation."""
    layer = DenseLayer(2, 3, activation=ReLU(), init_method='random')
    
    X = np.array([[1, 1]])
    y = layer.forward(X)
    
    # With ones weights and no bias: output = [2, 2, 2] after ReLU
    assert y.shape == (1, 3), f"Wrong output shape: {y.shape}"
    
    print("✓ DenseLayer forward with activation test passed")


def test_dense_layer_backward():
    """Test DenseLayer backward pass shapes."""
    layer = DenseLayer(3, 2, activation=ReLU(), init_method='random')
    
    X = np.random.randn(4, 3)
    y = layer.forward(X)
    
    # Backward pass
    dA = np.random.randn(4, 2)
    dX = layer.backward(dA)
    
    # Check gradient shapes
    assert dX.shape == X.shape, f"dX shape wrong: {dX.shape}"
    assert layer.gradients['weights'].shape == (3, 2), f"dW shape wrong"
    assert layer.gradients['biases'].shape == (1, 2), f"db shape wrong"
    
    print("✓ DenseLayer backward test passed")


def test_weight_initialization():
    """Test different weight initialization methods."""
    for method in ['zeros', 'random', 'xavier', 'he']:
        layer = DenseLayer(10, 5, activation=None, init_method=method)
        assert layer.weights.shape == (10, 5), f"{method} init shape wrong"
        assert layer.biases.shape == (1, 5), f"{method} bias shape wrong"
    
    print("✓ Weight initialization tests passed")


if __name__ == "__main__":
    print("Running DenseLayer tests...")
    test_dense_layer_forward()
    test_dense_layer_forward_with_activation()
    test_dense_layer_backward()
    test_weight_initialization()
    print("\nAll layer tests passed!")
