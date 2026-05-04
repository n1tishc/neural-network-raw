"""
Tests for activation functions.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from activations import Sigmoid, Tanh, ReLU, LeakyReLU, Softmax, Linear


def test_sigmoid():
    """Test Sigmoid activation."""
    sigmoid = Sigmoid()
    
    # Test forward
    x = np.array([0, 1, -1])
    y = sigmoid.forward(x)
    expected = 1 / (1 + np.exp(-x))
    assert np.allclose(y, expected), f"Sigmoid forward failed: {y} != {expected}"
    
    # Test derivative
    d = sigmoid.derivative()
    expected_d = y * (1 - y)
    assert np.allclose(d, expected_d), f"Sigmoid derivative failed"
    
    print("✓ Sigmoid tests passed")


def test_tanh():
    """Test Tanh activation."""
    tanh = Tanh()
    
    x = np.array([0, 1, -1])
    y = tanh.forward(x)
    expected = np.tanh(x)
    assert np.allclose(y, expected), f"Tanh forward failed"
    
    d = tanh.derivative()
    expected_d = 1 - y ** 2
    assert np.allclose(d, expected_d), f"Tanh derivative failed"
    
    print("✓ Tanh tests passed")


def test_relu():
    """Test ReLU activation."""
    relu = ReLU()
    
    x = np.array([-1, 0, 1, 2])
    y = relu.forward(x)
    expected = np.maximum(0, x)
    assert np.allclose(y, expected), f"ReLU forward failed"
    
    d = relu.derivative()
    expected_d = (x > 0).astype(float)
    assert np.allclose(d, expected_d), f"ReLU derivative failed"
    
    print("✓ ReLU tests passed")


def test_leaky_relu():
    """Test Leaky ReLU activation."""
    leaky = LeakyReLU(alpha=0.01)
    
    x = np.array([-1, 0, 1])
    y = leaky.forward(x)
    expected = np.where(x > 0, x, 0.01 * x)
    assert np.allclose(y, expected), f"LeakyReLU forward failed"
    
    print("✓ Leaky ReLU tests passed")


def test_softmax():
    """Test Softmax activation."""
    softmax = Softmax()
    
    x = np.array([[1, 2, 3]])
    y = softmax.forward(x)
    
    # Check that outputs sum to 1
    assert np.allclose(np.sum(y), 1.0), f"Softmax outputs don't sum to 1"
    
    # Check numerical stability
    x_large = np.array([[1000, 1001, 1002]])
    y_large = softmax.forward(x_large)
    assert np.allclose(np.sum(y_large), 1.0), f"Softmax numerical stability failed"
    
    print("✓ Softmax tests passed")


def test_linear():
    """Test Linear activation."""
    linear = Linear()
    
    x = np.array([1, 2, 3])
    y = linear.forward(x)
    assert np.allclose(y, x), f"Linear forward failed"
    
    d = linear.derivative()
    assert np.allclose(d, np.ones_like(x)), f"Linear derivative failed"
    
    print("✓ Linear tests passed")


if __name__ == "__main__":
    print("Running activation function tests...")
    test_sigmoid()
    test_tanh()
    test_relu()
    test_leaky_relu()
    test_softmax()
    test_linear()
    print("\nAll activation tests passed!")
