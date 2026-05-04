"""
Tests for loss functions.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy


def test_mse():
    """Test Mean Squared Error."""
    mse = MSE()
    
    y_true = np.array([[1, 2], [3, 4]])
    y_pred = np.array([[1, 2], [3, 4]])
    
    loss = mse.forward(y_true, y_pred)
    assert np.allclose(loss, 0), f"MSE should be 0 for perfect predictions"
    
    # Test gradient
    y_true = np.array([[0, 0]])
    y_pred = np.array([[1, 1]])
    mse.forward(y_true, y_pred)
    grad = mse.backward()
    
    # Gradient should be -2(y_true - y_pred) / n = -2(0-1)/2 = 1
    expected_grad = np.array([[1, 1]])
    assert np.allclose(grad, expected_grad), f"MSE gradient wrong: {grad}"
    
    print("✓ MSE tests passed")


def test_binary_crossentropy():
    """Test Binary Cross-Entropy."""
    bce = BinaryCrossEntropy()
    
    # Perfect predictions
    y_true = np.array([[0], [1]])
    y_pred = np.array([[0.0001], [0.9999]])  # Near perfect
    loss = bce.forward(y_true, y_pred)
    assert loss < 0.1, f"BCE should be low for good predictions: {loss}"
    
    print("✓ Binary Cross-Entropy tests passed")


def test_categorical_crossentropy():
    """Test Categorical Cross-Entropy."""
    cce = CategoricalCrossEntropy()
    
    # Perfect predictions
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred = np.array([[0.9999, 0.0001, 0.0001], [0.0001, 0.9999, 0.0001]])
    loss = cce.forward(y_true, y_pred)
    assert loss < 0.1, f"CCE should be low for good predictions: {loss}"
    
    print("✓ Categorical Cross-Entropy tests passed")


if __name__ == "__main__":
    print("Running loss function tests...")
    test_mse()
    test_binary_crossentropy()
    test_categorical_crossentropy()
    print("\nAll loss tests passed!")
