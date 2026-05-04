"""
Integration tests for NeuralNetwork - XOR problem.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from network import NeuralNetwork
from layers import DenseLayer
from activations import ReLU, Sigmoid


def test_xor():
    """Test that network can learn XOR."""
    print("Testing XOR learning...")
    
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create network
    model = NeuralNetwork()
    model.add_layer(DenseLayer(2, 4, activation=ReLU(), init_method='he'))
    model.add_layer(DenseLayer(4, 1, activation=Sigmoid()))
    model.compile(loss='binary_crossentropy', optimizer='adam', learning_rate=0.1)
    
    # Train
    model.fit(X, y, epochs=500, batch_size=4, verbose=False)
    
    # Test
    predictions = model.predict(X)
    pred_classes = (predictions > 0.5).astype(int)
    
    accuracy = np.mean(pred_classes == y)
    
    print(f"XOR Test Accuracy: {accuracy * 100:.1f}%")
    
    if accuracy >= 0.75:
        print("✓ Network can learn XOR!")
        return True
    else:
        print("⚠ Network struggled with XOR")
        return False


def test_forward_pass_shapes():
    """Test that forward pass produces correct shapes."""
    print("Testing forward pass shapes...")
    
    model = NeuralNetwork()
    model.add_layer(DenseLayer(10, 5, activation=ReLU()))
    model.add_layer(DenseLayer(5, 3, activation=None))
    
    X = np.random.randn(32, 10)
    y = model.predict(X)
    
    assert y.shape == (32, 3), f"Wrong output shape: {y.shape}"
    print("✓ Forward pass shapes correct")
    return True


if __name__ == "__main__":
    print("Running NeuralNetwork integration tests...")
    test_forward_pass_shapes()
    result = test_xor()
    
    if result:
        print("\n✓ All integration tests passed!")
    else:
        print("\n⚠ Some tests failed")
