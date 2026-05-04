"""
MNIST Digit Classification Example.
Target: >90% accuracy on test set.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from network import NeuralNetwork
from layers import DenseLayer
from activations import ReLU, Softmax


def load_mnist():
    """Load MNIST dataset (simplified - using sklearn or manual download)."""
    try:
        from sklearn.datasets import fetch_openml
        print("Loading MNIST from sklearn...")
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        return X.values if hasattr(X, 'values') else X, y
    except ImportError:
        print("sklearn not available. Using synthetic data for demo...")
        # Generate synthetic data for demonstration
        np.random.seed(42)
        n_samples = 1000
        X = np.random.randn(n_samples, 784) * 0.5 + 0.5
        y = np.random.randint(0, 10, n_samples)
        return X, y


def main():
    print("=" * 50)
    print("MNIST Digit Classification")
    print("=" * 50)
    
    # Load data
    X, y = load_mnist()
    
    # Normalize
    X = X / 255.0
    
    # Convert labels to one-hot
    n_classes = 10
    y_onehot = np.zeros((len(y), n_classes))
    y_onehot[np.arange(len(y)), y] = 1
    
    # Split into train/validation
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y_onehot[:split_idx], y_onehot[split_idx:]
    
    print(f"\nDataset: {len(X)} samples")
    print(f"Train: {len(X_train)}, Validation: {len(X_val)}")
    print(f"Input shape: {X_train.shape[1]} (28x28 pixels)")
    print(f"Output classes: {n_classes}")
    
    # Create neural network
    # Architecture: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Softmax)
    model = NeuralNetwork()
    model.add_layer(DenseLayer(784, 128, activation=ReLU(), init_method='he'))
    model.add_layer(DenseLayer(128, 64, activation=ReLU(), init_method='he'))
    model.add_layer(DenseLayer(64, 10, activation=Softmax()))
    model.compile(loss='categorical_crossentropy', optimizer='adam', learning_rate=0.001)
    
    print("\nNetwork Architecture:")
    print("Input: 784 neurons (28x28 pixels)")
    print("Hidden 1: 128 neurons (ReLU)")
    print("Hidden 2: 64 neurons (ReLU)")
    print("Output: 10 neurons (Softmax)")
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=True
    )
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("Final Evaluation")
    print("=" * 50)
    
    val_results = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {val_results['loss']:.4f}")
    print(f"Validation Accuracy: {val_results['accuracy'] * 100:.2f}%")
    
    if val_results['accuracy'] > 0.9:
        print("\n✓ Achieved >90% accuracy target!")
    else:
        print(f"\n⚠ Target is >90%. Consider training longer or tuning hyperparameters.")
    
    # Save weights
    model.save_weights('mnist_weights.npy')


if __name__ == "__main__":
    main()
