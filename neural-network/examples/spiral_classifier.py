"""
Spiral Classification Example - 3-class spiral dataset with visualization.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from network import NeuralNetwork
from layers import DenseLayer
from activations import ReLU, Softmax
from utils import plot_decision_boundary


def generate_spiral_data(n_samples=300, noise=0.2):
    """Generate 3-class spiral dataset."""
    n_per_class = n_samples // 3
    X = np.zeros((n_samples, 2))
    y = np.zeros((n_samples, 3))  # One-hot
    
    for class_idx in range(3):
        # Spiral parameters
        ix = range(class_idx * n_per_class, (class_idx + 1) * n_per_class)
        r = np.linspace(0.0, 1, n_per_class)  # Radius
        t = np.linspace(class_idx * 4, (class_idx + 1) * 4, n_per_class) + np.random.randn(n_per_class) * noise
        
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix, class_idx] = 1
    
    return X, y


def main():
    print("=" * 50)
    print("Spiral Classification Demo")
    print("=" * 50)
    
    # Generate spiral data
    print("\nGenerating 3-class spiral dataset...")
    X, y = generate_spiral_data(n_samples=300, noise=0.2)
    
    print(f"Dataset: {len(X)} samples, 3 classes")
    print(f"Features: 2 (x, y coordinates)")
    
    # Split into train/validation
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create neural network
    # Architecture: 2 -> 16 (ReLU) -> 8 (ReLU) -> 3 (Softmax)
    model = NeuralNetwork()
    model.add_layer(DenseLayer(2, 16, activation=ReLU(), init_method='he'))
    model.add_layer(DenseLayer(16, 8, activation=ReLU(), init_method='he'))
    model.add_layer(DenseLayer(8, 3, activation=Softmax()))
    model.compile(loss='categorical_crossentropy', optimizer='adam', learning_rate=0.01)
    
    print("\nNetwork Architecture:")
    print("Input: 2 neurons (x, y)")
    print("Hidden 1: 16 neurons (ReLU)")
    print("Hidden 2: 8 neurons (ReLU)")
    print("Output: 3 neurons (Softmax)")
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=False
    )
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("Final Evaluation")
    print("=" * 50)
    
    train_results = model.evaluate(X_train, y_train)
    val_results = model.evaluate(X_val, y_val)
    
    print(f"Training Loss: {train_results['loss']:.4f}, Accuracy: {train_results['accuracy'] * 100:.2f}%")
    print(f"Validation Loss: {val_results['loss']:.4f}, Accuracy: {val_results['accuracy'] * 100:.2f}%")
    
    # Plot decision boundary
    print("\nPlotting decision boundary...")
    try:
        plot_decision_boundary(model, X, np.argmax(y, axis=1), save_path='spiral_boundary.png')
    except Exception as e:
        print(f"Could not plot: {e}")
    
    print("\n✓ Spiral classification complete!")


if __name__ == "__main__":
    main()
