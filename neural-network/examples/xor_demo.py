"""
XOR Problem Demo - Classic non-linear classification problem.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from network import NeuralNetwork
from layers import DenseLayer
from activations import ReLU, Tanh, Sigmoid


def main():
    print("=" * 50)
    print("XOR Problem Demo")
    print("=" * 50)
    
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    print("\nXOR Dataset:")
    print("Inputs: [0,0], [0,1], [1,0], [1,1]")
    print("Labels:  0  ,  1  ,  1  ,  0")
    
    # Create neural network
    # Architecture: 2 inputs -> 4 hidden (Tanh) -> 1 output (Sigmoid)
    # Using Tanh instead of ReLU to avoid dead neurons
    model = NeuralNetwork()
    model.add_layer(DenseLayer(2, 4, activation=Tanh(), init_method='xavier'))
    model.add_layer(DenseLayer(4, 1, activation=Sigmoid()))
    model.compile(loss='binary_crossentropy', optimizer='adam', learning_rate=0.5)
    
    print("\nNetwork Architecture:")
    print("Input: 2 neurons")
    print("Hidden: 2 neurons (ReLU)")
    print("Output: 1 neuron (Sigmoid)")
    
    # Train
    print("\nTraining...")
    history = model.fit(X, y, epochs=5000, batch_size=4, verbose=False)
    
    # Evaluate
    print("\nTraining Results:")
    predictions = model.predict(X)
    
    for i in range(len(X)):
        pred = predictions[i][0]
        pred_class = 1 if pred > 0.5 else 0
        print(f"Input: {X[i]} -> Predicted: {pred:.4f} ({pred_class}) - True: {y[i][0]}")
    
    final_loss = history['loss'][-1]
    print(f"\nFinal Loss: {final_loss:.6f}")
    
    # Test accuracy
    pred_classes = (predictions > 0.5).astype(int)
    accuracy = np.mean(pred_classes == y)
    print(f"Accuracy: {accuracy * 100:.1f}%")
    
    if accuracy == 1.0:
        print("\n✓ Network successfully learned the XOR function!")
    else:
        print("\n⚠ Network may need more training or tuning.")


if __name__ == "__main__":
    main()
