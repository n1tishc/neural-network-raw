# Neural Network From Scratch

A complete feedforward neural network implementation using Python and NumPy, with no deep learning frameworks (PyTorch, TensorFlow, etc.).

## Features

- **Fully connected (dense) feedforward neural network**
- **Multiple hidden layers** with configurable architecture
- **Activation functions**: Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax, Linear
- **Loss functions**: MSE, Binary Cross-Entropy, Categorical Cross-Entropy
- **Optimizers**: SGD, SGD with Momentum, RMSprop, Adam
- **Mini-batch training** with validation support
- **Metrics**: Accuracy, Precision, Recall, F1-Score, R², Confusion Matrix
- **Serialization**: Save/load model weights
- **Visualization**: Loss curves and decision boundaries

## Project Structure

```
neural-network/
├── src/
│   ├── __init__.py
│   ├── network.py          # Main NeuralNetwork class
│   ├── layers.py           # DenseLayer implementation
│   ├── activations.py      # Activation functions
│   ├── losses.py           # Loss functions
│   ├── optimizers.py       # Optimization algorithms
│   ├── metrics.py          # Evaluation metrics
│   └── utils.py            # Learning rate schedules & visualization
├── tests/
│   ├── __init__.py
│   ├── test_layers.py
│   ├── test_activations.py
│   ├── test_losses.py
│   └── test_network.py
├── examples/
│   ├── xor_demo.py         # XOR problem (non-linear)
│   ├── mnist_classifier.py # MNIST digit classification
│   └── spiral_classifier.py # Spiral dataset visualization
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### XOR Problem (Learn non-linear patterns)

```bash
cd examples
python xor_demo.py
```

### MNIST Digit Classification

```bash
cd examples
python mnist_classifier.py
```

### Spiral Classification with Visualization

```bash
cd examples
python spiral_classifier.py
```

## Usage Example

```python
from src.network import NeuralNetwork
from src.layers import DenseLayer
from src.activations import ReLU, Softmax

# Create network
model = NeuralNetwork()
model.add_layer(DenseLayer(784, 128, activation=ReLU()))
model.add_layer(DenseLayer(128, 64, activation=ReLU()))
model.add_layer(DenseLayer(64, 10, activation=Softmax()))

# Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', learning_rate=0.001)

# Train
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Evaluate
results = model.evaluate(X_test, y_test)
print(f"Accuracy: {results['accuracy'] * 100:.2f}%")

# Save weights
model.save_weights('model_weights.npy')
```

## Running Tests

```bash
cd tests
python test_activations.py
python test_layers.py
python test_losses.py
python test_network.py
```

## Implementation Details

- **Weight Initialization**: He initialization for ReLU, Xavier for others
- **Gradient Method**: Backpropagation with chain rule
- **Numerical Stability**: Softmax and cross-entropy with clipping
- **Mini-batch Training**: Shuffled each epoch

## Success Criteria

- ✓ All activation functions implemented with derivatives
- ✓ All loss functions implemented with gradients
- ✓ DenseLayer forward/backward working
- ✓ Network learns XOR (non-linear problem)
- ✓ Network achieves >90% on MNIST
- ✓ Clean, documented code
- ✓ Working examples with visualizations

## License

MIT License
