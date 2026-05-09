# Neural Network From Scratch - Detailed Plan

## Overview
Build a complete feedforward neural network implementation using Python and NumPy, with no deep learning frameworks (PyTorch, TensorFlow, etc.).

## Architecture Goals
- Fully connected (dense) feedforward neural network
- Support for multiple hidden layers
- Configurable activation functions
- Multiple loss functions
- Gradient-based optimization (SGD, Adam)
- Training with mini-batch support

---

## Phase 1: Foundation (Core Math & Components)

### Task 1.1: Project Structure & Setup
**Files to create:**
```
neural-network/
├── src/
│   ├── __init__.py
│   ├── network.py          # Main NeuralNetwork class
│   ├── layers.py           # Layer implementations
│   ├── activations.py      # Activation functions
│   ├── losses.py           # Loss functions
│   ├── optimizers.py       # Optimization algorithms
│   └── utils.py            # Helper functions
├── tests/
│   ├── __init__.py
│   ├── test_layers.py
│   ├── test_activations.py
│   ├── test_losses.py
│   └── test_network.py
├── examples/
│   ├── mnist_classifier.py
│   ├── regression_demo.py
│   └── spiral_classifier.py
├── requirements.txt
└── README.md
```

**Requirements:** `numpy`, `matplotlib` (for visualization)

---

### Task 1.2: Activation Functions
**File:** `src/activations.py`

Implement the following with both forward pass and derivatives:
- **Sigmoid**: σ(x) = 1/(1+e^(-x))
- **Tanh**: tanh(x)
- **ReLU**: max(0, x)
- **Leaky ReLU**: max(αx, x) where α=0.01
- **Softmax**: e^x_i / Σe^x_j (for output layer, classification)
- **Linear**: f(x) = x (no activation)

Each function should have:
- Forward pass method
- Derivative method (for backpropagation)
- Store the class design pattern (functions accessible as objects)

**Tests:** Verify derivatives numerically using finite differences.

---

### Task 1.3: Loss Functions
**File:** `src/losses.py`

Implement with forward pass and derivatives:
- **Mean Squared Error (MSE)**: 1/n Σ(y_true - y_pred)²
- **Binary Cross-Entropy**: -1/n Σ[y log(ŷ) + (1-y)log(1-ŷ)]
- **Categorical Cross-Entropy**: -Σ y_true * log(y_pred)

Each loss should return:
- Loss value (scalar)
- Gradient with respect to predictions (for backprop)

---

### Task 1.4: Layer Implementation
**File:** `src/layers.py`

#### DenseLayer (Fully Connected Layer)
**Attributes:**
- `input_size`: Number of input neurons
- `output_size`: Number of output neurons
- `weights`: Weight matrix (initialized using He/Xavier initialization)
- `biases`: Bias vector (initialized to zeros or small values)
- `activation`: Activation function object
- `cache`: Store forward pass values for backprop

**Methods:**
- `forward(X)`: Compute Z = XW + b, then A = activation(Z)
- `backward(dA)`: Compute gradients dW, db, dX using chain rule
- `update(params)`: Apply gradient updates (or delegate to optimizer)

**Weight Initialization Options:**
- Zeros (not recommended, but for comparison)
- Random normal (small values)
- Xavier/Glorot initialization
- He initialization (for ReLU)

---

## Phase 2: Network Assembly

### Task 2.1: NeuralNetwork Class
**File:** `src/network.py`

**Core class that manages layers and training:**

```python
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_fn = None
        self.optimizer = None
    
    def add_layer(self, layer):
        # Add a DenseLayer or ActivationLayer
        
    def compile(self, loss, optimizer):
        # Set loss function and optimizer
        
    def forward(self, X):
        # Pass through all layers
        
    def backward(self, y_true):
        # Backpropagate through all layers
        
    def train_step(self, X_batch, y_batch):
        # Forward, compute loss, backward, update
        
    def predict(self, X):
        # Forward pass without gradient tracking
        
    def evaluate(self, X, y):
        # Compute loss and metrics
```

---

### Task 2.2: Optimizers
**File:** `src/optimizers.py`

Implement:
- **SGD**: w = w - η * ∇w
- **SGD with Momentum**: v = βv + ∇w; w = w - ηv
- **RMSprop**: cache = β*cache + (1-β)*∇w²; w = w - η*∇w/(√cache + ε)
- **Adam**: Combines momentum + RMSprop with bias correction

Each optimizer should:
- Maintain state (velocity, cache, etc.) per parameter
- Have `update(params, grads)` method
- Support learning rate scheduling (optional for v1)

---

## Phase 3: Training Pipeline

### Task 3.1: Training Loop
**Add to NeuralNetwork class:**

```python
def fit(self, X_train, y_train, epochs, batch_size, 
        validation_data=None, verbose=True):
    # Main training loop
    # - Shuffle data each epoch
    # - Create mini-batches
    # - Run train_step for each batch
    # - Evaluate on validation set
    # - Print progress/losses
```

**Features:**
- Mini-batch gradient descent
- Epoch-based training with shuffling
- Validation set evaluation
- Loss history tracking
- Early stopping (optional)

---

### Task 3.2: Metrics & Evaluation
**File:** `src/metrics.py` (optional, can be in utils)

Implement:
- Accuracy (classification)
- Precision, Recall, F1 (classification)
- R² score (regression)
- Confusion matrix visualization

---

## Phase 4: Testing & Examples

### Task 4.1: Unit Tests
**Files:** `tests/`

Test each component:
- Activation functions (forward + backward)
- Loss functions (forward + backward)
- Layer forward/backward shapes
- Network training on simple problems
- Gradient checking (numerical verification)

---

### Task 4.2: Example 1 - XOR Problem
**File:** `examples/xor_demo.py`

Classic non-linear problem:
- 2 input neurons, 2 hidden (ReLU), 1 output (sigmoid)
- Binary classification
- Verify network learns non-linear decision boundary

---

### Task 4.3: Example 2 - MNIST Digit Classification
**File:** `examples/mnist_classifier.py`

- Load MNIST (use sklearn or manual download)
- Network: 784 → 128 (ReLU) → 64 (ReLU) → 10 (Softmax)
- Categorical cross-entropy loss
- Adam optimizer
- Target: >90% accuracy

---

### Task 4.4: Example 3 - Spiral Classification
**File:** `examples/spiral_classifier.py`

- Generate spiral dataset (3 classes)
- Visualize decision boundaries
- Show network learning process with plots

---

## Phase 5: Polish & Extensions (Optional)

### Task 5.1: Serialization
- Save/load model weights (JSON, NumPy .npy)
- Save/load full model state (optimizer state, etc.)

### Task 5.2: Regularization
- L1/L2 weight decay
- Dropout layer
- Batch normalization (advanced)

### Task 5.3: Learning Rate Scheduling
- Step decay
- Exponential decay
- Reduce on plateau

### Task 5.4: Visualization
- Plot loss curves
- Visualize weight distributions
- Activation heatmaps

---

## Implementation Order (Recommended)

1. **Start with activations.py** - Simple, testable functions
2. **Then losses.py** - Also simple, testable
3. **Then layers.py** - Depends on activations
4. **Then optimizers.py** - Can be developed in parallel
5. **Assemble in network.py** - Brings everything together
6. **Test with XOR problem** - Simplest verification
7. **Scale to MNIST** - Real-world validation

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python 3.x | Standard for ML, NumPy support |
| Matrix Library | NumPy only | "From scratch" requirement |
| Network Type | Feedforward only | Keep scope manageable |
| Gradient Method | Backpropagation | Standard, well-understood |
| Batch Strategy | Mini-batch | Balance between speed and stability |
| Weight Init | He/Xavier | Prevents vanishing/exploding gradients |

---

## Success Criteria

- [ ] All activation functions implemented with correct derivatives
- [ ] All loss functions implemented with correct gradients
- [ ] DenseLayer forward/backward working correctly
- [ ] Network can learn XOR (non-linear problem)
- [ ] Network achieves >90% on MNIST
- [ ] All unit tests passing
- [ ] Clean, documented code
- [ ] Working examples with visualizations

---

## Time Estimates (for reference)

| Phase | Tasks | Est. Time |
|-------|-------|-----------|
| Phase 1 | Foundation | 3-4 hours |
| Phase 2 | Assembly | 2-3 hours |
| Phase 3 | Training | 2-3 hours |
| Phase 4 | Examples | 3-4 hours |
| Phase 5 | Polish | 2-3 hours |
| **Total** | | **12-17 hours** |

---

## Next Steps

1. Review and approve this plan
2. Set up project structure
3. Begin with Task 1.2 (Activation Functions)
4. Or, convert this plan into Taskplane tasks for organized execution using `/orch`

Would you like me to:
- **Start implementing** immediately (begin with activations.py)?
- **Create Taskplane tasks** from this plan for structured execution?
- **Modify the plan** before proceeding?
