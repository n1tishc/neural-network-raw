"""
Neural Network Playground - Interactive Webapp
Visualize and experiment with neural network features, training dynamics,
and decision boundaries.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from network import NeuralNetwork
from layers import DenseLayer
from activations import Sigmoid, Tanh, ReLU, LeakyReLU, Softmax, Linear
from losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy
from optimizers import SGD, SGDMomentum, RMSprop, Adam
from utils import generate_spiral_data, generate_xor_data, normalize_data
import metrics

st.set_page_config(page_title="NN Playground", page_icon="🧠", layout="wide")

st.title("🧠 Neural Network Playground")
st.markdown("Experiment with neural network features and visualize their impact.")

# Sidebar configuration
st.sidebar.header("Configuration")

# Dataset selection
dataset = st.sidebar.selectbox(
    "Dataset",
    ["XOR", "Spiral (2 classes)", "Spiral (3 classes)", "Regression"]
)

# Network architecture
st.sidebar.subheader("Network Architecture")
num_hidden = st.sidebar.slider("Hidden Layers", 1, 5, 2)
layers_config = []
for i in range(num_hidden):
    size = st.sidebar.slider(f"Layer {i+1} size", 2, 128, 16, key=f"layer_{i}")
    activation = st.sidebar.selectbox(
        f"Layer {i+1} activation",
        ["ReLU", "Leaky ReLU", "Sigmoid", "Tanh"],
        key=f"act_{i}"
    )
    layers_config.append((size, activation))

output_activation = st.sidebar.selectbox(
    "Output Activation",
    ["Sigmoid", "Softmax", "Linear"],
    key="output_act"
)

# Training parameters
st.sidebar.subheader("Training Parameters")
optimizer_name = st.sidebar.selectbox("Optimizer", ["SGD", "SGD+Momentum", "RMSprop", "Adam"])
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 1.0, 0.01, format="%.4f")
epochs = st.sidebar.slider("Epochs", 10, 1000, 100)
batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32, 64, 128], index=2)

# Loss function
if dataset == "Regression":
    loss_name = "MSE"
else:
    loss_name = st.sidebar.selectbox("Loss", ["Binary Cross-Entropy", "Categorical Cross-Entropy"])

# Main content area
col1, col2 = st.columns([1, 1])

def get_activation(name):
    """Map activation name to class."""
    mapping = {
        "ReLU": ReLU(),
        "Leaky ReLU": LeakyReLU(),
        "Sigmoid": Sigmoid(),
        "Tanh": Tanh(),
        "Softmax": Softmax(),
        "Linear": Linear()
    }
    return mapping[name]

def get_optimizer(name, lr):
    """Map optimizer name to instance."""
    if name == "SGD":
        return SGD(learning_rate=lr)
    elif name == "SGD+Momentum":
        return SGDMomentum(learning_rate=lr, beta=0.9)
    elif name == "RMSprop":
        return RMSprop(learning_rate=lr, beta=0.9)
    else:
        return Adam(learning_rate=lr, beta1=0.9, beta2=0.999)

def get_loss(name):
    """Map loss name to instance."""
    mapping = {
        "MSE": MSE(),
        "Binary Cross-Entropy": BinaryCrossEntropy(),
        "Categorical Cross-Entropy": CategoricalCrossEntropy()
    }
    return mapping[name]

def generate_dataset(name):
    """Generate selected dataset."""
    if name == "XOR":
        X, y = generate_xor_data(200)
        y = y.reshape(-1, 1)
        return X, y, 2, 1
    elif "Spiral" in name:
        n_classes = 3 if "3" in name else 2
        X, y = generate_spiral_data(n_samples=300, n_classes=n_classes)
        # One-hot encode for multi-class
        if n_classes > 2:
            y_onehot = np.zeros((y.shape[0], n_classes))
            y_onehot[np.arange(y.shape[0]), y] = 1
            y = y_onehot
        else:
            y = y.reshape(-1, 1)
        return X, y, 2, n_classes
    else:  # Regression
        np.random.seed(42)
        X = np.linspace(-5, 5, 200).reshape(-1, 1)
        y = np.sin(X) + 0.1 * np.random.randn(*X.shape)
        return X, y, 1, 1

with col1:
    st.subheader("Training")
    
    if st.button("🚀 Train Network", type="primary"):
        # Generate data
        X, y, input_dim, output_dim = generate_dataset(dataset)
        
        # Normalize
        X = normalize_data(X)
        
        # Build network
        nn = NeuralNetwork()
        prev_dim = input_dim
        
        for size, act_name in layers_config:
            nn.add_layer(DenseLayer(prev_dim, size))
            nn.add_layer(get_activation(act_name))
            prev_dim = size
        
        nn.add_layer(DenseLayer(prev_dim, output_dim))
        nn.add_layer(get_activation(output_activation))
        
        # Compile
        optimizer = get_optimizer(optimizer_name, learning_rate)
        loss_fn = get_loss(loss_name)
        nn.compile(loss=loss_fn, optimizer=optimizer)
        
        # Train with progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_chart = st.line_chart()
        
        # Custom training loop for progress tracking
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            for batch in range(n_batches):
                start = batch * batch_size
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                loss = nn.train_step(X_batch, y_batch)
                epoch_loss += loss
            
            avg_loss = epoch_loss / n_batches
            
            # Update progress
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            loss_chart.add_rows([avg_loss])
        
        st.success("Training complete!")
        
        # Store in session state
        st.session_state['nn'] = nn
        st.session_state['X'] = X
        st.session_state['y'] = y
        st.session_state['dataset'] = dataset

with col2:
    st.subheader("Results")
    
    if 'nn' in st.session_state:
        nn = st.session_state['nn']
        X = st.session_state['X']
        y = st.session_state['y']
        
        # Predictions
        y_pred = nn.predict(X)
        
        # Metrics
        if dataset == "Regression":
            mse = np.mean((y - y_pred) ** 2)
            st.metric("MSE", f"{mse:.4f}")
        elif "Spiral" in dataset and "3" in dataset:
            y_true_labels = np.argmax(y, axis=1)
            y_pred_labels = np.argmax(y_pred, axis=1)
            acc = np.mean(y_true_labels == y_pred_labels)
            st.metric("Accuracy", f"{acc*100:.2f}%")
        else:
            y_true = y.flatten()
            y_pred_labels = (y_pred.flatten() > 0.5).astype(int)
            acc = np.mean(y_true == y_pred_labels)
            st.metric("Accuracy", f"{acc*100:.2f}%")
        
        # Visualize decision boundary for 2D data
        if X.shape[1] == 2 and dataset != "Regression":
            st.subheader("Decision Boundary")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create grid
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 100),
                np.linspace(y_min, y_max, 100)
            )
            
            # Predict on grid
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            grid_points = normalize_data(grid_points)
            Z = nn.predict(grid_points)
            
            # Reshape for contour
            if Z.shape[1] == 1:  # Binary
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
            else:  # Multi-class
                Z = np.argmax(Z, axis=1).reshape(xx.shape)
                ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
            
            # Plot data points
            if "Spiral" in dataset and "3" in dataset:
                y_labels = np.argmax(y, axis=1)
                scatter = ax.scatter(X[:, 0], X[:, 1], c=y_labels, cmap='viridis', edgecolors='k')
            else:
                ax.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='RdYlBu', edgecolors='k')
            
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_title("Decision Boundary")
            
            st.pyplot(fig)
        
        # Weight distribution
        st.subheader("Weight Statistics")
        for i, layer in enumerate(nn.layers):
            if hasattr(layer, 'weights'):
                w = layer.weights.flatten()
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric(f"Layer {i} weights", f"{len(w)}")
                with col_b:
                    st.metric("Mean", f"{w.mean():.4f}")
                with col_c:
                    st.metric("Std", f"{w.std():.4f}")
    else:
        st.info("Train a network to see results here.")

# Footer
st.markdown("---")
st.markdown("**Neural Network from Scratch** - Adjust parameters in the sidebar and train to see the impact!")
