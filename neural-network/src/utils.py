"""
Utility functions for neural network: learning rate schedules and visualization.
"""

import numpy as np


def step_decay(initial_lr, epoch, drop_rate=0.5, epochs_drop=10):
    """Step decay: lr = initial_lr * drop_rate^(epoch // epochs_drop)"""
    return initial_lr * (drop_rate ** (epoch // epochs_drop))


def exponential_decay(initial_lr, epoch, decay_rate=0.1):
    """Exponential decay: lr = initial_lr * exp(-decay_rate * epoch)"""
    return initial_lr * np.exp(-decay_rate * epoch)


def plot_losses(history, save_path=None):
    """Plot training and validation loss curves."""
    try:
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(history['loss']) + 1)
        
        plt.figure(figsize=(10, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['loss'], 'b-', label='Training Loss')
        if 'val_loss' in history and history['val_loss']:
            plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        if 'accuracy' in history and history['accuracy']:
            plt.subplot(1, 2, 2)
            plt.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
            if 'val_accuracy' in history and history['val_accuracy']:
                plt.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("matplotlib not installed. Cannot plot.")


def plot_decision_boundary(model, X, y, save_path=None):
    """Plot decision boundary for 2D classification problems."""
    try:
        import matplotlib.pyplot as plt
        
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                           np.arange(y_min, y_max, 0.02))
        
        # Predict for each point in mesh
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        if Z.ndim > 1 and Z.shape[1] > 1:
            Z = np.argmax(Z, axis=1)
        else:
            Z = (Z > 0.5).astype(int).ravel()
        
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("matplotlib not installed. Cannot plot.")
