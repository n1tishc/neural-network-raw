"""
Main NeuralNetwork class that manages layers and training.
"""

import numpy as np
from layers import DenseLayer
from activations import Linear
from losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy


class NeuralNetwork:
    """Feedforward neural network implementation."""
    
    def __init__(self):
        self.layers = []
        self.loss_fn = None
        self.optimizer = None
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    
    def add_layer(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)
    
    def compile(self, loss='mse', optimizer='adam', learning_rate=0.001):
        """
        Configure the network for training.
        
        Args:
            loss: Loss function ('mse', 'binary_crossentropy', 'categorical_crossentropy')
            optimizer: Optimizer ('sgd', 'sgd_momentum', 'rmsprop', 'adam')
            learning_rate: Learning rate for optimizer
        """
        # Set loss function
        if loss == 'mse':
            self.loss_fn = MSE()
        elif loss == 'binary_crossentropy':
            self.loss_fn = BinaryCrossEntropy()
        elif loss == 'categorical_crossentropy':
            self.loss_fn = CategoricalCrossEntropy()
        else:
            raise ValueError(f"Unknown loss function: {loss}")
        
        # Set optimizer
        if optimizer == 'sgd':
            from optimizers import SGD
            self.optimizer = SGD(learning_rate=learning_rate)
        elif optimizer == 'sgd_momentum':
            from optimizers import SGDMomentum
            self.optimizer = SGDMomentum(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            from optimizers import RMSprop
            self.optimizer = RMSprop(learning_rate=learning_rate)
        elif optimizer == 'adam':
            from optimizers import Adam
            self.optimizer = Adam(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
    
    def forward(self, X):
        """
        Forward pass through all layers.
        
        Args:
            X: Input data of shape (batch_size, input_size)
        
        Returns:
            Network output
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, y_true):
        """
        Backward pass through all layers.
        
        Args:
            y_true: True labels
        """
        # Get gradient of loss w.r.t. predictions
        dA = self.loss_fn.backward()
        
        # Backpropagate through layers in reverse order
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
    
    def train_step(self, X_batch, y_batch):
        """
        Perform one training step.
        
        Args:
            X_batch: Batch of input data
            y_batch: Batch of true labels
        
        Returns:
            Loss value for this batch
        """
        # Forward pass
        y_pred = self.forward(X_batch)
        
        # Compute loss
        loss = self.loss_fn.forward(y_batch, y_pred)
        
        # Backward pass
        self.backward(y_batch)
        
        # Update weights
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'update'):
                layer.update(self.optimizer, f"layer_{i}")
        
        return loss
    
    def predict(self, X):
        """
        Make predictions (forward pass without gradient tracking).
        
        Args:
            X: Input data
        
        Returns:
            Predictions
        """
        return self.forward(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        
        Args:
            X: Input data
            y: True labels
        
        Returns:
            Dictionary with loss and metrics
        """
        y_pred = self.predict(X)
        loss = self.loss_fn.forward(y, y_pred)
        
        results = {'loss': loss}
        
        # Compute accuracy for classification
        if isinstance(self.loss_fn, (BinaryCrossEntropy, CategoricalCrossEntropy)):
            if y_pred.shape[1] > 1:  # Multi-class
                y_pred_class = np.argmax(y_pred, axis=1)
                y_true_class = np.argmax(y, axis=1)
            else:  # Binary
                y_pred_class = (y_pred > 0.5).astype(int)
                y_true_class = y.astype(int)
            accuracy = np.mean(y_pred_class == y_true_class)
            results['accuracy'] = accuracy
        
        return results
    
    def fit(self, X_train, y_train, epochs=10, batch_size=32, 
            validation_data=None, verbose=True):
        """
        Train the network.
        
        Args:
            X_train: Training input data
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Mini-batch size
            validation_data: Tuple (X_val, y_val) for validation
            verbose: Print progress if True
        
        Returns:
            Training history
        """
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for batch in range(n_batches):
                start = batch * batch_size
                end = min(start + batch_size, n_samples)
                
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                loss = self.train_step(X_batch, y_batch)
                epoch_loss += loss
            
            # Average loss for this epoch
            epoch_loss /= n_batches
            self.history['loss'].append(epoch_loss)
            
            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                val_results = self.evaluate(X_val, y_val)
                self.history['val_loss'].append(val_results['loss'])
                if 'accuracy' in val_results:
                    self.history['val_accuracy'].append(val_results['accuracy'])
            
            # Training accuracy
            train_results = self.evaluate(X_train, y_train)
            if 'accuracy' in train_results:
                self.history['accuracy'].append(train_results['accuracy'])
            
            if verbose:
                msg = f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}"
                if validation_data is not None:
                    msg += f" - val_loss: {self.history['val_loss'][-1]:.4f}"
                if 'accuracy' in train_results:
                    msg += f" - accuracy: {train_results['accuracy']:.4f}"
                print(msg)
        
        return self.history
    
    def save_weights(self, path):
        """Save model weights to .npy file."""
        weights_dict = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                weights_dict[f"layer_{i}_weights"] = layer.weights
                weights_dict[f"layer_{i}_biases"] = layer.biases
        
        np.save(path, weights_dict)
        print(f"Weights saved to {path}")
    
    def load_weights(self, path):
        """Load model weights from .npy file."""
        weights_dict = np.load(path, allow_pickle=True).item()
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                layer.weights = weights_dict[f"layer_{i}_weights"]
                layer.biases = weights_dict[f"layer_{i}_biases"]
        
        print(f"Weights loaded from {path}")
