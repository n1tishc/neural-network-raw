"""
Optimization algorithms for neural network training.
"""

import numpy as np


class SGD:
    """Stochastic Gradient Descent: w = w - η * ∇w"""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.state = {}
    
    def update(self, key, param, grad):
        """Update parameter using SGD."""
        param -= self.learning_rate * grad
        return param


class SGDMomentum:
    """SGD with Momentum: v = βv + ∇w; w = w - ηv"""
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.state = {}
    
    def update(self, key, param, grad):
        """Update parameter using SGD with momentum."""
        if key not in self.state:
            self.state[key] = np.zeros_like(param)
        
        # Update velocity
        self.state[key] = self.momentum * self.state[key] + grad
        
        # Update parameter
        param -= self.learning_rate * self.state[key]
        
        return param


class RMSprop:
    """RMSprop: cache = β*cache + (1-β)*∇w²; w = w - η*∇w/(√cache + ε)"""
    
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.state = {}
    
    def update(self, key, param, grad):
        """Update parameter using RMSprop."""
        if key not in self.state:
            self.state[key] = np.zeros_like(param)
        
        # Update cache (exponentially weighted average of squared gradients)
        self.state[key] = self.decay_rate * self.state[key] + (1 - self.decay_rate) * grad ** 2
        
        # Update parameter
        param -= self.learning_rate * grad / (np.sqrt(self.state[key]) + self.epsilon)
        
        return param


class Adam:
    """Adam: Combines momentum + RMSprop with bias correction."""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.state = {}
        self.t = 0
    
    def update(self, key, param, grad):
        """Update parameter using Adam."""
        self.t += 1
        
        if key not in self.state:
            self.state[key] = {
                'm': np.zeros_like(param),  # First moment
                'v': np.zeros_like(param),  # Second moment
            }
        
        # Update biased first moment estimate
        self.state[key]['m'] = self.beta1 * self.state[key]['m'] + (1 - self.beta1) * grad
        
        # Update biased second moment estimate
        self.state[key]['v'] = self.beta2 * self.state[key]['v'] + (1 - self.beta2) * grad ** 2
        
        # Bias correction
        m_corrected = self.state[key]['m'] / (1 - self.beta1 ** self.t)
        v_corrected = self.state[key]['v'] / (1 - self.beta2 ** self.t)
        
        # Update parameter
        param -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
        
        return param
