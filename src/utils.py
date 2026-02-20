import numpy as np

def truncated_normal(mean, std, low, high):
    """Generates a value from a truncated normal distribution."""
    while True:
        val = np.random.normal(mean, std)
        if low <= val <= high:
            return val

def normalize(value, min_val, max_val):
    """Normalizes a value between 0 and 1."""
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0

def sigmoid(x):
    """Sigmoid activation function for smooth transitions."""
    return 1 / (1 + np.exp(-x))
