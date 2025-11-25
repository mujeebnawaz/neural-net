import numpy as np

"""
Static script to house all the functional methods. 
"""

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))  
    return e_x / np.sum(e_x)  


def softmax_derivative(a):
    """
    Compute the Jacobian of softmax vector 'a'
    a: softmax output vector (n,)
    returns: (n,n) Jacobian matrix
    """
    # reshape a to column vector
    a = a.reshape(-1,1)
    # diag(a) - a @ a.T
    return np.diagflat(a) - np.dot(a, a.T)

def mock(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)

def cross_entropy_loss(y_true, y_pred, epsilon=1e-12):
    """
    Compute the cross-entropy loss between true labels and predicted probabilities.
    Adding epsilon to avoid log(0).
    """
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon) 
    loss = -np.sum(y_true * np.log(y_pred))
    return loss

def binary_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)

def get_one_hot( index_of_true_class, number_of_classes ):
    if number_of_classes == 1:
        return index_of_true_class
    classes = np.zeros( number_of_classes )
    classes[ index_of_true_class ] = 1
    return classes

def relu_derivative(x):
    return (x > 0).astype(float)  # Derivative of ReLU is 1 for positive, 0 for negative inputs

def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_pred, axis=1) == y_true)

"""
Weight managament
"""
def get_random(shape, method='xavier'):
    """
    Returns a weight matrix initialized using Xavier or He initialization.
    
    shape: tuple, (n_out, n_in)
    method: 'xavier' or 'he'
    """
    n_out, n_in = shape  # automatically infer from shape

    if method == 'xavier':
        std = np.sqrt(2 / (n_in + n_out))
        return np.random.randn(*shape) * std
    elif method == 'he':
        std = np.sqrt(2 / n_in)
        return np.random.randn(*shape) * std
    else:
        return np.random.randn(*shape)
