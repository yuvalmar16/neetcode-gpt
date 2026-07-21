import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        X = np.array(X)
        y = np.array(y).reshape(-1, 1) # critical so y is two dimentional
        num_samples, num_features = X.shape
        w = np.zeros((num_features,1))
        b=0.0
        
        for _ in range(epochs):
             y_hat = X@w + b
             L = (1/num_samples) * np.sum((y_hat - y)**2)
             grad_w = (2 / num_samples) * (X.T @ (y_hat - y))
             grad_b = (2/num_samples) * (np.sum(y_hat - y))
             w = w - lr*grad_w
             b = b - lr*grad_b
        w= np.round(w.flatten() , 5)
        b= np.round(float(b), 5)    
        return (w, b)




        