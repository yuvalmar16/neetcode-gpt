import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        X=np.array(X)
        W=np.array(weights)
        Y=X@W
        return np.round(Y,5)






    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        model_prediction=np.array(model_prediction)
        ground_truth = np.array(ground_truth)
        n=ground_truth.shape[0]
        squares= (ground_truth - model_prediction) ** 2
        MSE= (1/n)*np.sum(squares)
        return np.round(MSE, 5)












