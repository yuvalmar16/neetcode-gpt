import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        y_true=np.array(y_true)
        epsilon= 1e-15
        y_pred=np.clip(y_pred, epsilon, 1 - epsilon)
        formula=[]
        n=len(y_true)
        for i in range(n):
            formula.append( y_true[i]*np.log(y_pred[i])  + (1-y_true[i])*np.log(1-y_pred[i]) )
        formula_sum= np.sum(formula)
        L=-(1/n)*formula_sum
        return round(float(L),4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        y_true=np.array(y_true)
        y_pred=np.array(y_pred)
        epsilon= 1e-15
        y_pred=np.clip(y_pred, epsilon, 1 - epsilon)
        formula=[]
        n=y_true.shape[0]
        c=y_true.shape[1]
        outter=[]
        for i in range(n):
            inner=[]
            inner_sum=0
            for j in range(c):
                inner.append(y_true[i][j]*np.log(y_pred[i][j]))
            inner_sum=np.sum(inner)
            outter.append(inner_sum)
        outter_sum=np.sum(outter)    
    

        L=-(1/n)*outter_sum
        return round(float(L),4)