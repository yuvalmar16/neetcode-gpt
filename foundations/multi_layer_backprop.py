import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x)
        w1 = np.array(W1)
        w2 = np.array(W2)
        b1 = np.array(b1)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        
        # ----------------------------------------------------
        # (Forward Pass) linear+relu+predictions+loss
        # ----------------------------------------------------
        z1 = w1 @ x + b1
        a1 = np.maximum(0 , z1) #relue
        predictions = w2 @ a1 + b2
        Loss = np.mean((predictions - y_true)**2)
        print('done')
        # ----------------------------------------------------
        # (Backward Pass) d_pred + d
        # ----------------------------------------------------
        n = len(y_true) #deriviate the loss by predictions
        d_pred  = (2/n)* (predictions-y_true)

        db2 = d_pred #deriviate of the loss by b2
        dw2 = np.outer(d_pred, a1) #deriviate of the loss by w2
         
        da1 = w2.T @ d_pred  #deriviate of the loss by a1
        dz1 = da1 * (z1 > 0) 

        db1 = dz1 
        dw1 = np.outer(dz1, x)
       
        
        return {
            'loss': float(np.round(Loss, 4)),
            'dW1': np.round(dw1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dw2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }







