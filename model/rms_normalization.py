import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x = np.array(x)
        gamma = np.array(gamma)
        eps = 1e-5
        n = len(x)
        rms = np.sqrt((1/n) * np.sum(x**2 + eps))
        x_hat = []
        for i in range(n):
            x_hat.append(x[i]/rms)
        return np.round(gamma*x_hat, 4)