import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        z=np.array(z)
        print(z)
        sigmoid = 1/(1+np.exp(-z))
        
        return np.round(sigmoid, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        z=np.array(z)
        
        rel=[]
        for i in range(0,len(z)):
           rel.append(max(0.0,z[i]))
        return rel 
