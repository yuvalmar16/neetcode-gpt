import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z=np.array(z)
        softmax=[]
        for i in range(len(z)):
            square_numenator=z[i]
            print(square_numenator)
            numenator=np.exp(square_numenator-np.max(z))
            total_denomenator=0
            for j in range(len(z)):
                square_denomenator=z[j]
                total_denomenator+=np.exp(square_denomenator-np.max(z)) 
            print(total_denomenator)   
               
            softmax.append(numenator/total_denomenator)    
        return np.round(softmax,4)