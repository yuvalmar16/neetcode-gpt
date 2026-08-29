import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        #this automaticlly chains the sequence in the model
        self.model = nn.Sequential(
        nn.Linear(784, 512), #the photo is flattend where each value represents the brightness
        nn.ReLU(),  #the input layer 784(input pixels) -> the hidden layer (ReLU+Dropout) is 512
        nn.Dropout(0.2),
        nn.Linear(512,10), #output Layer is 10 (digits between 0-9)
        nn.Sigmoid() #the output is probability for each digit
        )



    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        predictions = self.model(images)
        return torch.round(predictions, decimals= 4)
