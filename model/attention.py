import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key=nn.Linear(embedding_dim,attention_dim, bias =False)
        self.query=nn.Linear(embedding_dim,attention_dim, bias =False)
        self.value=nn.Linear(embedding_dim,attention_dim, bias =False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        K=self.key(embedded) #(K=X⋅WK) 
        Q=self.query(embedded) #(Q=X*WQ)
        V=self.value(embedded) #(V=X*WV)

        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        attetion_dim = K.shape[2] #HOW MANY ROWS
        scores = (Q @ K.transpose(-1,1))/(attention_dim**0.5)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        seq_len = embedded.shape[1] #HOW MANY COLUMNS
        mask = torch.tril(torch.ones(seq_len, seq_len, device=embedded.device))
    
        #    then masked_fill positions where mask == 0 with float('-inf')
        scores  = scores.masked_fill(mask==0, float('-inf'))
        # 4. Apply softmax(dim=2) to masked scores
        attention_weights = torch.softmax(scores, dim=2)
        # 5. Return (scores @ V) rounded to 4 decimal places
        output = attention_weights@V
        return torch.round(output*10000)/10000
  
