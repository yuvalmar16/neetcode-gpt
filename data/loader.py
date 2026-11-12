import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        # data: 1D tensor of encoded text (integer token IDs)
        # context_length: number of tokens in each training example
        # batch_size: number of examples per batch
        #
        # Return (X, Y) where:
        # - X has shape (batch_size, context_length)
        # - Y has shape (batch_size, context_length)
        # - Y is X shifted right by 1 (Y[i][j] = data[start_i + j + 1])
        
        # Use torch.manual_seed(0) before generating random start indices
        # Use torch.randint to pick random starting positions

        # Fix the random number generator seed so the random batch indices are reproducible across runs
        torch.manual_seed(0)

        # Randomly pick `batch_size` starting indices from valid positions (leaving enough room for `context_length` + 1 target token)
        ix = torch.randint(len(data) - context_length, (batch_size,))

        # Extract `batch_size` input sequences of length `context_length` and stack them into a 2D tensor of shape (batch_size, context_length)
        X = torch.stack([data[i : i + context_length] for i in ix])

        # Extract the corresponding target sequences shifted 1 token to the right (next-token prediction targets) of shape (batch_size, context_length)
        Y = torch.stack([data[i + 1 : i + 1 + context_length] for i in ix])

        # Return the batch of input contexts (X) and target next-tokens (Y)
        return X, Y



