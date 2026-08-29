import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        all_santances= positive + negative
        words = set()
        for sentence in all_santances:
            for word in sentence.split():
                words.add(word)

        unique_words = sorted(words)
        word_vocab = {word: idx for idx, word in enumerate(unique_words, start=1)}
        # 2. Encode each sentence by replacing words with their IDs so a sentance is just full of word ids
        encoded_tensors = []
        for sentence in all_santances:
            token_ids = [word_vocab[word] for word in sentence.split()]
            encoded_tensors.append(torch.tensor(token_ids, dtype=torch.long))





        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        padded_dataset = nn.utils.rnn.pad_sequence(encoded_tensors, batch_first=True, padding_value=0)
        return padded_dataset.float()
            

