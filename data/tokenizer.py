from typing import List
from collections import Counter


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens=list(corpus)
        mergers=[]
        for _ in range(num_merges):
            most_frequent_pair = self.most_frequent_pair(tokens)
            if most_frequent_pair is None:  # Stop early if fewer than 2 tokens remain
                break
            token_a, token_b=most_frequent_pair
            merged_token = token_a + token_b
            tokens= self.merge_non_overlapping(tokens, token_a, token_b, merged_token)
            mergers.append([token_a, token_b])
        return mergers


    def most_frequent_pair(self, tokens):
        if len(tokens) < 2:
            return None
        
        counts = Counter(zip(tokens, tokens[1:]))
        # Negative count finds the maximum frequency; tuple comparison breaks ties lexicographically
        return min(counts.items(), key=lambda x: (-x[1], x[0]))[0]   


    def merge_non_overlapping(self,tokens, token_a, token_b, merged_token):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == token_a
                and tokens[i + 1] == token_b
            ):
                new_tokens.append(merged_token)
                i += 2  # Skip next token to ensure non-overlapping left-to-right merge
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens


