"""
This scpirt is a simplified version of the Byte Pair Encoding algorithm,
which is a data compression and tokenization technique commonly used in NLP.
The algorithm iteratively identifies the most frequent pair of adjacent symbols
in a corpus and merges them into a single token. This process reduces the vocabulary
size while preserving important linguistic patterns.
Module Functions:
    - get_pair_frequencies(): Counts occurrences of all adjacent symbol pairs
    - merge_corpus(): Replaces all instances of a specific pair with a merged token
    
For educational purpoe the implementation processes a medical terminology corpus 
containing French medical terms (hepatitis, hepatomegaly, gastritis, gastroenterologist) 
split into individual characters with word boundary markers. Over 10 iterations, 
the algorithm learns to merge the most frequent character sequences, progressively 
building more meaningful subword units from the raw character-level input.
"""

import re
from collections import defaultdict
from typing import Dict, Tuple

def get_pair_frequencies(corpus: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """Calculate frequency of all adjacent symbol pairs in corpus."""
    pairs = defaultdict(int)
    for word, frequency in corpus.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += frequency
    return pairs

def merge_corpus(pair: Tuple[str, str], corpus: Dict[str, int]) -> Dict[str, int]:
    """Merge all occurrences of a pair in the corpus."""
    new_corpus = {}
    bigram = re.escape(' '.join(pair))
    replacement = ''.join(pair)
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    for word, frequency in corpus.items():
        new_word = pattern.sub(replacement, word)
        new_corpus[new_word] = frequency
    
    return new_corpus

# Medical terminology corpus
corpus = {
    'h é p a t i t e </w>': 10,
    'h é p a t o m é g a l i e </w>': 5,
    'g a s t r i t e </w>': 8,
    'g a s t r o l o g u e </w>': 3
}

print("--- Byte Pair Encoding Training ---")
for iteration in range(10):
    pairs = get_pair_frequencies(corpus)
    if not pairs:
        break
    
    most_frequent_pair = max(pairs, key=pairs.get)
    corpus = merge_corpus(most_frequent_pair, corpus)
    print(f"Merge {iteration + 1}: {most_frequent_pair}")

print("\nFinal result:", corpus)