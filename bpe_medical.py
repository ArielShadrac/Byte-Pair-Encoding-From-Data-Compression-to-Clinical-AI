"""
This module initializes and trains a BPE (Byte Pair Encoding) tokenizer specifically
designed for medical text processing. The tokenizer learns subword units from medical
corpus data, enabling efficient tokenization of medical terms and clinical text.
The tokenizer is configured with:
- Vocabulary size: 500 tokens
- Minimum frequency threshold: 2 occurrences
- Unknown token handler: [UNK] for out-of-vocabulary words
Key Features:
    - Custom medical terminology training data
    - Subword tokenization for handling complex medical terms
    - Unknown token handling for rare/unseen terms
    - Token encoding capability for inference
Example:
    In thios case the tokenizer is trained on a small medical corpus containing examples of
    hepatomegaly, viral hepatitis, and gastroenterological conditions. After
    training, it can encode medical terms into their corresponding token sequences.
Usage:
    After training, the tokenizer can be used to convert medical text into
    numerical token representations suitable for downstream NLP tasks such as
    classification, named entity recognition, or text analysis in healthcare
    applications.
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# Initialize BPE tokenizer with unknown token handler
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Configure trainer with vocabulary size and minimum frequency threshold
trainer = BpeTrainer(vocab_size=500, min_frequency=2)

# Medical terminology training data
training_data = [
    "The patient presents hepatomegaly with signs of gastritis.",
    "Viral hepatitis requires strict hepato-biliary follow-up.",
    "Gastroenterological examination normal."
]

# Train tokenizer on medical corpus
tokenizer.train_from_iterator(training_data, trainer)

# Encode sample medical term
tokens = tokenizer.encode("hepatitis")
print("Tokens:", tokens.tokens)