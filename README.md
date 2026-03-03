# Byte Pair Encoding : From Data Compression to Clinical AI

**From 1994 compression algorithm to the heart of every modern medical AI model**  
A complete educational repository showing how BPE evolved from simple data compression into the core tokenization engine behind GPT-4, Llama, BioBERT, DrBERT and clinical language models.

---

## Historical Journey

**1994** : Philip Gage publishes [A New Algorithm for Data Compression](https://dl.acm.org/doi/10.5555/177910.177914).  
The goal was simple: replace the most frequent byte pairs with a single unused symbol to save storage space.  


**2016** : Rico Sennrich, Barry Haddow, and Alexandra Birch (University of Edinburgh) publish their groundbreaking paper:  
[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909).  
They apply Byte Pair Encoding to NLP and solve the "out-of-vocabulary" (OOV) problem forever.  
Today, **every large language model** (including all medical LLMs) uses Byte-level BPE or a direct descendant.  

## Why BPE is Extremely Powerful in Healthcare

Medical language is one of the hardest domains for tokenization. BPE excels here:

- **hepatomegaly** → `hepato` + `megaly`  
- **pneumothorax** → `pneumo` + `thorax`  
- **splenomegaly** → `spleno` + `megaly`  
- Handles `anti-TNF-alpha`, rare pharmaceuticals, brand-new drug names, and typing errors in patient records without breaking  
- Perfect for **k-mer analysis** in genomics (DNA/RNA sequences have no natural spaces or word boundaries)

This subword approach gives clinical AI models perfect coverage of complex medical terminology while remaining robust to real-world clinical text.


## Repository Contents

```
Byte-Pair-Encoding-From-Data-Compression-to-Clinical-AI/
├── bpe_scratch.py                  # Pure Python BPE (step-by-step educational version)
├── train_medical_tokenizer.py      # Real medical tokenizer trained with Hugging Face
├── benchmark.py                    # Performances comparison
├── requirments.txt
└── README.md
```


## What You Will Learn

1. **Pure Python implementation** : see every merge step with medical terms  
2. **Production tokenizer** : trained specifically on clinical French/English text  
3. **Performance reality check** : Python vs Rust


## Installation & Quick Start

```bash
# 1. Clone with the exact repo name
git clone https://github.com/ArielShadrac/Byte-Pair-Encoding-From-Data-Compression-to-Clinical-AI.git
cd Byte-Pair-Encoding-From-Data-Compression-to-Clinical-AI

# 2. Install dependencies
pip install rich tokenizers

# 3. Run in order:
python bpe_scratch.py          # Watch the algorithm learn medical subwords
python train_medical_tokenizer.py   # Train a real clinical tokenizer
python benchmark.py            # See the dramatic speed difference (colored table)
```

**Requires**: Python 3.10+


## Disclaimer

**Educational Only**  
Not certified for clinical use. Always validate on your own data and use production libraries (`tokenizers`, `tiktoken`) in real healthcare projects.
