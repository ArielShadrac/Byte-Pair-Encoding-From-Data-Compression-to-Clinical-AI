"""
Benchmark script comparing pure Python BPE implementation vs Hugging Face tokenizers (Rust backend).

This script demonstrates the performance difference between:
- A pure-Python educational BPE implementation (character-level with regex merges)
- Hugging Face's production-grade tokenizers library (implemented in Rust)

Key goal: Show why the scratch version is only suitable for learning,
while Hugging Face tokenizers should be used in any real-world medical NLP pipeline.

Uses an external file 'medical_corpus.txt' containing realistic English medical texts
(clinical notes, discharge summaries, etc.) to make the benchmark more meaningful.
"""

import time
import os
from bpe_scratch import get_pair_frequencies, merge_corpus, corpus as med_corpus_original
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text


# path to the external medical corpus file

CORPUS_FILE = "corpus.txt"


def load_external_corpus():
    """
    Load and preprocess the external medical corpus from file.

    """
    if not os.path.exists(CORPUS_FILE):
        raise FileNotFoundError(
            f"Corpus file '{CORPUS_FILE}' not found.\n"
        )

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Prepare corpus for scratch BPE (character-level with word-end marker)
    extended_corpus = med_corpus_original.copy()
    for line in lines:
        # Lowercase → split into characters → join with spaces → add </w>
        processed = ' '.join(c for c in line.lower()) + ' </w>'
        extended_corpus[processed] = extended_corpus.get(processed, 0) + 1

    return extended_corpus, lines


# Load corpus once at startup
EXTENDED_CORPUS, RAW_TEXTS = load_external_corpus()


def benchmark_scratch(max_merges: int = 100) -> tuple[float, int]:
    """
    Benchmark the pure-Python scratch BPE implementation.
    Runs up to max_merges merges on the extended medical corpus.
    Measures total execution time and actual number of merges performed.
    """
    start = time.time()
    local_corpus = EXTENDED_CORPUS.copy()
    merges_done = 0

    for _ in range(max_merges):
        pairs = get_pair_frequencies(local_corpus)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        local_corpus = merge_corpus(best_pair, local_corpus)
        merges_done += 1

    return time.time() - start, merges_done


def benchmark_hf(max_merges: int = 100, repetition: int = 20) -> float:
    """
    Benchmark Hugging Face BPE tokenizer (Rust backend).
    Trains a new tokenizer from scratch on repeated medical texts
    to simulate a larger corpus and highlight Rust's efficiency.

    """
    start = time.time()

    # Repeat the corpus to make training more representative of real data volume
    texts = RAW_TEXTS * repetition

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=256 + max_merges,    # base bytes + learned merges
        min_frequency=2,
        show_progress=False
    )

    tokenizer.train_from_iterator(texts, trainer)

    return time.time() - start

# Main execution output
if __name__ == "__main__":
    console = Console()

    # Header
    title = Text("BPE Performance Benchmark", style="bold cyan")
    subtitle = Text("– External Medical Corpus (English) –", style="dim")
    console.print(Panel.fit(title + "\n" + subtitle, border_style="bright_blue"))

    # Show corpus info
    console.print(f"[yellow]Loaded {len(RAW_TEXTS)} lines from '{CORPUS_FILE}'[/yellow]\n")

    # Run benchmarks
    t_scratch, merges = benchmark_scratch(100)
    t_hf = benchmark_hf(100, repetition=20)
    speed_up = t_scratch / t_hf if t_hf > 0 else 1.0

    # Results table
    table = Table(title="Results", box=box.ROUNDED, show_lines=True, expand=True)
    table.add_column("Implementation", style="cyan", no_wrap=True)
    table.add_column("Time (seconds)", style="green", justify="right")
    table.add_column("Data note", style="magenta", justify="center")
    table.add_column("Speed-up", style="yellow", justify="right")

    table.add_row(
        "Pure Python (scratch)",
        f"{t_scratch:.4f}s",
        f"{len(EXTENDED_CORPUS)} entries",
        "1.0× (baseline)",
        style="dim"
    )

    table.add_row(
        "Hugging Face (Rust)",
        f"{t_hf:.4f}s",
        f"~{len(RAW_TEXTS) * 20} sentences",
        f"[bold green]{speed_up:.1f}× faster[/bold green]" if speed_up > 1 else f"[red]{speed_up:.1f}× (slower on small data)[/red]"
    )

    console.print(table)

    console.print("\n")

    # Dynamic conclusion based on actual speedup
    if speed_up > 3:
        color = "bright_green"
        msg = (
            f"Rust is [bold]{speed_up:.1f}× faster[/bold] on this real medical corpus!\n"
            "On large-scale clinical datasets (thousands/millions of lines),\n"
            "the difference typically reaches [bold]50×–500×[/bold] or more.\n\n"
            "→ Always use [bold cyan]tokenizers[/bold cyan] in production!"
        )
    else:
        color = "yellow"
        msg = (
            "On this dataset size, times are still relatively close.\n"
            "Add more lines to 'medical_corpus.txt' or increase 'repetition'\n"
            "to see Rust's true advantage on bigger data."
        )

    console.print(Panel(msg, title="Conclusion", border_style=color))

    console.print(
        "\n[bold green]✓ Benchmark completed![/bold green]   "
        "[dim]Educational version vs Production version[/dim]",
        justify="center"
    )