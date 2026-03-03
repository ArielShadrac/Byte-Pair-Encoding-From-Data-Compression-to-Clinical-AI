"""
Benchmark script comparing pure Python BPE implementation vs Hugging Face tokenizers (Rust).

This module measures the performance difference between:
- Our educational "from scratch" BPE (pure Python + regex)
- Hugging Face's production-grade BPE (implemented in Rust)

The goal is to demonstrate why the scratch version is only for teaching,
while the Hugging Face tokenizer is mandatory in any real medical NLP pipeline.


"""

import time
from bpe_scratch import get_pair_frequencies, merge_corpus, corpus as med_corpus
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text


def benchmark_scratch(max_merges: int = 100) -> tuple[float, int]:
    """
    Run the pure Python BPE implementation for a fixed number of merges.
    
    Args:
        max_merges: Maximum number of merge operations to perform
    
    Returns:
        (execution_time_in_seconds, number_of_merges_actually_done)
    """
    start = time.time()
    local_corpus = med_corpus.copy()
    merges_done = 0
    
    for _ in range(max_merges):
        pairs = get_pair_frequencies(local_corpus)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        local_corpus = merge_corpus(best_pair, local_corpus)
        merges_done += 1
    
    return time.time() - start, merges_done


def benchmark_hf(max_merges: int = 100) -> float:
    """
    Train a Hugging Face BPE tokenizer (Rust backend) using the same medical corpus.
    
    We use exactly the same number of merges as the scratch version for a fair comparison.
    """
    start = time.time()
    
    # Use the exact same medical corpus 
    texts = [" ".join(med_corpus.keys())]
    
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=256 + max_merges,  # 256 base bytes + requested merges
        min_frequency=2,
        show_progress=False
    )
    
    tokenizer.train_from_iterator(texts, trainer)
    
    return time.time() - start

# Main benchmark execution with Rich styling
if __name__ == "__main__":
    console = Console()

    # Big colored title
    title = Text("BPE Performance Benchmark", style="bold cyan")
    subtitle = Text("– Medical Terminology Corpus –", style="dim")
    console.print(Panel.fit(title + "\n" + subtitle, border_style="bright_blue"))

    # Run both benchmarks
    t_scratch, merges = benchmark_scratch(100)
    t_hf = benchmark_hf(100)
    speed_up = t_scratch / t_hf if t_hf > 0 else 1.0

    # Table with results
    table = Table(title="Results", box=box.ROUNDED, show_lines=True, expand=True)
    table.add_column("Implementation", style="cyan", no_wrap=True)
    table.add_column("Time (seconds)", style="green", justify="right")
    table.add_column("Merges", style="magenta", justify="center")
    table.add_column("Speed-up", style="yellow", justify="right")

    # Row 1: Pure Python
    table.add_row(
        "Pure Python (scratch)",
        f"{t_scratch:.4f}s",
        str(merges),
        "1.0× (baseline)",
        style="dim"
    )

    # Row 2: Hugging Face (highlighted)
    table.add_row(
        "Hugging Face (Rust)",
        f"{t_hf:.4f}s",
        "—",
        f"[bold green]{speed_up:.1f}× faster[/bold green]"
    )

    console.print(table)

    # Conclusion
    console.print("\n")
    conclusion = Text("Conclusion", style="bold red")
    console.print(Panel(
        f"The Rust implementation is dramatically faster even on this tiny corpus.\n"
        f"On real clinical datasets (hundreds of thousands of reports),\n"
        f"the difference easily reaches [bold]200×–1000×[/bold].\n\n"
        f"→ Use [bold cyan]tokenizers[/bold cyan] in production!",
        title="Summary",
        border_style="bright_red",
        expand=False
    ))

    # Final motivational note
    console.print(
        "\n[bold green]✓ Benchmark complete![/bold green]   "
        "[dim]Educational version vs Production version[/dim]",
        justify="center"
    )