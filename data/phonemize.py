"""Multithreaded grapheme-to-phoneme conversion using Epitran → 1-byte SAMPA."""

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from data.utils import SampaPhonemizer


def _process_chunk(lines: list[str]) -> list[str]:
    """Convert a chunk of text lines to SAMPA phoneme sequences."""
    phonemizer = SampaPhonemizer(lang_code="eng-Latn")
    processed: list[str] = []
    for line in lines:
        words = line.strip().split()
        sampa_words = [phonemizer.convert_word(w) for w in words]
        sampa_words = [w for w in sampa_words if w]
        if sampa_words:
            processed.append(" ".join(sampa_words))
    return processed


def parallel_phonemize(
    input_path: str,
    output_path: str,
    chunk_size: int = 5000,
    max_workers: int | None = None,
) -> None:
    """Phonemize a text file in parallel using all available CPU cores.

    Args:
        input_path: Path to raw text file (one sentence per line).
        output_path: Path to write phonemized SAMPA output.
        chunk_size: Number of lines per worker chunk.
        max_workers: Override for number of worker processes.
    """
    with open(input_path, encoding="utf-8") as f:
        lines = f.readlines()

    workers = max_workers or mp.cpu_count()
    chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]
    print(f"Processing {len(lines)} lines over {workers} CPU cores...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_process_chunk, chunks))

    with open(output_path, "w", encoding="utf-8") as out:
        for chunk in results:
            for line in chunk:
                out.write(line + "\n")

    print(f"Parallel phonemization complete. Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phonemize raw text to SAMPA")
    parser.add_argument(
        "--input", type=str, default="data/raw_conversations.txt", help="Input text file"
    )
    parser.add_argument(
        "--output", type=str, default="data/phonemized.txt", help="Output SAMPA file"
    )
    parser.add_argument("--chunk-size", type=int, default=5000, help="Lines per worker chunk")
    args = parser.parse_args()
    parallel_phonemize(args.input, args.output, args.chunk_size)
