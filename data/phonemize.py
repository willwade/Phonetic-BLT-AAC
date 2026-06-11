"""Multithreaded grapheme-to-phoneme conversion using Epitran → 1-byte SAMPA."""

import argparse
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from data.utils import SampaPhonemizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _process_chunk(lines: list[str]) -> tuple[list[str], dict[str, int]]:
    """Convert a chunk of text lines to SAMPA phoneme sequences.

    Returns:
        Tuple of (processed_lines, stats) where stats contains:
        - 'total_words': Total words processed
        - 'skipped_words': Words that failed phonemization
        - 'skipped_lines': Lines that produced no output
    """
    phonemizer = SampaPhonemizer(lang_code="eng-Latn")
    processed: list[str] = []
    stats = {"total_words": 0, "skipped_words": 0, "skipped_lines": 0}

    for line in lines:
        words = line.strip().split()
        if not words:
            continue

        stats["total_words"] += len(words)
        sampa_words = []
        for w in words:
            result = phonemizer.convert_word(w)
            if result:
                sampa_words.append(result)
            else:
                stats["skipped_words"] += 1

        if sampa_words:
            processed.append(" ".join(sampa_words))
        else:
            stats["skipped_lines"] += 1

    return processed, stats


def parallel_phonemize(
    input_path: str,
    output_path: str,
    chunk_size: int = 5000,
    max_workers: int | None = None,
    verify_ascii: bool = True,
    output_bytes: bool = False,
) -> None:
    """Phonemize a text file in parallel using all available CPU cores.

    Args:
        input_path: Path to raw text file (one sentence per line).
        output_path: Path to write phonemized SAMPA output.
        chunk_size: Number of lines per worker chunk.
        max_workers: Override for number of worker processes.
        verify_ascii: If True, verify all output is ASCII (0-127).
        output_bytes: If True, output byte-encoded file instead of text.
    """
    start_time = time.time()
    input_file = Path(input_path)
    output_file = Path(output_path)

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with input_file.open(encoding="utf-8") as f:
        lines = f.readlines()

    workers = max_workers or mp.cpu_count()
    chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]
    logger.info(f"Processing {len(lines)} lines over {workers} CPU cores...")

    total_stats = {"total_words": 0, "skipped_words": 0, "skipped_lines": 0}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_process_chunk, chunks))

    # Aggregate results and stats
    with output_file.open("w" if not output_bytes else "wb", encoding=None if output_bytes else "utf-8") as out:
        for chunk_lines, chunk_stats in results:
            for line in chunk_lines:
                if output_bytes:
                    out.write(line.encode("utf-8") + b"\n")
                else:
                    out.write(line + "\n")

            # Accumulate stats
            total_stats["total_words"] += chunk_stats["total_words"]
            total_stats["skipped_words"] += chunk_stats["skipped_words"]
            total_stats["skipped_lines"] += chunk_stats["skipped_lines"]

    elapsed = time.time() - start_time
    lines_per_sec = len(lines) / elapsed if elapsed > 0 else 0

    logger.info(f"Parallel phonemization complete. Output: {output_path}")
    logger.info(f"Performance: {lines_per_sec:.1f} lines/second")
    logger.info(f"Stats: {total_stats['total_words']} words, "
                f"{total_stats['skipped_words']} skipped ({total_stats['skipped_words']/max(total_stats['total_words'],1)*100:.1f}%), "
                f"{total_stats['skipped_lines']} empty lines")

    if verify_ascii and not output_bytes:
        # Verify ASCII output
        non_ascii_count = 0
        with output_file.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if any(ord(char) > 127 for char in line):
                    non_ascii_count += 1
                    if non_ascii_count <= 5:  # Show first 5 examples
                        logger.warning(f"Non-ASCII character found in line {line_num}: {line.strip()[:50]}...")

        if non_ascii_count > 0:
            logger.warning(f"Found {non_ascii_count} lines with non-ASCII characters (outside 0-127 range)")
        else:
            logger.info("✓ All output verified as ASCII (0-127)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phonemize raw text to SAMPA")
    parser.add_argument(
        "--input", type=str, default="data/raw_conversations.txt", help="Input text file"
    )
    parser.add_argument(
        "--output", type=str, default="data/phonemized.txt", help="Output SAMPA file"
    )
    parser.add_argument("--chunk-size", type=int, default=5000, help="Lines per worker chunk")
    parser.add_argument(
        "--no-verify-ascii",
        action="store_true",
        help="Skip ASCII verification of output",
    )
    parser.add_argument(
        "--output-bytes",
        action="store_true",
        help="Output byte-encoded file instead of text",
    )
    args = parser.parse_args()

    parallel_phonemize(
        args.input,
        args.output,
        args.chunk_size,
        verify_ascii=not args.no_verify_ascii,
        output_bytes=args.output_bytes,
    )
