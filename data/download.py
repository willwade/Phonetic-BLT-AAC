"""Download AAC-scored conversational data from Hugging Face."""

import argparse
import hashlib
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


DATASET_SOURCES = {
    "c4": "figmtu/aac_c4_deberta_classified",  # 4.35B tokens
    "subtitles": "figmtu/aac_subtitle_deberta_classified",  # 52.6M tokens
    "c4-fast": "figmtu/aac_c4_deberta_classified_0.90",  # Pre-filtered high-density
}


def download_aac_data(
    score_threshold: float = 0.85,
    output_path: str = "data/raw_conversations.txt",
    source: str = "c4",
    resume: bool = True,
    deduplicate: bool = True,
    max_rows: int | None = None,
):
    """Download AAC-scored conversational data from HuggingFace.

    Args:
        score_threshold: Minimum DeBERTa AAC score (0-1)
        output_path: Where to save the filtered text
        source: Dataset source to download
        resume: If True and output_path exists, skip lines already written
        deduplicate: If True, skip exact duplicate lines using SHA-256
        max_rows: Optional limit on number of rows to download (for testing)

    Returns:
        Number of lines written to output file
    """
    if source not in DATASET_SOURCES:
        raise ValueError(f"Invalid source: {source}. Must be one of {list(DATASET_SOURCES.keys())}")

    dataset_name = DATASET_SOURCES[source]
    print(f"Downloading {dataset_name} (threshold: {score_threshold})...")

    # Handle resume logic
    output_file = Path(output_path)
    start_line = 0
    if resume and output_file.exists():
        start_line = sum(1 for _ in output_file.open(encoding="utf-8"))
        print(f"Resuming from line {start_line}...")

    # Track seen hashes for deduplication
    seen_hashes = set() if deduplicate else None

    dataset = load_dataset(dataset_name, split="train", streaming=True)

    # Skip lines if resuming
    if start_line > 0:
        for _ in range(start_line):
            next(dataset)

    with output_file.open("a", encoding="utf-8") as f:
        count = start_line
        duplicates = 0
        total_scanned = 0

        pbar = tqdm(desc="Processing", unit="lines")
        for example in dataset:
            total_scanned += 1

            if max_rows and total_scanned > max_rows:
                print(f"Reached max_rows limit ({max_rows}), stopping download.")
                break

            text = example.get("text", "").strip()

            if not text:
                continue

            # Use dialogue_prob as the AAC score
            score = example.get("dialogue_prob", 0.0)
            if score < score_threshold:
                continue


            # Check for duplicates
            if deduplicate:
                text_hash = hashlib.sha256(text.encode()).hexdigest()
                if text_hash in seen_hashes:
                    duplicates += 1
                    continue
                seen_hashes.add(text_hash)

            f.write(text + "\n")
            count += 1
            pbar.set_postfix({"kept": count, "scanned": total_scanned, "dups": duplicates})

            pbar.update(1)

    print(f"Done. {count} lines written to {output_path}.")
    print(f"Scanned {total_scanned} total lines, skipped {duplicates} duplicates.")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AAC-scored conversational data")
    parser.add_argument("--threshold", type=float, default=0.85, help="Minimum DeBERTa AAC score")
    parser.add_argument(
        "--output", type=str, default="data/raw_conversations.txt", help="Output file path"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="c4",
        choices=list(DATASET_SOURCES.keys()),
        help="Dataset source to download",
    )
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing file")
    parser.add_argument("--no-deduplicate", action="store_true", help="Don't remove duplicates")
    parser.add_argument("--max-rows", type=int, help="Maximum rows to download (for testing)")

    args = parser.parse_args()

    download_aac_data(
        score_threshold=args.threshold,
        output_path=args.output,
        source=args.source,
        resume=not args.no_resume,
        deduplicate=not args.no_deduplicate,
        max_rows=args.max_rows,
    )
