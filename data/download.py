"""Download and filter EMNLP 2025 AAC-scored conversational datasets from Hugging Face."""

import argparse
import hashlib
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

DATASET_SOURCES = {
    "c4": "figmtu/aac_c4_deberta_classified",  # 4.35B tokens
    "subtitles": "figmtu/aac_subtitle_deberta_classified",  # 52.6M tokens
    "c4-fast": "figmtu/aac_c4_deberta_classified_0.90",  # Pre-filtered high-density
}


def download_and_filter(
    score_threshold: float = 0.85,
    output_path: str = "data/raw_conversations.txt",
    source: str = "c4",
    resume: bool = True,
    deduplicate: bool = True,
):
    """Stream the DeBERTa-classified AAC dataset and keep high-confidence lines.

    Args:
        score_threshold: Minimum DeBERTa AAC score to keep a row.
        output_path: Path to write filtered lines.
        source: Dataset source - one of 'c4', 'subtitles', 'c4-fast'
        resume: If True and output_path exists, skip lines already written
        deduplicate: If True, skip exact duplicate lines using SHA-256
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

        with tqdm(desc="Processing", unit="lines") as pbar:
            for row in dataset:
                total_scanned += 1

                if row.get("aac_score", 0.0) >= score_threshold:
                    clean_text = row["text"].strip().replace("\n", " ")

                    # Check for duplicates
                    if deduplicate:
                        text_hash = hashlib.sha256(clean_text.encode()).hexdigest()
                        if text_hash in seen_hashes:
                            duplicates += 1
                            continue
                        seen_hashes.add(text_hash)

                    f.write(clean_text + "\n")
                    count += 1

                    if count % 50000 == 0:
                        pbar.set_postfix({"kept": count, "scanned": total_scanned, "dups": duplicates})

                pbar.update(1)

    print(f"Done. {count} lines written to {output_path}.")
    print(f"Scanned {total_scanned} total lines, skipped {duplicates} duplicates.")


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
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume if output file exists (overwrite instead)",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Don't deduplicate lines (keep all lines even if exact duplicates)",
    )
    args = parser.parse_args()

    download_and_filter(
        score_threshold=args.threshold,
        output_path=args.output,
        source=args.source,
        resume=not args.no_resume,
        deduplicate=not args.no_deduplicate,
    )
