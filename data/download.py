"""Download and filter EMNLP 2025 AAC-scored conversational datasets from Hugging Face."""

import argparse

from datasets import load_dataset


def download_and_filter(
    score_threshold: float = 0.85, output_path: str = "data/raw_conversations.txt"
):
    """Stream the DeBERTa-classified AAC C4 dataset and keep high-confidence lines.

    Args:
        score_threshold: Minimum DeBERTa AAC score to keep a row.
        output_path: Path to write filtered lines.
    """
    print(f"Downloading EMNLP 2025 AAC C4 dataset (threshold: {score_threshold})...")

    dataset = load_dataset("figmtu/aac_c4_deberta_classified", split="train", streaming=True)

    with open(output_path, "w", encoding="utf-8") as f:
        count = 0
        for row in dataset:
            if row.get("aac_score", 0.0) >= score_threshold:
                clean_text = row["text"].strip().replace("\n", " ")
                f.write(clean_text + "\n")
                count += 1
                if count % 50000 == 0:
                    print(f"  Extracted {count} lines...")

    print(f"Done. {count} lines written to {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AAC-scored conversational data")
    parser.add_argument("--threshold", type=float, default=0.85, help="Minimum DeBERTa AAC score")
    parser.add_argument(
        "--output", type=str, default="data/raw_conversations.txt", help="Output file path"
    )
    args = parser.parse_args()
    download_and_filter(args.threshold, args.output)
