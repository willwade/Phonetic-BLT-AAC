"""Debug download process to see what's happening."""

from datasets import load_dataset
from tqdm import tqdm


def test_download():
    print("Testing download with debug output...")

    dataset = load_dataset("figmtu/aac_c4_deberta_classified", split="train", streaming=True)
    print("Dataset loaded successfully")

    count = 0
    max_rows = 10
    score_threshold = 0.95

    pbar = tqdm(desc="Processing", unit="lines", total=max_rows)
    for example in dataset:
        if count >= max_rows:
            break

        text = example.get("text", "").strip()
        score = example.get("dialogue_prob", 0.0)

        print(f"Row {count}: score={score:.4f}, text_len={len(text)}, text='{text[:50]}...'")

        if not text:
            print("  -> Skipped: empty text")
            continue

        if score < score_threshold:
            print(f"  -> Skipped: score {score:.4f} < {score_threshold}")
            continue

        print("  -> ACCEPTED: writing to file")
        count += 1
        pbar.update(1)

    print(f"\nDone: {count} lines would be written")


if __name__ == "__main__":
    test_download()
