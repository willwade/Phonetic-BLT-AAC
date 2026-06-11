"""Shared data utilities: text cleaning, byte encoding, and the SampaPhonemizer wrapper."""

import re
import unicodedata

import epitran


class SampaPhonemizer:
    """Wraps Epitran for deterministic grapheme-to-phoneme conversion to SAMPA.

    Args:
        lang_code: Epitran language code (e.g. ``"eng-Latn"``).
    """

    def __init__(self, lang_code: str = "eng-Latn"):
        self._epi = epitran.Epitran(lang_code)

    def convert_word(self, word: str) -> str:
        """Convert a single word to its SAMPA phoneme string.

        Args:
            word: A plain-text word.

        Returns:
            SAMPA phoneme string, or empty string on failure.
        """
        try:
            return self._epi.transliterate(word)
        except Exception:
            return ""

    def convert_sentence(self, sentence: str) -> str:
        """Convert a full sentence to space-separated SAMPA phonemes."""
        words = sentence.strip().split()
        results = [self.convert_word(w) for w in words]
        return " ".join(r for r in results if r)


def clean_text(text: str) -> str:
    """Normalize and clean raw text for pipeline ingestion.

    Strips control characters, collapses whitespace, and normalizes Unicode.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_to_bytes(text: str) -> list[int]:
    """Encode a string to a list of byte values (0-255) for BLT input."""
    return list(text.encode("utf-8"))


def train_val_test_split(
    lines: list[str], train_frac: float = 0.9, val_frac: float = 0.05
) -> tuple[list[str], list[str], list[str]]:
    """Split a list of lines into train / val / test sets.

    Remaining fraction after train + val goes to test.
    """
    n = len(lines)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return lines[:train_end], lines[train_end:val_end], lines[val_end:]
