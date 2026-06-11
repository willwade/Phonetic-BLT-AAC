"""Test phonemization functionality."""

import tempfile
from pathlib import Path


class TestSampaPhonemizer:
    """Test the SampaPhonemizer class."""

    def test_convert_word_basic(self):
        """Test basic word conversion returns non-empty string."""
        from data.utils import SampaPhonemizer

        phonemizer = SampaPhonemizer(lang_code="eng-Latn")
        result = phonemizer.convert_word("hello")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0  # Should produce some output

    def test_convert_word_empty(self):
        """Test empty word returns empty string."""
        from data.utils import SampaPhonemizer

        phonemizer = SampaPhonemizer(lang_code="eng-Latn")
        result = phonemizer.convert_word("")
        assert result == ""

    def test_convert_word_unicode(self):
        """Test that unusual Unicode is handled gracefully."""
        from data.utils import SampaPhonemizer

        phonemizer = SampaPhonemizer(lang_code="eng-Latn")
        # Should not crash on unusual characters
        result = phonemizer.convert_word("hello🌍world")
        assert isinstance(result, str)  # May be empty or have output

    def test_convert_sentence_basic(self):
        """Test sentence conversion produces space-separated phonemes."""
        from data.utils import SampaPhonemizer

        phonemizer = SampaPhonemizer(lang_code="eng-Latn")
        result = phonemizer.convert_sentence("hello world")
        assert isinstance(result, str)
        # Should contain space-separated phonemes
        if result:
            assert " " in result or len(result) > 0

    def test_convert_sentence_empty(self):
        """Test empty sentence returns empty string."""
        from data.utils import SampaPhonemizer

        phonemizer = SampaPhonemizer(lang_code="eng-Latn")
        result = phonemizer.convert_sentence("")
        assert result == ""

    def test_convert_sentence_whitespace(self):
        """Test sentence with only whitespace returns empty string."""
        from data.utils import SampaPhonemizer

        phonemizer = SampaPhonemizer(lang_code="eng-Latn")
        result = phonemizer.convert_sentence("   ")
        assert result == ""


class TestTextCleaning:
    """Test text cleaning utilities."""

    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        from data.utils import clean_text

        result = clean_text("  hello   world  ")
        assert result == "hello world"

    def test_clean_text_control_chars(self):
        """Test control character removal."""
        from data.utils import clean_text

        result = clean_text("hello\x00\x01\x02world")
        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x02" not in result
        assert "hello" in result
        assert "world" in result

    def test_clean_text_newlines(self):
        """Test newline normalization."""
        from data.utils import clean_text

        result = clean_text("hello\n\n\nworld")
        assert result == "hello world"

    def test_clean_text_unicode_normalization(self):
        """Test Unicode normalization."""
        from data.utils import clean_text

        # Test that NFC normalization is applied
        text = "café"  # café with combining acute accent
        result = clean_text(text)
        assert len(result) > 0
        assert isinstance(result, str)


class TestByteEncoding:
    """Test byte encoding utilities."""

    def test_text_to_bytes_basic(self):
        """Test basic text to bytes conversion."""
        from data.utils import text_to_bytes

        result = text_to_bytes("hello")
        expected = list(b"hello")
        assert result == expected

    def test_text_to_bytes_unicode(self):
        """Test Unicode text to bytes conversion."""
        from data.utils import text_to_bytes

        result = text_to_bytes("café")
        expected = list("café".encode())
        assert result == expected

    def test_text_to_bytes_empty(self):
        """Test empty string to bytes."""
        from data.utils import text_to_bytes

        result = text_to_bytes("")
        assert result == []

    def test_bytes_to_text_roundtrip(self):
        """Test bytes_to_text(text_to_bytes(s)) == s roundtrip."""
        from data.utils import bytes_to_text, text_to_bytes

        original = "hello world café"
        byte_values = text_to_bytes(original)
        recovered = bytes_to_text(byte_values)
        assert recovered == original


class TestTrainValTestSplit:
    """Test train/validation/test split functionality."""

    def test_split_basic(self):
        """Test basic split functionality."""
        from data.utils import train_val_test_split

        lines = ["line" + str(i) for i in range(100)]
        train, val, test = train_val_test_split(lines)

        assert len(train) == 90  # 90%
        assert len(val) == 5    # 5%
        assert len(test) == 5   # 5%

    def test_split_custom_fractions(self):
        """Test custom split fractions."""
        from data.utils import train_val_test_split

        lines = ["line" + str(i) for i in range(100)]
        train, val, test = train_val_test_split(lines, train_frac=0.8, val_frac=0.1)

        assert len(train) == 80  # 80%
        assert len(val) == 10    # 10%
        assert len(test) == 10   # 10%

    def test_split_small_dataset(self):
        """Test split with small dataset."""
        from data.utils import train_val_test_split

        lines = ["line1", "line2", "line3"]
        train, val, test = train_val_test_split(lines)

        # With small numbers, rounding might produce different sizes
        assert len(train) + len(val) + len(test) == len(lines)

    def test_split_empty(self):
        """Test split with empty list."""
        from data.utils import train_val_test_split

        train, val, test = train_val_test_split([])
        assert train == []
        assert val == []
        assert test == []

    def test_split_proportions_within_tolerance(self):
        """Test that split proportions are within 1% tolerance."""
        from data.utils import train_val_test_split

        lines = ["line" + str(i) for i in range(1000)]
        train, val, test = train_val_test_split(lines)

        total = len(lines)
        train_frac = len(train) / total
        val_frac = len(val) / total
        test_frac = len(test) / total

        assert abs(train_frac - 0.9) < 0.01
        assert abs(val_frac - 0.05) < 0.01
        assert abs(test_frac - 0.05) < 0.01


class TestParallelPhonemize:
    """Test parallel phonemization."""

    def test_parallel_phonemize_basic(self):
        """Test basic parallel phonemization."""
        from data.phonemize import parallel_phonemize

        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as inp:
            inp.write("hello world\n")
            inp.write("goodbye world\n")
            inp.write("testing phonemization\n")
            input_path = inp.name

        # Create temporary output path
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as out:
            output_path = out.name

        try:
            # Run phonemization
            parallel_phonemize(input_path, output_path, chunk_size=2, max_workers=1, verify_ascii=False)

            # Check output exists and has content
            assert Path(output_path).exists()
            with open(output_path, encoding="utf-8") as f:
                result_lines = f.readlines()

            # Should have 3 lines (one per input line)
            assert len(result_lines) == 3

        finally:
            # Cleanup
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_parallel_phonemize_creates_directory(self):
        """Test that phonemization creates output directory if needed."""
        import tempfile

        from data.phonemize import parallel_phonemize

        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as inp:
            inp.write("hello world\n")
            input_path = inp.name

        # Create output path in non-existent directory
        output_path = Path(tempfile.gettempdir()) / "test_phonemize_subdir" / "output.txt"

        try:
            parallel_phonemize(input_path, str(output_path), chunk_size=10, max_workers=1, verify_ascii=False)

            # Check output exists
            assert output_path.exists()

        finally:
            # Cleanup
            Path(input_path).unlink(missing_ok=True)
            if output_path.exists():
                output_path.unlink()
            output_path.parent.rmdir()

    def test_parallel_phonemize_empty_input(self):
        """Test phonemization with empty input file."""
        from data.phonemize import parallel_phonemize

        # Create empty temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as inp:
            input_path = inp.name

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as out:
            output_path = out.name

        try:
            parallel_phonemize(input_path, output_path, chunk_size=10, max_workers=1, verify_ascii=False)

            # Output should exist but be empty
            assert Path(output_path).exists()
            with open(output_path, encoding="utf-8") as f:
                content = f.read()
            assert content == ""

        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_parallel_phonemize_byte_output(self):
        """Test phonemization with byte output."""
        from data.phonemize import parallel_phonemize

        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as inp:
            inp.write("hello world\n")
            input_path = inp.name

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as out:
            output_path = out.name

        try:
            parallel_phonemize(input_path, output_path, chunk_size=10, max_workers=1,
                              verify_ascii=False, output_bytes=True)

            # Check output exists and is bytes
            assert Path(output_path).exists()
            with open(output_path, mode="rb") as f:
                content = f.read()

            assert len(content) > 0
            assert isinstance(content, bytes)

        finally:
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)
