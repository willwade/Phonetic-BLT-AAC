"""Test data download and filtering functionality."""

from unittest.mock import MagicMock, patch

import pytest


class TestDownloadAndFilter:
    """Test the main download_and_filter function."""

    @patch("data.download.load_dataset")
    @patch("data.download.Path")
    def test_filters_rows_below_threshold(self, mock_path, mock_load_dataset):
        """Test that rows below threshold are dropped."""
        from data.download import download_and_filter

        # Mock dataset with mixed scores
        mock_rows = [
            {"text": "High score text", "aac_score": 0.9},
            {"text": "Low score text", "aac_score": 0.7},
            {"text": "Medium score text", "aac_score": 0.85},
        ]
        mock_dataset = iter(mock_rows)
        mock_load_dataset.return_value = mock_dataset

        # Mock file operations
        mock_file = MagicMock()
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_instance.open.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_path_instance.open.return_value.__exit__ = MagicMock(return_value=False)
        mock_path.return_value = mock_path_instance

        download_and_filter(score_threshold=0.85, output_path="test.txt", resume=False, deduplicate=False)

        # Should write only 2 lines (0.9 and 0.85)
        assert mock_file.write.call_count == 2

    @patch("data.download.load_dataset")
    @patch("data.download.Path")
    def test_handles_missing_aac_score(self, mock_path, mock_load_dataset):
        """Test that rows missing aac_score are handled with default 0.0."""
        from data.download import download_and_filter

        mock_rows = [
            {"text": "No score text", "not_aac_score": 0.9},  # Missing aac_score
            {"text": "Has score", "aac_score": 0.9},
        ]
        mock_dataset = iter(mock_rows)
        mock_load_dataset.return_value = mock_dataset

        mock_file = MagicMock()
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_instance.open.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_path_instance.open.return_value.__exit__ = MagicMock(return_value=False)
        mock_path.return_value = mock_path_instance

        download_and_filter(score_threshold=0.5, output_path="test.txt", resume=False, deduplicate=False)

        # Should write only 1 line (missing score defaults to 0.0, below threshold)
        assert mock_file.write.call_count == 1

    @patch("data.download.load_dataset")
    @patch("data.download.Path")
    def test_clean_text_operations(self, mock_path, mock_load_dataset):
        """Test that text is cleaned properly."""
        from data.download import download_and_filter

        mock_rows = [
            {"text": "  text with  spaces  and\nnewlines  ", "aac_score": 0.9},
        ]
        mock_dataset = iter(mock_rows)
        mock_load_dataset.return_value = mock_dataset

        mock_file = MagicMock()
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_instance.open.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_path_instance.open.return_value.__exit__ = MagicMock(return_value=False)
        mock_path.return_value = mock_path_instance

        download_and_filter(score_threshold=0.85, output_path="test.txt", resume=False, deduplicate=False)

        # Check that internal newlines are replaced with spaces and text is stripped
        written_arg = mock_file.write.call_args[0][0]
        assert written_arg.count("\n") == 1  # Only the final newline should remain
        assert written_arg == "text with  spaces  and newlines\n"

    @patch("data.download.load_dataset")
    @patch("data.download.Path")
    def test_resumes_skips_existing_lines(self, mock_path, mock_load_dataset):
        """Test that resume skips lines already in output file."""
        from data.download import download_and_filter

        mock_rows = [
            {"text": "Line 1", "aac_score": 0.9},
            {"text": "Line 2", "aac_score": 0.9},
            {"text": "Line 3", "aac_score": 0.9},
        ]
        mock_dataset = iter(mock_rows)
        mock_load_dataset.return_value = mock_dataset

        mock_file = MagicMock()
        mock_path_instance = MagicMock()
        # Simulate existing file with 1 line
        mock_path_instance.exists.return_value = True
        mock_path_instance.open.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_path_instance.open.return_value.__exit__ = MagicMock(return_value=False)

        # Mock the line counting
        mock_path_instance.open.return_value.__iter__ = MagicMock(return_value=iter(["existing line\n"]))

        mock_path.return_value = mock_path_instance

        download_and_filter(score_threshold=0.85, output_path="test.txt", resume=True, deduplicate=False)

        # Should skip 1 line and write 2 lines
        assert mock_file.write.call_count == 2

    @patch("data.download.load_dataset")
    @patch("data.download.Path")
    def test_deduplication_removes_exact_duplicates(self, mock_path, mock_load_dataset):
        """Test that exact duplicates are removed when deduplicate=True."""
        from data.download import download_and_filter

        mock_rows = [
            {"text": "Duplicate text", "aac_score": 0.9},
            {"text": "Unique text", "aac_score": 0.9},
            {"text": "Duplicate text", "aac_score": 0.9},  # Exact duplicate
        ]
        mock_dataset = iter(mock_rows)
        mock_load_dataset.return_value = mock_dataset

        mock_file = MagicMock()
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path_instance.open.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_path_instance.open.return_value.__exit__ = MagicMock(return_value=False)
        mock_path.return_value = mock_path_instance

        download_and_filter(score_threshold=0.85, output_path="test.txt", resume=False, deduplicate=True)

        # Should write only 2 unique lines
        assert mock_file.write.call_count == 2

    def test_invalid_source_raises_error(self):
        """Test that invalid source raises ValueError."""
        from data.download import download_and_filter

        with pytest.raises(ValueError, match="Invalid source"):
            download_and_filter(source="invalid_source")

    def test_valid_sources(self):
        """Test that valid sources are accepted."""
        from data.download import DATASET_SOURCES

        assert "c4" in DATASET_SOURCES
        assert "subtitles" in DATASET_SOURCES
        assert "c4-fast" in DATASET_SOURCES
