"""Unit tests for training data loading utilities.

Why: Validates that CodeDataset and StreamingCodeDataset correctly load and tokenize
code files, and that collation produces properly padded batches. Critical for ensuring
training pipeline handles variable-length sequences correctly.
"""

import json

import pytest
import torch

from tritter.tokenization import MultiModalTokenizer
from tritter.training.data import (
    CodeDataset,
    DataConfig,
    StreamingCodeDataset,
    collate_fn,
    create_dataloader,
)


@pytest.fixture
def tokenizer():
    """Create a MultiModalTokenizer for testing.

    Why: Tests need a tokenizer instance to pass to datasets. Uses default
    vocab_size=65536 which is sufficient for byte-level encoding.
    """
    return MultiModalTokenizer(vocab_size=65536, max_length=2048)


@pytest.fixture
def temp_code_dir(tmp_path):
    """Create a temporary directory with sample code files.

    Why: Provides realistic test data for CodeDataset without requiring
    external fixtures. Creates multiple file types (.py, .rs, .jsonl) to
    test different loading paths.

    Returns:
        Path to temporary directory containing:
            - sample.py: Python code
            - sample.rs: Rust code
            - sample.txt: Plain text
            - data.jsonl: JSONL with code samples
    """
    # Create Python file
    py_file = tmp_path / "sample.py"
    py_file.write_text("def hello():\n    print('world')\n")

    # Create Rust file
    rs_file = tmp_path / "sample.rs"
    rs_file.write_text('fn main() {\n    println!("Hello, world!");\n}\n')

    # Create text file
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("This is a text file for testing.\n")

    # Create JSONL file
    jsonl_file = tmp_path / "data.jsonl"
    with open(jsonl_file, "w") as f:
        f.write(json.dumps({"text": "Sample code line 1"}) + "\n")
        f.write(json.dumps({"content": "Sample code line 2"}) + "\n")
        f.write(json.dumps({"text": "Sample code line 3"}) + "\n")

    return tmp_path


class TestDataConfig:
    """Test suite for DataConfig dataclass.

    Why: Validates that DataConfig correctly stores hyperparameters and provides
    sensible defaults for training data loading.
    """

    def test_default_values(self):
        """Test DataConfig uses correct default values.

        Why: Ensures default hyperparameters match project requirements
        (2048 seq length for local testing, batch_size=8 for 16GB VRAM).
        """
        config = DataConfig()
        assert config.max_seq_length == 2048
        assert config.batch_size == 8
        assert config.num_workers == 4
        assert config.shuffle is True
        assert config.seed == 42

    def test_custom_values(self):
        """Test DataConfig accepts custom values.

        Why: Validates that users can override defaults for experimentation
        (e.g., larger batch_size for bigger GPUs, longer sequences for eval).
        """
        config = DataConfig(
            max_seq_length=4096,
            batch_size=16,
            num_workers=8,
            shuffle=False,
            seed=123,
        )
        assert config.max_seq_length == 4096
        assert config.batch_size == 16
        assert config.num_workers == 8
        assert config.shuffle is False
        assert config.seed == 123


class TestCodeDataset:
    """Test suite for CodeDataset.

    Why: Validates that CodeDataset correctly loads files, tokenizes them,
    and handles edge cases (truncation, different file formats).
    """

    def test_load_directory(self, temp_code_dir, tokenizer):
        """Test CodeDataset loads all files from directory.

        Why: Directory mode is the primary use case for training on codebases.
        Must recursively find all supported file types (.py, .rs, .txt, .jsonl).
        """
        dataset = CodeDataset(temp_code_dir, tokenizer, max_seq_length=2048)
        # Should find: sample.py, sample.rs, sample.txt, data.jsonl
        assert len(dataset) == 4

    def test_load_single_file(self, temp_code_dir, tokenizer):
        """Test CodeDataset loads single file.

        Why: Single-file mode is useful for testing or fine-tuning on specific files.
        """
        py_file = temp_code_dir / "sample.py"
        dataset = CodeDataset(py_file, tokenizer, max_seq_length=2048)
        assert len(dataset) == 1

    def test_tokenize_python_file(self, temp_code_dir, tokenizer):
        """Test CodeDataset correctly tokenizes Python file.

        Why: Validates that code is tokenized with special tokens (BOS, EOS,
        CODE prefix) and returns proper tensor format.
        """
        py_file = temp_code_dir / "sample.py"
        dataset = CodeDataset(py_file, tokenizer, max_seq_length=2048)
        item = dataset[0]

        # Check structure
        assert "input_ids" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert item["input_ids"].dtype == torch.long

        # Check tokens are reasonable (BOS + CODE_PREFIX + content + EOS)
        # Minimum: 4 tokens (BOS, CODE_PREFIX, at least 1 content byte, EOS)
        assert len(item["input_ids"]) >= 4

        # First token should be BOS (1), second should be CODE_PREFIX (5)
        assert item["input_ids"][0] == tokenizer.special_tokens[tokenizer.BOS_TOKEN]
        assert item["input_ids"][1] == tokenizer.special_tokens[tokenizer.CODE_PREFIX]

    def test_tokenize_jsonl_file(self, temp_code_dir, tokenizer):
        """Test CodeDataset tokenizes first line of JSONL file.

        Why: JSONL files may contain multiple samples, but CodeDataset treats
        each file as single item (takes first line only).
        """
        jsonl_file = temp_code_dir / "data.jsonl"
        dataset = CodeDataset(jsonl_file, tokenizer, max_seq_length=2048)
        item = dataset[0]

        # Should tokenize first line: "Sample code line 1"
        assert "input_ids" in item
        assert len(item["input_ids"]) >= 4  # BOS + PREFIX + content + EOS

    def test_truncation(self, temp_code_dir, tokenizer):
        """Test CodeDataset truncates sequences longer than max_seq_length.

        Why: Truncation is critical for preventing OOM during training.
        Must respect max_seq_length constraint.
        """
        # Create file with long content
        long_file = temp_code_dir / "long.py"
        long_content = "x = 1\n" * 1000  # Repeat to create long file
        long_file.write_text(long_content)

        # Use small max_seq_length to force truncation
        dataset = CodeDataset(temp_code_dir, tokenizer, max_seq_length=50)

        # Find the long file (dataset is sorted)
        for i in range(len(dataset)):
            item = dataset[i]
            # All sequences must be <= max_seq_length
            assert len(item["input_ids"]) <= 50


class TestStreamingCodeDataset:
    """Test suite for StreamingCodeDataset.

    Why: Validates streaming dataset correctly iterates through JSONL files
    without loading entire dataset into memory.
    """

    def test_iterate_single_jsonl(self, temp_code_dir, tokenizer):
        """Test StreamingCodeDataset iterates through single JSONL file.

        Why: Single-file streaming is common for large preprocessed datasets.
        Should yield one item per line.
        """
        jsonl_file = temp_code_dir / "data.jsonl"
        dataset = StreamingCodeDataset(jsonl_file, tokenizer, max_seq_length=2048)

        items = list(dataset)
        # data.jsonl has 3 lines
        assert len(items) == 3

        # Each item should have input_ids
        for item in items:
            assert "input_ids" in item
            assert isinstance(item["input_ids"], torch.Tensor)
            assert item["input_ids"].dtype == torch.long

    def test_iterate_directory(self, temp_code_dir, tokenizer):
        """Test StreamingCodeDataset iterates through all JSONL files in directory.

        Why: Directory mode enables sharding across workers. Should find all
        .jsonl files and yield items from each.
        """
        # Create second JSONL file
        jsonl_file2 = temp_code_dir / "data2.jsonl"
        with open(jsonl_file2, "w") as f:
            f.write(json.dumps({"text": "Another sample"}) + "\n")

        dataset = StreamingCodeDataset(temp_code_dir, tokenizer, max_seq_length=2048)
        items = list(dataset)

        # Should have 3 items from data.jsonl + 1 from data2.jsonl = 4 total
        assert len(items) == 4

    def test_skip_malformed_json(self, tmp_path, tokenizer):
        """Test StreamingCodeDataset skips malformed JSON lines.

        Why: Robustness to noisy data is critical for large-scale training.
        Malformed lines should be skipped without crashing.
        """
        jsonl_file = tmp_path / "malformed.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps({"text": "Valid line 1"}) + "\n")
            f.write("This is not valid JSON\n")  # Malformed
            f.write(json.dumps({"text": "Valid line 2"}) + "\n")
            f.write("\n")  # Empty line

        dataset = StreamingCodeDataset(jsonl_file, tokenizer, max_seq_length=2048)
        items = list(dataset)

        # Should only yield 2 valid items
        assert len(items) == 2

    def test_truncation(self, tmp_path, tokenizer):
        """Test StreamingCodeDataset truncates long sequences.

        Why: Same truncation behavior as CodeDataset - ensures memory safety.
        """
        jsonl_file = tmp_path / "long.jsonl"
        long_text = "x" * 10000  # Very long text
        with open(jsonl_file, "w") as f:
            f.write(json.dumps({"text": long_text}) + "\n")

        dataset = StreamingCodeDataset(jsonl_file, tokenizer, max_seq_length=100)
        items = list(dataset)

        assert len(items) == 1
        assert len(items[0]["input_ids"]) <= 100


class TestCollateFn:
    """Test suite for collate_fn.

    Why: Validates that collation correctly pads sequences to batch max length
    and creates proper attention masks.
    """

    def test_pad_to_max_length(self, tokenizer):
        """Test collate_fn pads sequences to max length in batch.

        Why: Dynamic padding reduces wasted computation vs fixed-length padding.
        """
        # Create batch with varying lengths
        batch = [
            {"input_ids": torch.tensor([1, 2, 3])},  # Length 3
            {"input_ids": torch.tensor([1, 2, 3, 4, 5])},  # Length 5
            {"input_ids": torch.tensor([1, 2])},  # Length 2
        ]

        result = collate_fn(batch)

        # Should pad to max length in batch (5)
        assert result["input_ids"].shape == (3, 5)
        assert result["attention_mask"].shape == (3, 5)

    def test_attention_mask_correct(self, tokenizer):
        """Test collate_fn creates correct attention mask.

        Why: Attention mask must be 1 for real tokens, 0 for padding.
        Critical for preventing gradients from flowing to padding tokens.
        """
        batch = [
            {"input_ids": torch.tensor([1, 2, 3])},  # Length 3
            {"input_ids": torch.tensor([1, 2, 3, 4, 5])},  # Length 5
        ]

        result = collate_fn(batch)

        # First sequence: 3 real tokens, 2 padding
        expected_mask_0 = torch.tensor([1, 1, 1, 0, 0], dtype=torch.long)
        assert torch.all(result["attention_mask"][0] == expected_mask_0)

        # Second sequence: 5 real tokens, 0 padding
        expected_mask_1 = torch.tensor([1, 1, 1, 1, 1], dtype=torch.long)
        assert torch.all(result["attention_mask"][1] == expected_mask_1)

    def test_padding_token_is_zero(self, tokenizer):
        """Test collate_fn pads with token ID 0 (PAD_TOKEN).

        Why: Must match tokenizer.PAD_TOKEN (0) to maintain consistency.
        Embedding layer has padding_idx=0 to prevent gradients.
        """
        batch = [
            {"input_ids": torch.tensor([1, 2, 3])},  # Length 3
            {"input_ids": torch.tensor([1, 2, 3, 4, 5])},  # Length 5
        ]

        result = collate_fn(batch)

        # First sequence should have padding tokens (0) at positions 3, 4
        assert result["input_ids"][0, 3] == 0
        assert result["input_ids"][0, 4] == 0

    def test_no_padding_when_equal_length(self, tokenizer):
        """Test collate_fn handles equal-length sequences correctly.

        Why: When all sequences have same length, no padding needed.
        Attention mask should be all 1s.
        """
        batch = [
            {"input_ids": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5, 6])},
        ]

        result = collate_fn(batch)

        # No padding needed, all masks should be 1
        assert torch.all(result["attention_mask"] == 1)


class TestCreateDataloader:
    """Test suite for create_dataloader.

    Why: Validates DataLoader factory creates loaders with correct configuration
    for training and evaluation.
    """

    def test_create_train_dataloader(self, temp_code_dir, tokenizer):
        """Test create_dataloader creates training DataLoader with correct settings.

        Why: Training loaders need shuffle=True, drop_last=True, pin_memory=True
        for optimal throughput and gradient accumulation compatibility.
        """
        dataset = CodeDataset(temp_code_dir, tokenizer, max_seq_length=2048)
        config = DataConfig(batch_size=2, num_workers=0, shuffle=True)

        loader = create_dataloader(dataset, config, is_train=True)

        assert loader.batch_size == 2
        # Note: shuffle is internal state, hard to check directly
        assert loader.pin_memory is True
        assert loader.drop_last is True

    def test_create_eval_dataloader(self, temp_code_dir, tokenizer):
        """Test create_dataloader creates evaluation DataLoader with correct settings.

        Why: Eval loaders need shuffle=False (deterministic order), drop_last=False
        (evaluate on all data).
        """
        dataset = CodeDataset(temp_code_dir, tokenizer, max_seq_length=2048)
        config = DataConfig(batch_size=2, num_workers=0, shuffle=True)

        loader = create_dataloader(dataset, config, is_train=False)

        assert loader.batch_size == 2
        assert loader.pin_memory is True
        assert loader.drop_last is False  # Eval should not drop last batch

    def test_streaming_dataset_no_shuffle(self, temp_code_dir, tokenizer):
        """Test create_dataloader disables shuffle for IterableDataset.

        Why: IterableDataset has no random access, so shuffle is incompatible.
        Must disable automatically to prevent errors.
        """
        jsonl_file = temp_code_dir / "data.jsonl"
        dataset = StreamingCodeDataset(jsonl_file, tokenizer, max_seq_length=2048)
        config = DataConfig(batch_size=2, num_workers=0, shuffle=True)

        # Should not raise error even though config.shuffle=True
        loader = create_dataloader(dataset, config, is_train=True)
        assert loader.batch_size == 2

    def test_batching_works(self, temp_code_dir, tokenizer):
        """Test create_dataloader produces proper batches.

        Why: End-to-end test that DataLoader + collate_fn work together
        to produce correctly shaped batches.
        """
        dataset = CodeDataset(temp_code_dir, tokenizer, max_seq_length=2048)
        config = DataConfig(batch_size=2, num_workers=0, shuffle=False)

        loader = create_dataloader(dataset, config, is_train=False)

        # Get first batch
        batch = next(iter(loader))

        assert "input_ids" in batch
        assert "attention_mask" in batch
        # Batch size should be 2 (or less if dataset < 2)
        assert batch["input_ids"].shape[0] <= 2
        assert batch["attention_mask"].shape[0] <= 2
        # Sequence length dimension should match
        assert batch["input_ids"].shape[1] == batch["attention_mask"].shape[1]
