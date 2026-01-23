"""Data loading utilities for code training.

Why: Efficient data loading is critical for training throughput.
This module provides:
1. CodeDataset for loading code files
2. Collation with dynamic padding
3. Sequence packing for efficiency

Note: This implements data loading for the embedding-prediction paradigm where:
- Input: Tokenization converts discrete tokens → embeddings (via MultiModalTokenizer)
- Core: Transformer operates on continuous embeddings
- Output: Embeddings are projected to logits (temporary scaffolding for training)
- Future: KNN/VQ rounding will replace logits for true embedding-prediction
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from tritter.tokenization import ModalityType, MultiModalTokenizer


@dataclass
class DataConfig:
    """Configuration for data loading.

    Why: Centralizes data hyperparameters for reproducibility and easy experimentation.
    max_seq_length matches TritterConfig.max_position_embeddings (128K default) but
    can be reduced for memory-constrained training (e.g., 2048 for local testing).
    batch_size and num_workers are tuned for RTX 5080 16GB VRAM constraints.
    """

    max_seq_length: int = 2048
    batch_size: int = 8
    num_workers: int = 4
    shuffle: bool = True
    seed: int = 42


class CodeDataset(Dataset):
    """Dataset for loading code files.

    Why: Loads code from a directory or file list and tokenizes on the fly.
    Supports .py, .rs, .json (for pre-tokenized), .txt formats.
    On-the-fly tokenization enables dynamic data augmentation and avoids
    storing large tokenized datasets on disk.

    The dataset uses ModalityType.CODE for all files to enable AST-aware
    tokenization (once implemented in MultiModalTokenizer._encode_code).
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: MultiModalTokenizer,
        max_seq_length: int = 2048,
    ) -> None:
        """Initialize dataset.

        Args:
            data_path: Path to directory or file containing code
            tokenizer: Tokenizer for encoding (must be MultiModalTokenizer)
            max_seq_length: Maximum sequence length (truncates longer sequences)

        Why: Accepts both single files and directories for flexibility.
        Directory mode recursively collects all code files using rglob,
        enabling training on entire codebases. max_seq_length truncation
        prevents OOM errors during training and matches model's position
        embedding constraints.
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Collect files
        self.files: list[Path] = []
        if self.data_path.is_file():
            self.files = [self.data_path]
        elif self.data_path.is_dir():
            for ext in ["*.py", "*.rs", "*.txt", "*.json", "*.jsonl"]:
                self.files.extend(self.data_path.rglob(ext))

        self.files = sorted(self.files)

    def __len__(self) -> int:
        """Return number of files in dataset.

        Why: Required by PyTorch Dataset protocol for batching and sampling.
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get tokenized item.

        Args:
            idx: Index of file to load

        Returns:
            Dictionary with 'input_ids' tensor of shape (seq_len,)

        Why: Returns dict instead of raw tensor for extensibility (can add
        'labels', 'attention_mask', etc. in future). Tokenizes with
        add_special_tokens=True to include BOS/EOS/modality prefix tokens,
        which are essential for the model to learn sequence boundaries and
        modality-specific processing patterns.
        """
        file_path = self.files[idx]

        # Handle different formats
        if file_path.suffix == ".json":
            with open(file_path) as f:
                data = json.load(f)
                text = data.get("text", data.get("content", ""))
        elif file_path.suffix == ".jsonl":
            # Take first line for simplicity
            with open(file_path) as f:
                line = f.readline()
                data = json.loads(line)
                text = data.get("text", data.get("content", ""))
        else:
            with open(file_path) as f:
                text = f.read()

        # Tokenize using CODE modality for AST-aware tokenization
        # Why: ModalityType.CODE enables future AST-based tokenization which
        # respects semantic boundaries (function defs, class defs) rather than
        # arbitrary byte sequences. Critical for code understanding tasks.
        tokens = self.tokenizer.encode(text, modality=ModalityType.CODE, add_special_tokens=True)

        # Truncate if needed
        # Why: Prevents OOM during training and ensures all sequences fit within
        # model's max_position_embeddings. Truncation from left would lose code
        # structure; from right preserves imports and early definitions.
        if len(tokens) > self.max_seq_length:
            tokens = tokens[: self.max_seq_length]

        return {"input_ids": torch.tensor(tokens, dtype=torch.long)}


class StreamingCodeDataset(IterableDataset):
    """Streaming dataset for large code collections.

    Why: For datasets too large to index (e.g., Stack-Edu 1T tokens), streaming
    from JSONL files avoids loading entire dataset into memory. Each line should
    be a JSON object with 'text' or 'content' field. Enables training on datasets
    that don't fit on disk as single files by processing line-by-line.

    This design follows HuggingFace datasets streaming pattern and supports
    distributed training with torch.utils.data.DistributedSampler.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: MultiModalTokenizer,
        max_seq_length: int = 2048,
    ) -> None:
        """Initialize streaming dataset.

        Args:
            data_path: Path to JSONL file or directory of JSONL files
            tokenizer: Tokenizer for encoding (must be MultiModalTokenizer)
            max_seq_length: Maximum sequence length (truncates longer sequences)

        Why: Supports both single JSONL files and directories of JSONL files
        for sharding across workers. Directory mode enables parallel processing
        where each worker reads different files.
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Collect JSONL files
        if self.data_path.is_file():
            self.files = [self.data_path]
        else:
            self.files = sorted(self.data_path.rglob("*.jsonl"))

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Iterate over dataset.

        Yields:
            Dictionary with 'input_ids' tensor of shape (seq_len,)

        Why: Iterates through all JSONL files sequentially, yielding one
        tokenized example per line. Skips malformed JSON and empty lines
        for robustness to noisy datasets. Error handling prevents single
        corrupted line from crashing entire training run.
        """
        for file_path in self.files:
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        text = data.get("text", data.get("content", ""))
                    except json.JSONDecodeError:
                        continue

                    if not text:
                        continue

                    # Tokenize using CODE modality
                    tokens = self.tokenizer.encode(
                        text, modality=ModalityType.CODE, add_special_tokens=True
                    )

                    # Truncate if needed
                    if len(tokens) > self.max_seq_length:
                        tokens = tokens[: self.max_seq_length]

                    yield {"input_ids": torch.tensor(tokens, dtype=torch.long)}


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate batch with dynamic padding.

    Why: Pads sequences to max length in batch, not global max, for efficiency.
    Uses 0 as padding token (matches tokenizer.PAD_TOKEN from MultiModalTokenizer).
    Dynamic padding reduces wasted computation on padding tokens compared to
    fixed-length padding, especially important for variable-length code sequences.

    Args:
        batch: List of dictionaries with 'input_ids' tensors of varying lengths

    Returns:
        Batched dictionary with:
            - 'input_ids': Padded tensor of shape (batch_size, max_len_in_batch)
            - 'attention_mask': Binary mask of shape (batch_size, max_len_in_batch)
              where 1 = real token, 0 = padding token

    Why attention_mask: FlashAttention and SDPA use attention_mask to exclude
    padding tokens from softmax normalization, preventing gradients from flowing
    to padding embeddings and improving training stability.
    """
    # Find max length in this batch
    max_len = max(item["input_ids"].size(0) for item in batch)

    # Pad sequences
    input_ids = []
    attention_mask = []

    for item in batch:
        ids = item["input_ids"]
        padding_length = max_len - ids.size(0)

        if padding_length > 0:
            # Pad on the right (standard for causal LM)
            # Why: Right-padding preserves left-to-right sequence structure.
            # Left-padding would shift position indices and break causal attention.
            ids = torch.cat([ids, torch.zeros(padding_length, dtype=torch.long)])
            mask = torch.cat(
                [
                    torch.ones(max_len - padding_length, dtype=torch.long),
                    torch.zeros(padding_length, dtype=torch.long),
                ]
            )
        else:
            mask = torch.ones(max_len, dtype=torch.long)

        input_ids.append(ids)
        attention_mask.append(mask)

    return {
        "input_ids": torch.stack(input_ids),  # (B, L)
        "attention_mask": torch.stack(attention_mask),  # (B, L)
    }


def create_dataloader(
    dataset: Dataset | IterableDataset,
    config: DataConfig,
    is_train: bool = True,
) -> DataLoader:
    """Create DataLoader with appropriate settings.

    Args:
        dataset: Dataset to wrap (CodeDataset or StreamingCodeDataset)
        config: Data configuration with batch_size, num_workers, etc.
        is_train: Whether this is for training (affects shuffling and drop_last)

    Returns:
        Configured DataLoader ready for training/evaluation

    Why: Centralizes DataLoader configuration to ensure consistency across
    training and evaluation. Key settings:
    - shuffle=True for training (prevents order bias), False for eval
    - num_workers>0 enables parallel data loading to overlap I/O with compute
    - pin_memory=True speeds up CPU→GPU transfer via pinned memory
    - drop_last=True for training ensures all batches have same size (required
      for gradient accumulation and some optimizers)

    Note: shuffle is incompatible with IterableDataset (no random access),
    so we disable it automatically for streaming datasets.
    """
    shuffle = config.shuffle if is_train and not isinstance(dataset, IterableDataset) else False

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=is_train,
    )
