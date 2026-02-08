"""
Deduplication using MinHash for training data curation.

Detects near-duplicate code samples to prevent training on redundant data.
Uses MinHash (locality-sensitive hashing) for efficient similarity estimation.

Why: Training on duplicate or near-duplicate data wastes compute and can cause:
1. Memorization - model regurgitates training data instead of generalizing
2. Bias - overrepresented samples skew the model's distribution
3. Evaluation leakage - duplicates between train/test sets inflate metrics

MinHash enables approximate similarity detection in O(1) time per comparison,
making it practical for large datasets. The algorithm:
1. Shingle the text into n-grams (e.g., 3-grams)
2. Hash each shingle with multiple hash functions
3. Keep the minimum hash value for each function -> MinHash signature
4. Estimate Jaccard similarity from signature overlap

The datasketch library provides an optimized implementation. If unavailable,
we fall back to a pure Python implementation that's correct but slower.
"""

import hashlib
from dataclasses import dataclass

# Try to import datasketch for optimized MinHash
try:
    from datasketch import MinHash as DatasketchMinHash
    from datasketch import MinHashLSH

    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False
    DatasketchMinHash = None
    MinHashLSH = None

__all__ = [
    "MinHashSignature",
    "MinHashDeduplicator",
    "DATASKETCH_AVAILABLE",
]


@dataclass
class MinHashSignature:
    """A MinHash signature for a text document.

    Contains the minimum hash values for each hash function, enabling
    fast similarity estimation between documents.

    Why: Storing the signature allows repeated comparisons without re-hashing.
    The signature size (num_perm) controls accuracy/speed tradeoff:
    - More permutations = more accurate similarity estimates
    - Fewer permutations = faster computation and less memory

    Typical values: 128 for speed, 256 for accuracy

    Attributes:
        signature: List of minimum hash values
        doc_id: Optional identifier for the document
        shingle_count: Number of unique shingles in the document
    """

    signature: list[int]
    doc_id: str = ""
    shingle_count: int = 0


class MinHashDeduplicator:
    """Deduplicator using MinHash for efficient near-duplicate detection.

    Computes MinHash signatures for documents and finds near-duplicates
    using locality-sensitive hashing (LSH).

    Why: Exact duplicate detection (hash matching) misses near-duplicates like:
    - Code with different variable names
    - Comments added/removed
    - Whitespace/formatting changes
    - Minor refactoring

    MinHash approximates Jaccard similarity (intersection/union of shingles),
    catching near-duplicates that exact hashing would miss.

    Usage:
        dedup = MinHashDeduplicator()

        # Build index from existing documents
        for doc_id, text in corpus:
            dedup.add_to_index(doc_id, text)

        # Check if new document is a duplicate
        duplicates = dedup.find_duplicates(new_text, threshold=0.8)
        if duplicates:
            print(f"Similar to: {duplicates}")

    Attributes:
        num_perm: Number of hash permutations (signature size)
        shingle_size: N-gram size for shingling (default 3)
        threshold: Default similarity threshold for duplicate detection
    """

    def __init__(
        self,
        num_perm: int = 128,
        shingle_size: int = 3,
        threshold: float = 0.8,
    ) -> None:
        """Initialize deduplicator.

        Why: Configurable parameters allow tuning for different use cases:
        - num_perm: Higher = more accurate, slower (128 is good default)
        - shingle_size: 3-5 works well for code; larger = fewer matches
        - threshold: 0.8 = ~80% similar; adjust based on false positive tolerance

        Args:
            num_perm: Number of hash permutations for MinHash
            shingle_size: Size of character n-grams for shingling
            threshold: Default similarity threshold for duplicate detection
        """
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.threshold = threshold

        # Index storage
        self._signatures: dict[str, MinHashSignature] = {}

        # LSH index for fast lookup (if datasketch available)
        if DATASKETCH_AVAILABLE:
            self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        else:
            self._lsh = None

    def compute_signature(self, text: str) -> MinHashSignature:
        """Compute MinHash signature for text.

        Creates character-level shingles and computes minimum hash values
        using multiple hash functions.

        Why: Shingling at the character level (rather than word level) better
        captures code structure where whitespace and symbols matter. The hash
        functions use salted MD5 for fast, uniform distribution.

        Args:
            text: Text to compute signature for

        Returns:
            MinHashSignature with computed hash values
        """
        # Create shingles (n-grams)
        shingles = self._create_shingles(text)

        if DATASKETCH_AVAILABLE:
            # Use optimized datasketch implementation
            mh = DatasketchMinHash(num_perm=self.num_perm)
            for shingle in shingles:
                mh.update(shingle.encode("utf-8"))
            return MinHashSignature(
                signature=list(mh.hashvalues),
                shingle_count=len(shingles),
            )
        else:
            # Pure Python fallback
            return self._compute_signature_pure_python(shingles)

    def _create_shingles(self, text: str) -> set[str]:
        """Create character n-gram shingles from text.

        Why: Character-level shingles work better for code than word-level:
        - Handles identifiers consistently (my_function vs myFunction)
        - Captures syntax patterns (brackets, operators)
        - Works across languages without tokenization

        Args:
            text: Text to shingle

        Returns:
            Set of unique shingles
        """
        # Normalize whitespace but preserve structure
        normalized = " ".join(text.split())

        if len(normalized) < self.shingle_size:
            return {normalized} if normalized else set()

        shingles: set[str] = set()
        for i in range(len(normalized) - self.shingle_size + 1):
            shingle = normalized[i : i + self.shingle_size]
            shingles.add(shingle)

        return shingles

    def _compute_signature_pure_python(self, shingles: set[str]) -> MinHashSignature:
        """Compute MinHash signature using pure Python.

        Fallback implementation when datasketch is not available. Uses salted
        MD5 hashing to simulate multiple hash functions.

        Why: Pure Python ensures the module works without external dependencies,
        though it's slower than the optimized C implementation in datasketch.

        Args:
            shingles: Set of text shingles

        Returns:
            MinHashSignature with computed values
        """
        if not shingles:
            # Empty document gets all-max signature
            return MinHashSignature(
                signature=[2**32 - 1] * self.num_perm,
                shingle_count=0,
            )

        # Simulate multiple hash functions using salted MD5
        signature: list[int] = []

        for i in range(self.num_perm):
            min_hash = 2**32 - 1  # Start with max value

            for shingle in shingles:
                # Hash shingle with salt (permutation index)
                hash_input = f"{i}:{shingle}".encode()
                hash_value = int(hashlib.md5(hash_input).hexdigest()[:8], 16)

                if hash_value < min_hash:
                    min_hash = hash_value

            signature.append(min_hash)

        return MinHashSignature(
            signature=signature,
            shingle_count=len(shingles),
        )

    def is_duplicate(
        self,
        sig1: MinHashSignature,
        sig2: MinHashSignature,
        threshold: float | None = None,
    ) -> bool:
        """Check if two signatures indicate duplicates.

        Uses Jaccard similarity estimation: count of matching hash values
        divided by total hash functions.

        Why: This is the core similarity check. The threshold determines
        how similar documents must be to be considered duplicates.
        0.8 = ~80% similar content.

        Args:
            sig1: First signature
            sig2: Second signature
            threshold: Similarity threshold (uses default if None)

        Returns:
            True if estimated similarity >= threshold
        """
        threshold = threshold if threshold is not None else self.threshold

        similarity = self.estimate_similarity(sig1, sig2)
        return similarity >= threshold

    def estimate_similarity(
        self,
        sig1: MinHashSignature,
        sig2: MinHashSignature,
    ) -> float:
        """Estimate Jaccard similarity from signatures.

        Why: The fraction of matching hash values approximates Jaccard similarity
        (intersection/union). Error decreases as num_perm increases.

        Args:
            sig1: First signature
            sig2: Second signature

        Returns:
            Estimated Jaccard similarity [0.0, 1.0]
        """
        if len(sig1.signature) != len(sig2.signature):
            raise ValueError(
                f"Signature lengths must match: {len(sig1.signature)} != {len(sig2.signature)}"
            )

        if not sig1.signature:
            return 0.0

        matches = sum(1 for h1, h2 in zip(sig1.signature, sig2.signature, strict=True) if h1 == h2)
        return matches / len(sig1.signature)

    def add_to_index(self, doc_id: str, text: str) -> MinHashSignature:
        """Add document to deduplication index.

        Computes signature and stores it for future duplicate checking.
        If datasketch LSH is available, also adds to LSH index for fast lookup.

        Why: Building an index enables O(1) duplicate detection instead of
        O(n) comparison against all documents.

        Args:
            doc_id: Unique identifier for the document
            text: Document text

        Returns:
            Computed MinHash signature
        """
        signature = self.compute_signature(text)
        signature.doc_id = doc_id

        self._signatures[doc_id] = signature

        # Add to LSH index if available
        if DATASKETCH_AVAILABLE and self._lsh is not None:
            mh = DatasketchMinHash(num_perm=self.num_perm)
            for shingle in self._create_shingles(text):
                mh.update(shingle.encode("utf-8"))
            self._lsh.insert(doc_id, mh)

        return signature

    def find_duplicates(
        self,
        text: str,
        threshold: float | None = None,
    ) -> list[str]:
        """Find documents in index that are near-duplicates of text.

        Uses LSH for fast candidate retrieval when available, falling back
        to linear scan of all signatures.

        Why: LSH provides sub-linear lookup time by only checking documents
        that hash to the same buckets. Without LSH, we compare against all
        indexed documents.

        Args:
            text: Text to check for duplicates
            threshold: Similarity threshold (uses default if None)

        Returns:
            List of doc_ids that are near-duplicates
        """
        threshold = threshold if threshold is not None else self.threshold

        query_sig = self.compute_signature(text)
        duplicates: list[str] = []

        if DATASKETCH_AVAILABLE and self._lsh is not None:
            # Use LSH for fast candidate retrieval
            mh = DatasketchMinHash(num_perm=self.num_perm)
            for shingle in self._create_shingles(text):
                mh.update(shingle.encode("utf-8"))

            # LSH query returns candidates above threshold
            candidates = self._lsh.query(mh)
            duplicates = list(candidates)

        else:
            # Linear scan fallback
            for doc_id, stored_sig in self._signatures.items():
                if self.is_duplicate(query_sig, stored_sig, threshold):
                    duplicates.append(doc_id)

        return duplicates

    def get_index_size(self) -> int:
        """Get number of documents in index.

        Why: Useful for monitoring index size and memory usage.

        Returns:
            Number of indexed documents
        """
        return len(self._signatures)

    def clear_index(self) -> None:
        """Clear all documents from index.

        Why: Enables reuse of deduplicator for multiple dataset passes
        without recreating the object.
        """
        self._signatures.clear()

        if DATASKETCH_AVAILABLE:
            self._lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
