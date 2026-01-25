# Embedding-Prediction Paradigm Specification

**Spec ID**: SPEC-003
**Status**: Draft
**Author**: Claude (Agentic Development)
**Created**: 2026-01-22
**Target Modules**: Core architectural paradigm affecting all modules

## 1. Overview

### 1.1 Purpose

This specification defines Tritter's core architectural paradigm: **embedding prediction**. Unlike standard language models that predict discrete token distributions, Tritter operates in continuous embedding space throughout its computation, treating tokenization as entry/exit points rather than the core computation medium.

### 1.2 Paradigm Summary

```
Standard LM:     tokens → embed → transformer → logits → tokens
Tritter:         tokens → embed → transformer → embeddings → (KNN/VQ) → tokens
                          ↑______________________↓
                          Continuous embedding space
```

### 1.3 Key Insight

The model's "reasoning" happens in continuous embedding space, not discrete token space. Token prediction is **temporary scaffolding** for training compatibility—production inference will use embedding rounding (KNN lookup, vector quantization) to convert continuous outputs to discrete tokens only when necessary.

### 1.4 Research Foundations

| Paper | Key Contribution | Relevance to Tritter |
|-------|------------------|----------------------|
| [Coconut](https://arxiv.org/abs/2412.06769) | Chain of Continuous Thought | Latent reasoning without tokens |
| [LCM](https://arxiv.org/abs/2412.08821) | Large Concept Models | Sentence-level embeddings |
| [Hrrformer](https://arxiv.org/abs/2305.19534) | Holographic attention | O(TH log H) complexity |

---

## 2. Architecture Principles

### 2.1 Entry Point: Tokenization → Embedding

**Location**: `src/tritter/tokenization/multimodal.py`

```
Input:  "def hello():"     (discrete tokens)
        ↓
        [101, 2532, 1234]  (token IDs)
        ↓
        [[0.1, 0.2, ...],  (continuous embeddings)
         [0.3, 0.1, ...],
         [0.4, 0.5, ...]]
```

**Principle**: Tokenization is the **only** place where discrete symbols enter the model. After embedding lookup, all computation operates on continuous vectors.

**Design Implications**:
- Tokenizer must support symmetric encode/decode
- Embedding dimension is the model's "native" representation
- Special tokens (PAD, BOS, EOS) have learned embeddings

### 2.2 Core Computation: Embedding Space Operations

**Location**: `src/tritter/models/architecture.py`

```
Embeddings → [Transformer Layers] → Embeddings
             ↓                 ↓
        No token discretization between layers
```

**Principle**: Transformer layers operate **only** on continuous embeddings. Each layer transforms embeddings to embeddings without any intermediate tokenization.

**Design Implications**:
- Hidden states are always continuous vectors
- Attention computes similarities in embedding space
- FFN transforms embeddings non-linearly
- Residual connections add embeddings (continuous addition)

### 2.3 Exit Point: Embedding → Token (When Needed)

**Location**: `src/tritter/inference/__init__.py` (future)

**Current (Training)**:
```
Embeddings → Linear projection → Logits → Cross-entropy loss
```

**Target (Production)**:
```
Embeddings → KNN/VQ rounding → Tokens (only when outputting text)
```

**Principle**: Token prediction is a **training convenience**, not the model's core operation. Production inference can maintain embeddings and only discretize at output boundaries.

---

## 3. Training Architecture

### 3.1 Current Implementation (Token Prediction Scaffolding)

```python
class TritterModel(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with token prediction head.

        Why: Standard cross-entropy training requires logits over vocabulary.
        This is temporary scaffolding - the model learns to predict embeddings,
        and we project to tokens only for the loss computation.

        Production Note: Remove output_projection and use embedding rounding.
        """
        # Entry: tokens → embeddings
        hidden_states = self.embed_tokens(input_ids)  # (B, L, D)

        # Core: embeddings → embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states)  # Still (B, L, D)

        # Exit (temporary): embeddings → logits
        logits = self.output_projection(hidden_states)  # (B, L, vocab_size)

        return logits
```

### 3.2 Target Implementation (Embedding Prediction)

```python
class TritterModel(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        """Forward pass with embedding prediction.

        Args:
            input_ids: Token IDs (B, L)
            return_embeddings: If True, return embeddings instead of logits

        Returns:
            If return_embeddings: hidden_states (B, L, D)
            Else: logits (B, L, vocab_size)

        Why: Embedding prediction enables continuous reasoning. The model
        predicts what the next embedding should be, not what token it is.
        Logits are only computed when explicitly training with cross-entropy.
        """
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        if return_embeddings:
            return hidden_states

        return self.output_projection(hidden_states)
```

### 3.3 Loss Function Evolution

**Phase 1: Token Cross-Entropy (Current)**
```python
loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
```

**Phase 2: Hybrid Loss (Transition)**
```python
# α starts at 0, increases to 1 during training
embedding_loss = F.mse_loss(predicted_embeddings, target_embeddings)
token_loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
loss = α * embedding_loss + (1 - α) * token_loss
```

**Phase 3: Pure Embedding Loss (Target)**
```python
# Target embeddings are the embeddings of the next token
target_embeddings = self.embed_tokens(labels)  # (B, L, D)
loss = F.mse_loss(predicted_embeddings, target_embeddings)
```

---

## 4. Inference Architecture

### 4.1 Standard Generation (Token-by-Token)

```python
def generate_tokens(model, prompt_ids, max_new_tokens):
    """Standard autoregressive generation.

    Note: This is the compatibility mode, not the embedding-native mode.
    """
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
    return input_ids
```

### 4.2 Embedding-Native Generation (Target)

```python
def generate_embeddings(model, prompt_ids, max_new_tokens, rounder):
    """Embedding-native generation with discretization only at output.

    Args:
        model: TritterModel
        prompt_ids: Initial token IDs
        max_new_tokens: Number of steps to generate
        rounder: EmbeddingRounder for KNN/VQ discretization

    Why: Keep generation in continuous space as long as possible.
    The model predicts embeddings, not tokens. We only discretize
    when we need to output text.
    """
    # Initial embedding
    embeddings = model.embed_tokens(prompt_ids)  # (B, L, D)

    for _ in range(max_new_tokens):
        # Transform embeddings (no tokenization)
        hidden = model.transformer(embeddings)
        next_embedding = hidden[:, -1:, :]  # Last position embedding

        # Append to sequence (in embedding space)
        embeddings = torch.cat([embeddings, next_embedding], dim=1)

    # Discretize only at the end
    output_tokens = rounder.round(embeddings[:, prompt_ids.size(1):, :])
    return output_tokens
```

### 4.3 Coconut-Style Latent Reasoning

```python
def latent_reasoning(model, prompt_ids, num_thinking_steps):
    """Continuous chain-of-thought without token boundaries.

    Why: Reasoning in embedding space avoids the information bottleneck
    of discrete tokens. The model can maintain multiple hypotheses,
    explore alternative paths, and refine understanding without
    committing to specific word choices.

    Reference: Coconut paper shows this enables BFS-like reasoning
    where multiple paths are explored simultaneously in latent space.
    """
    embeddings = model.embed_tokens(prompt_ids)

    # Thinking phase: iterate in embedding space
    for _ in range(num_thinking_steps):
        # Each step refines understanding without tokenizing
        embeddings = model.transformer(embeddings, is_causal=False)

    # Only discretize the final output
    output_embedding = embeddings[:, -1, :]
    return output_embedding  # Can be rounded to tokens if needed
```

---

## 5. Embedding Rounding Strategies

### 5.1 KNN Lookup

```python
class KNNRounder:
    """Round embeddings to nearest token via KNN lookup.

    Why: Simple and fast. Precompute embedding matrix index,
    then find nearest neighbor at inference time.

    Complexity: O(D) with approximate nearest neighbor index
    Quality: Good for well-separated embeddings
    """
    def __init__(self, embedding_matrix: torch.Tensor):
        self.embeddings = embedding_matrix  # (vocab_size, D)
        self.index = self._build_index()

    def round(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Round embeddings to nearest token IDs.

        Args:
            embeddings: (B, L, D) continuous embeddings

        Returns:
            token_ids: (B, L) discrete token IDs
        """
        # Flatten for batch lookup
        flat = embeddings.view(-1, embeddings.size(-1))

        # Find nearest neighbors
        distances = torch.cdist(flat, self.embeddings)
        token_ids = distances.argmin(dim=-1)

        return token_ids.view(embeddings.size(0), embeddings.size(1))
```

### 5.2 Vector Quantization (VQ)

```python
class VQRounder:
    """Round embeddings via learned vector quantization.

    Why: VQ codebook can be trained to maximize reconstruction quality.
    Better than KNN when embedding space is complex or multi-modal.

    Complexity: O(codebook_size * D) per token
    Quality: Best with trained codebook
    """
    def __init__(self, codebook_size: int, embedding_dim: int):
        self.codebook = nn.Embedding(codebook_size, embedding_dim)

    def round(self, embeddings: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(embeddings, self.codebook.weight)
        return distances.argmin(dim=-1)
```

### 5.3 Latent Refinement Decoding (LRD)

```python
class LRDRounder:
    """Progressive refinement from continuous to discrete.

    Why: Two-phase approach reduces rounding error. First phase
    finds approximate candidates, second phase refines among them.

    Complexity: O(K * D) where K is candidate set size
    Quality: Best quality, highest latency

    Reference: LCM paper §5
    """
    def round(self, embeddings: torch.Tensor, num_candidates: int = 10) -> torch.Tensor:
        # Phase 1: Get top-K candidates via KNN
        distances = torch.cdist(embeddings, self.embeddings)
        _, candidates = distances.topk(num_candidates, largest=False)

        # Phase 2: Refine among candidates (e.g., via scoring model)
        refined = self._refine_candidates(embeddings, candidates)

        return refined
```

---

## 6. Implementation Checklist

### Phase 1: Training Scaffolding (Current)

- [x] Token embedding layer
- [x] Transformer layers operating on embeddings
- [x] Output projection for logits
- [x] Cross-entropy loss training

### Phase 2: Embedding Interface (Next)

- [ ] Add `return_embeddings` parameter to forward()
- [ ] Add `get_embeddings()` method for extraction
- [ ] Document embedding-prediction paradigm in all docstrings
- [ ] Add embedding dimension assertions

### Phase 3: Hybrid Training (Future)

- [ ] Implement EmbeddingPredictionLoss
- [ ] Implement CurriculumScheduler for α
- [ ] Add target embedding computation
- [ ] Benchmark hybrid vs pure token loss

### Phase 4: Inference Rounding (Future)

- [ ] Implement KNNRounder
- [ ] Implement VQRounder
- [ ] Implement LRDRounder
- [ ] Benchmark rounding quality vs latency

### Phase 5: Latent Reasoning (Research)

- [ ] Implement Coconut-style generation
- [ ] Benchmark latent vs token reasoning
- [ ] Evaluate on planning/reasoning tasks

---

## 7. Documentation Requirements

Per DEVELOPMENT_STANDARDS.md, all code touching the embedding-prediction paradigm must include explicit documentation:

### Required Docstring Pattern

```python
def some_function(embeddings: torch.Tensor) -> torch.Tensor:
    """Brief description.

    Args:
        embeddings: Continuous embeddings (B, L, D)

    Returns:
        Transformed embeddings (B, L, D)

    Why (Embedding-Prediction Context):
        This function operates in continuous embedding space as part of
        Tritter's embedding-prediction paradigm. [Explain how this fits
        into the continuous computation model, why discrete tokens are
        not involved, etc.]

    Note (Token Interface):
        [If this function interacts with tokens, explain that this is
        an entry/exit point, not core computation.]
    """
```

### Module Docstring Pattern

```python
"""Module description.

Why (Embedding-Prediction Paradigm):
    Tritter operates in continuous embedding space:
    - Entry point: Tokenization converts discrete tokens → embeddings
    - Core computation: Transformer layers operate on continuous embeddings
    - Exit point: Output projection to logits is temporary; production
      uses KNN/VQ rounding only when outputting text

This module [explain role in the paradigm].
"""
```

---

## 8. Verification Criteria

### 8.1 Architectural Verification

| Criterion | Verification Method |
|-----------|---------------------|
| No intermediate tokenization | Code review: no argmax/softmax between layers |
| Embedding dimensions consistent | Assert hidden_states.shape[-1] == config.hidden_size |
| Residual connections in embedding space | Verify add operation, not token-level |

### 8.2 Training Verification

| Criterion | Verification Method |
|-----------|---------------------|
| Gradients flow through embeddings | Check embedding layer gradients non-zero |
| Loss decreases on embedding objective | Monitor embedding MSE during training |
| Curriculum schedule progresses | Log α value throughout training |

### 8.3 Inference Verification

| Criterion | Verification Method |
|-----------|---------------------|
| KNN rounding produces valid tokens | Output tokens in [0, vocab_size) |
| Embedding generation maintains quality | Compare perplexity with token generation |
| Latent reasoning improves on planning tasks | Benchmark on ARC/planning datasets |

---

## Appendix A: Comparison with Standard LM

| Aspect | Standard LM | Tritter Embedding-Prediction |
|--------|-------------|------------------------------|
| Core unit | Token | Embedding vector |
| Between layers | Embeddings | Embeddings (same) |
| Training objective | P(token | context) | E[embedding | context] |
| Generation | Token-by-token argmax | Embedding sequence → round at end |
| Reasoning | Token chain-of-thought | Continuous latent reasoning |
| Information bottleneck | Discrete tokens | None (continuous) |

---

## Appendix B: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-01-22 | Claude | Initial draft |
