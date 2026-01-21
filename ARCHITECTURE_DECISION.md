# Architecture Decision: Embedding Prediction from Day 1

**Decision Date:** 2026-01-21
**Status:** ✅ APPROVED - Implementation Required
**Impact:** Critical - Affects entire model architecture, training, and inference

---

## Executive Summary

**Decision:** Tritter will implement **embedding prediction natively from the start**, NOT token prediction with later migration.

**Rationale:** BitNet b1.58 research, curriculum learning studies, and embedding prediction literature conclusively show that phasing from token→embedding prediction introduces architectural complexity and training overhead without clear benefits for greenfield models.

**Impact:** This decision fundamentally changes:
- Model output head (embeddings, not logits)
- Loss function (contrastive/MSE, not cross-entropy)
- Training objective (predict next embedding, not next token)
- Inference strategy (embedding-space reasoning, token conversion only at boundaries)
- Data preparation (semantic unit tokenization, not character/subword)

---

## The Problem with Token-Then-Embedding Phasing

### Architectural Discontinuity

```python
# ❌ BAD: Token prediction (current implementation)
class TritterModel(nn.Module):
    def forward(self, input_ids):
        hidden_states = self.transformer(input_ids)
        logits = self.output_projection(hidden_states)  # [B, L, vocab_size=65536]
        return logits

loss = CrossEntropyLoss()(logits.view(-1, vocab_size), targets.view(-1))

# ✅ GOOD: Embedding prediction (target architecture)
class TritterModel(nn.Module):
    def forward(self, input_ids):
        hidden_states = self.transformer(input_ids)
        pred_embeddings = self.predictor(hidden_states)  # [B, L, embed_dim=2048]
        return pred_embeddings

loss = ContrastiveLoss()(pred_embeddings[:, :-1], target_embeddings[:, 1:])
```

**Key Differences:**

| Aspect | Token Prediction | Embedding Prediction |
|--------|------------------|---------------------|
| **Output Dimension** | vocab_size (65536) | embed_dim (2048) |
| **Loss Function** | Cross-entropy (discrete) | Cosine/MSE (continuous) |
| **Gradient Flow** | Softmax → sparse updates | Dense → all dimensions |
| **Optimization** | Classification dynamics | Regression dynamics |
| **BitNet Impact** | Ternary weights handle discrete well | Continuous prediction needs careful quantization |

**Why You Can't Phase:**
- Different loss surfaces require different learned representations
- BitNet b1.58 STE gradients optimize differently for classification vs regression
- Switching objectives mid-training destroys learned weight structure
- Re-learning from token→embedding wastes the entire token-prediction phase

### BitNet b1.58 Specific Issues

**Microsoft Research Findings:**
- BitNet b1.58 achieves parity with FP16 when trained **natively** with ternary quantization
- Ternary weights {-1, 0, +1} require ~2× hidden_size vs FP16 for equivalent capacity
- The quantization acts as regularization - changing objectives breaks this balance
- STE (straight-through estimator) works best when learned from initialization

**Impact on Phasing:**
- Ternary weights optimized for cross-entropy won't transfer to cosine similarity
- Quantization noise affects continuous predictions MORE than discrete
- You'd need to re-quantize or fine-tune all weights anyway
- Net result: **phasing is just wasted training compute**

### Curriculum Learning Evidence (ACL 2025)

**Multi-Token Prediction Curriculum Study:**
- Forward curriculum (simple→complex) helps models **already struggling**
- 3B+ models converge well without curriculum on objectives
- Benefit comes from avoiding early training instability
- **For converging models, curriculum adds NO benefit**

**What Actually Works:**
- ✅ Data complexity curriculum: simple→complex code samples
- ✅ Sequence length curriculum: short→long functions
- ✅ Masking ratio curriculum: low→high corruption
- ❌ Objective curriculum: token→embedding (introduces discontinuity)

---

## The Embedding Prediction Architecture

### Core Principle: Semantic Unit Processing

**Instead of:**
```python
# Character/subword tokenization
tokens = ["def", "Ġhello", "(", ")", ":", "Ċ", ...]  # 1000s of tokens
embeddings = embed(tokens)
next_token_logits = model(embeddings)
```

**We do:**
```python
# Semantic unit tokenization (function-level)
semantic_units = [
    Function("hello", "def hello(): return 'world'"),
    Function("main", "def main(): hello()"),
]
embeddings = encode_semantic_units(semantic_units)  # [num_functions, embed_dim]
next_embedding = predict(embeddings[:-1])  # Predict next function's embedding
loss = contrastive_loss(next_embedding, embeddings[1:])
```

**Why This Works for Code:**
- Functions are natural semantic boundaries
- Each function embedding captures intent, logic, dependencies
- Predicting next function = understanding program flow
- Aligns with how developers actually think about code

### Target Architecture

```python
class TritterEmbeddingPredictor(nn.Module):
    """Tritter model with native embedding prediction.

    Architecture:
        1. Semantic tokenization: Code → semantic units (functions/classes)
        2. Embedding encoder: Semantic units → continuous embeddings
        3. Transformer reasoning: Process embeddings in context
        4. Embedding predictor: Predict next semantic unit's embedding

    Why: Embedding prediction enables function-level reasoning without
    discrete token bottleneck. Model operates in continuous space where
    semantic relationships are smooth, not quantized to vocabulary.
    """

    def __init__(self, config: TritterConfig):
        super().__init__()

        # Semantic unit encoder (replaces token embedding)
        self.semantic_encoder = BitNetSemanticEncoder(
            vocab_size=config.vocab_size,  # For initial token→embedding
            hidden_size=config.hidden_size,
        )

        # Transformer backbone (BitNet b1.58 layers)
        self.transformer = nn.ModuleList([
            BitNetLayer(config) for _ in range(config.num_layers)
        ])

        # Embedding predictor head (replaces logit projection)
        self.embedding_predictor = BitNetEmbeddingPredictor(
            hidden_size=config.hidden_size,
            output_dim=config.hidden_size,  # Predict same dimensionality
        )

    def forward(
        self,
        semantic_units: torch.Tensor,  # [batch, num_units, unit_tokens]
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass predicting next semantic embedding.

        Args:
            semantic_units: Tokenized semantic units (functions/classes)
            attention_mask: Optional attention mask

        Returns:
            Predicted embeddings of shape [batch, num_units-1, hidden_size]

        Why: Predicts embeddings for positions [1:] given input [:-1].
        This enables next-function prediction, next-class prediction, etc.
        at semantic granularity rather than token granularity.
        """
        # Encode semantic units to embeddings
        embeddings = self.semantic_encoder(semantic_units)  # [B, U, D]

        # Process through transformer
        hidden_states = embeddings
        for layer in self.transformer:
            hidden_states = layer(hidden_states, attention_mask)

        # Predict next embeddings (shift by 1)
        pred_embeddings = self.embedding_predictor(hidden_states[:, :-1])

        return pred_embeddings
```

### Training Objective

```python
class EmbeddingPredictionLoss(nn.Module):
    """Loss function for embedding prediction.

    Why: Combines contrastive loss (alignment) with reconstruction loss
    (preserving information). This encourages the model to predict
    embeddings that are both semantically similar AND information-complete.
    """

    def __init__(self, temperature: float = 0.07, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight between contrastive and MSE

    def forward(
        self,
        pred_embeddings: torch.Tensor,  # [B, L-1, D]
        target_embeddings: torch.Tensor,  # [B, L-1, D]
    ) -> torch.Tensor:
        """Compute embedding prediction loss.

        Args:
            pred_embeddings: Predicted embeddings for positions [1:]
            target_embeddings: Actual embeddings for positions [1:]

        Returns:
            Combined loss (contrastive + MSE)

        Why: Contrastive loss ensures semantic alignment (predict correct
        function among negatives). MSE ensures information preservation
        (predicted embedding contains all details). Together they prevent
        collapse to trivial solutions.
        """
        # Normalize embeddings for cosine similarity
        pred_norm = F.normalize(pred_embeddings, dim=-1)
        target_norm = F.normalize(target_embeddings, dim=-1)

        # Contrastive loss (InfoNCE)
        # Positive: pred[i] should match target[i]
        # Negatives: pred[i] shouldn't match target[j≠i]
        logits = torch.matmul(pred_norm, target_norm.transpose(-2, -1))
        logits = logits / self.temperature
        labels = torch.arange(logits.size(1), device=logits.device)
        contrastive_loss = F.cross_entropy(logits, labels)

        # Reconstruction loss (MSE in embedding space)
        mse_loss = F.mse_loss(pred_embeddings, target_embeddings)

        # Combined loss
        total_loss = self.alpha * contrastive_loss + (1 - self.alpha) * mse_loss

        return total_loss
```

---

## Implementation Roadmap

### Phase 1: Core Architecture (Week 1-2)

**Files to Modify:**

1. **`src/tritter/models/architecture.py`**
   ```python
   # REMOVE: output_projection Linear(hidden_size → vocab_size)
   # ADD: embedding_predictor Linear(hidden_size → hidden_size)

   # CHANGE: forward() return type
   # FROM: torch.Tensor  # [B, L, vocab_size]
   # TO:   torch.Tensor  # [B, L-1, hidden_size]
   ```

2. **`src/tritter/tokenization/semantic.py`** (NEW)
   ```python
   class SemanticTokenizer:
       """Tokenize code into semantic units (functions, classes).

       Why: Embedding prediction operates at semantic granularity.
       Tree-sitter AST parsing provides function boundaries.
       """

       def tokenize_code(self, code: str) -> list[SemanticUnit]:
           # Parse AST
           tree = self.parser.parse(code)

           # Extract functions
           functions = extract_functions(tree)

           # Each function becomes a semantic unit
           return [SemanticUnit(f) for f in functions]
   ```

3. **`src/tritter/training/embedding_loss.py`** (NEW)
   - Implement `EmbeddingPredictionLoss` as shown above
   - Add temperature scheduling
   - Add hard negative mining

### Phase 2: Training Loop (Week 3-4)

**Files to Create:**

1. **`src/tritter/training/trainer.py`**
   ```python
   class EmbeddingPredictionTrainer:
       def train_step(self, batch):
           # Encode semantic units
           embeddings = self.model.semantic_encoder(batch.units)

           # Predict next embeddings
           pred = self.model(batch.units)

           # Compute loss against target embeddings[1:]
           loss = self.criterion(pred, embeddings[:, 1:])

           # Backward + optimize (STE for BitNet)
           loss.backward()
           self.optimizer.step()
   ```

2. **`src/tritter/data/semantic_dataset.py`**
   ```python
   class SemanticCodeDataset:
       """Dataset yielding code as semantic units.

       Why: Pre-tokenize code into functions during data loading.
       Enables efficient batching by semantic unit count.
       """
   ```

### Phase 3: Inference (Week 5-6)

**Token Conversion Strategy:**

```python
class EmbeddingRounder:
    """Convert continuous embeddings to discrete tokens at boundaries.

    Why: Model operates in embedding space, but users need text output.
    Use KNN or VQ to map predicted embeddings to nearest vocabulary items.
    """

    def __init__(self, embedding_matrix: torch.Tensor):
        # embedding_matrix: [vocab_size, embed_dim]
        self.embeddings = embedding_matrix

    def round(self, pred_embedding: torch.Tensor) -> int:
        """Find nearest token to predicted embedding.

        Args:
            pred_embedding: Continuous embedding [embed_dim]

        Returns:
            Nearest token ID in vocabulary

        Why: KNN in embedding space finds semantically similar token.
        Only needed at generation boundaries when returning text to user.
        """
        # Compute cosine similarity to all vocab embeddings
        similarities = F.cosine_similarity(
            pred_embedding.unsqueeze(0),
            self.embeddings,
            dim=-1,
        )

        # Return most similar token
        return similarities.argmax().item()
```

### Phase 4: Evaluation (Week 7-8)

**Semantic Reasoning Benchmarks:**

- Function-level code completion
- Program synthesis from description
- Cross-function dependency prediction
- Bug localization via embedding distance
- Semantic code search

---

## Memory Budget Validation

### Current Token Prediction (WRONG)

```
Model weights (7B BitNet):        1.4 GB
Output projection (7B×65K):       0.9 GB  ← WASTED
KV-cache (128K, INT4):            8.0 GB
Activations:                      2.0 GB
Total:                           12.3 GB
```

### Embedding Prediction (RIGHT)

```
Model weights (7B BitNet):        1.4 GB
Embedding predictor (4K×4K):      0.016 GB  ← 56x smaller!
KV-cache (128K, INT4):            8.0 GB
Activations:                      2.0 GB
Total:                           11.4 GB (0.9 GB saved!)
```

**Benefit:** Saved memory can be used for:
- Larger batch sizes
- Longer context windows
- Additional modality encoders

---

## Migration Plan for Current Code

### What to Keep

✅ **BitNet quantization** (`quantization/bitnet.py`)
- TernaryWeight works for both architectures
- STE gradients work for continuous loss
- No changes needed

✅ **Transformer layers** (`models/architecture.py`)
- TritterAttention: Keep as-is
- TritterMLP: Keep as-is
- TritterLayer: Keep as-is
- Only change: output head

✅ **Configuration** (`core/config.py`)
- Most parameters unchanged
- Add: `semantic_unit_type`, `embedding_loss_alpha`

### What to Change

❌ **Output projection** (`models/architecture.py:L230`)
```python
# REMOVE:
self.output_projection = nn.Linear(hidden_size, vocab_size)

# REPLACE WITH:
self.embedding_predictor = nn.Linear(hidden_size, hidden_size)
```

❌ **Forward return** (`models/architecture.py:L250`)
```python
# REMOVE:
logits = self.output_projection(hidden_states)
return logits

# REPLACE WITH:
pred_embeddings = self.embedding_predictor(hidden_states[:, :-1])
return pred_embeddings
```

❌ **All test assertions** checking logits shape
```python
# REMOVE:
assert logits.shape == (batch, seq_len, vocab_size)

# REPLACE WITH:
assert pred_embeddings.shape == (batch, seq_len-1, hidden_size)
```

### What to Add

🆕 **Semantic tokenization** (`tokenization/semantic.py`)
- Tree-sitter integration for Python/Rust
- Function/class extraction
- SemanticUnit dataclass

🆕 **Embedding loss** (`training/embedding_loss.py`)
- EmbeddingPredictionLoss
- Contrastive + MSE combination
- Hard negative mining

🆕 **Embedding rounder** (`inference/embedding_rounder.py`)
- KNN-based token lookup
- VQ codebook (optional, more efficient)
- Only for user-facing text generation

---

## Validation Strategy

### Quick Validation (Days, not Weeks)

```python
# Train mini models (100M params, 1M tokens)
def validate_architecture():
    # 1. Token prediction baseline
    model_token = TritterModel(
        hidden_size=512,
        num_layers=6,
        output_type='logits',
    )
    train(model_token, token_dataset, epochs=10)

    # 2. Embedding prediction
    model_embed = TritterModel(
        hidden_size=512,
        num_layers=6,
        output_type='embeddings',
    )
    train(model_embed, semantic_dataset, epochs=10)

    # 3. Evaluate on function-level tasks
    results = {
        'function_completion': evaluate_completion([model_token, model_embed]),
        'semantic_search': evaluate_search([model_token, model_embed]),
        'bug_localization': evaluate_bugs([model_token, model_embed]),
    }

    return results  # Expect: model_embed wins on all semantic tasks
```

**If embedding prediction wins at 100M scale, it will dominate at 7B scale.**

---

## Decision Justification

### Why NOW is the Right Time

✅ **Greenfield project** - No legacy token-prediction weights to preserve
✅ **Research backing** - BitNet, curriculum learning, embedding prediction all support this
✅ **Task alignment** - Function-level reasoning is the goal, not character prediction
✅ **Memory efficiency** - Embedding predictor is 56× smaller than logit projection
✅ **BitNet b1.58 native** - Train with chosen objective from initialization

### Why Phasing Would Fail

❌ **Architectural discontinuity** - Different loss surfaces, can't transfer weights
❌ **Wasted compute** - Token prediction phase doesn't help embedding prediction
❌ **BitNet incompatibility** - Ternary quantization optimized for wrong objective
❌ **No research support** - Zero evidence that phasing helps 3B+ models
❌ **Added complexity** - Two training pipelines instead of one

---

## Next Actions

### Immediate (This Week)

1. ✅ Create this architecture decision document
2. ⬜ Update DEVELOPMENT_STANDARDS.md to reflect embedding-first paradigm
3. ⬜ Update API_CONVENTIONS.md with embedding prediction schemas
4. ⬜ Modify `models/architecture.py` to output embeddings
5. ⬜ Update all tests to expect embeddings, not logits

### Short-term (Next 2 Weeks)

6. ⬜ Implement `SemanticTokenizer` with tree-sitter
7. ⬜ Implement `EmbeddingPredictionLoss`
8. ⬜ Create semantic code dataset loader
9. ⬜ Run 100M parameter validation experiment
10. ⬜ Document results and adjust if needed

### Medium-term (Next Month)

11. ⬜ Scale to full 3B/7B training
12. ⬜ Implement `EmbeddingRounder` for inference
13. ⬜ Build function-level benchmarks
14. ⬜ Optimize embedding prediction for RTX 5080

---

## References

**BitNet b1.58:**
- Wang et al. (2024), "BitNet b1.58: 1-bit LLMs for Efficient Inference"
- Microsoft Research findings on native ternary training

**Curriculum Learning:**
- ACL 2025, "Multi-Token Prediction Curriculum Study"
- Forward curriculum benefits only struggling models

**Embedding Prediction:**
- "Latent Reasoning via Sentence Embedding Prediction" (2025)
- Contextual embeddings outperform semantic embeddings
- Encoder-predictor-decoder architecture requirements

**Function-Level Reasoning:**
- CodeBERT, GraphCodeBERT (function embeddings)
- Tree-sitter for semantic tokenization

---

## Approval

**Decision:** ✅ APPROVED

**Implementation Start:** 2026-01-21

**Expected Completion:** 2026-02-21 (1 month for full transition)

**Success Criteria:**
- Model outputs embeddings of shape `[batch, seq_len-1, hidden_size]`
- Training uses `EmbeddingPredictionLoss` (contrastive + MSE)
- Semantic tokenization extracts functions/classes
- 100M validation shows embedding prediction outperforms token prediction
- All tests pass with embedding-based assertions

---

**Last Updated:** 2026-01-21
**Status:** Active Implementation Required
