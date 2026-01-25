"""Tritter multimodal transformer architecture.

Implements a transformer-based model with BitNet quantization, multimodal support,
and optimizations for 128K context window processing.

Why (Embedding-Prediction Paradigm):
    Tritter operates in continuous embedding space:
    - Entry point: Tokenization converts discrete tokens → embeddings
    - Core computation: Transformer layers operate on continuous embeddings
    - Exit point: Output projection to logits is temporary scaffolding;
      production uses KNN/VQ rounding only when outputting text

    The model's "reasoning" happens in continuous embedding space, not discrete
    token space. Token prediction via logits is temporary for training compatibility.
    See SPEC-003-embedding-prediction.md for full paradigm documentation.
"""

import torch
import torch.nn as nn

from tritter.core.config import TritterConfig
from tritter.quantization.bitnet import TernaryWeight
from tritter.tokenization.multimodal import UnifiedEmbedding


class TritterAttention(nn.Module):
    """Multi-head attention with QK-Norm, FlashAttention, and optional sliding window support.

    Why: Multi-head attention enables the model to attend to different representation subspaces
    simultaneously. QK-Norm (query-key normalization) from Chameleon/BitNet papers provides
    training stability by preventing attention score explosion. FlashAttention reduces memory
    from O(N²) to O(N) and speeds up computation.

    Note: Sliding window attention is configured but not yet implemented. The sliding_window_size
    parameter is present for future implementation but currently has no effect on attention
    computation. Full attention is used regardless of this setting.
    """

    def __init__(self, config: TritterConfig) -> None:
        """Initialize attention module.

        Args:
            config: Model configuration

        Why: QK-Norm is critical for stability with BitNet quantization and long contexts.
        Query and key normalization after projection prevents attention scores from exploding
        as sequence length increases. FlashAttention integration provides memory-efficient
        attention computation essential for 128K context on 16GB VRAM.
        """
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.sliding_window = config.sliding_window_size

        # Query, Key, Value projections
        if config.use_bitnet:
            self.q_proj = TernaryWeight(self.hidden_size, self.hidden_size)
            self.k_proj = TernaryWeight(self.hidden_size, self.hidden_size)
            self.v_proj = TernaryWeight(self.hidden_size, self.hidden_size)
            self.o_proj = TernaryWeight(self.hidden_size, self.hidden_size)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
            self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
            self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
            self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # QK-Norm: LayerNorm for queries and keys
        # Why: Normalizing Q and K after projection but before attention computation prevents
        # score explosion and provides training stability, especially critical for BitNet
        # quantization and long context (128K). This follows Chameleon and BitNet b1.58 papers.
        self.q_norm = nn.LayerNorm(self.head_dim, eps=config.layer_norm_eps)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with multi-head attention.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask for padding/prefix masking. This module
                           enforces causal masking, so any provided mask will be combined
                           with a causal mask. Shape: (batch_size, 1, seq_len, seq_len)
                           or broadcastable. Values: 0 = attend, -inf = mask.
                           If None, uses pure causal masking for autoregressive generation.

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)

        Why: Causal masking is essential for autoregressive (left-to-right) generation to
        prevent the model from attending to future tokens. Without it, the model would "cheat"
        during training by looking ahead. FlashAttention integration provides O(N) memory
        complexity instead of O(N²), critical for 128K context. When FlashAttention is enabled
        and no external mask is provided, we use is_causal=True which triggers the optimized
        causal kernel instead of manually creating an O(N²) mask. When a custom mask is provided,
        we combine it with a causal mask to preserve causality. QK-Norm prevents attention
        score explosion with long sequences and ternary quantization.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply QK-Norm (per-head normalization)
        # Why: Normalize after reshaping to apply per-head. This prevents any single head
        # from dominating attention and ensures stable training across all heads.
        query = self.q_norm(query)
        key = self.k_norm(key)

        # Transpose for attention computation: (batch, heads, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Use FlashAttention if enabled (significantly faster and more memory efficient)
        if getattr(self.config, "use_flash_attention", False):
            # PyTorch's scaled_dot_product_attention uses FlashAttention when available
            # Why: Use is_causal=True for optimal kernel dispatch. This triggers FlashAttention-2's
            # optimized causal kernel that:
            # 1. Never materializes the O(N²) causal mask (critical for 128K context)
            # 2. Uses fused attention computation (faster than separate ops)
            # 3. Enables optimal tiling for long sequences
            #
            # Embedding-Prediction Context: Attention operates in continuous embedding space.
            # The Q/K/V projections transform embeddings (not discrete tokens) to enable
            # semantic similarity matching, consistent with the Coconut/LCM paradigm.
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,  # (B, H, L, head_dim)
                key,  # (B, H, L, head_dim)
                value,  # (B, H, L, head_dim)
                attn_mask=attention_mask,  # Optional mask (e.g., for padding)
                dropout_p=0.0,
                is_causal=True,  # Always use causal - FlashAttention handles this optimally
            )  # -> (B, H, L, head_dim)
        else:
            # Fallback: Standard scaled dot-product attention
            # Always enforce causal masking
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device),
                diagonal=1,
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

            # Combine with custom mask if provided
            if attention_mask is not None:
                combined_mask = attention_mask + causal_mask
            else:
                combined_mask = causal_mask

            scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)
            scores = scores + combined_mask
            attn_weights = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, value)

        # Reshape back: (batch, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # Output projection
        output = self.o_proj(attn_output)
        return output


class TritterMLP(nn.Module):
    """Feed-forward network with optional BitNet quantization.

    Why: The MLP (Multi-Layer Perceptron) implements the position-wise feed-forward network
    that processes each position independently after attention. Uses Squared ReLU activation
    as required by BitNet paper for stable training with ternary weights.
    """

    def __init__(self, config: TritterConfig) -> None:
        """Initialize MLP module.

        Args:
            config: Model configuration

        Why: SwiGLU-style gating (gate_proj * up_proj) with Squared ReLU activation provides
        better performance than standard FFN. The intermediate_size (typically 4x hidden_size)
        expands the representation space to capture complex patterns before projecting back.
        BitNet paper requires Squared ReLU (x * ReLU(x)) for numerical stability with ternary
        quantization - standard ReLU or SiLU can cause training instability.
        """
        super().__init__()
        self.config = config

        if config.use_bitnet:
            self.gate_proj = TernaryWeight(config.hidden_size, config.intermediate_size)
            self.up_proj = TernaryWeight(config.hidden_size, config.intermediate_size)
            self.down_proj = TernaryWeight(config.intermediate_size, config.hidden_size)
        else:
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)

    def squared_relu(self, x: torch.Tensor) -> torch.Tensor:
        """Squared ReLU activation: x * ReLU(x).

        Args:
            x: Input tensor

        Returns:
            Activated tensor

        Why: Squared ReLU (x * ReLU(x)) is required by BitNet b1.58 paper for stable training
        with ternary weights. Compared to standard ReLU, it provides:
        1. Smoother gradients (derivative is 2x for x > 0 instead of constant 1)
        2. Better numerical stability with quantized weights
        3. Stronger activation for large positive values
        This activation is critical for BitNet - using SiLU or standard ReLU can cause
        training instability and poor convergence with ternary quantization.
        """
        relu_x = torch.relu(x)
        return x * relu_x

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)

        Why: SwiGLU gating mechanism (gate * up) provides better performance than standard
        FFN by allowing the model to control information flow through the gate pathway.
        The down projection brings the expanded representation back to hidden_size for
        residual connection compatibility.
        """
        gate = self.squared_relu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        down = self.down_proj(gate * up)
        return down


class TritterLayer(nn.Module):
    """Single transformer layer with attention and MLP.

    Why: Each transformer layer applies self-attention followed by feed-forward network (MLP),
    with residual connections around each sub-layer. Post-FFN LayerNorm placement (rather than
    pre-FFN) follows Chameleon's stability improvements and provides better gradient flow for
    deep networks (24-32 layers) with BitNet quantization.
    """

    def __init__(self, config: TritterConfig) -> None:
        """Initialize transformer layer.

        Args:
            config: Model configuration

        Why: Post-FFN LayerNorm is a key stability technique from Chameleon paper. Unlike
        standard Pre-LN transformers, this applies normalization after the residual addition
        in the MLP block, providing more stable training especially important for BitNet's
        ternary quantization and long context (128K). The pattern is:
        - Attention: residual + attention(norm(x))  [Pre-attention norm]
        - MLP: norm(residual + mlp(x))              [Post-MLP norm]
        """
        super().__init__()
        self.attention = TritterAttention(config)
        self.mlp = TritterMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_mlp_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through transformer layer.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)

        Why: Implements Chameleon's post-FFN LayerNorm pattern for stability:
        1. Attention block: Pre-normalization (norm before attention)
        2. MLP block: Post-normalization (norm after residual add)
        This asymmetric pattern provides better training stability than pure Pre-LN or Post-LN,
        especially critical for BitNet quantization where gradients can be unstable.
        """
        # Self-attention with residual and pre-attention normalization
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP with residual and POST-FFN normalization (Chameleon-style)
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.post_mlp_layernorm(hidden_states)  # Normalize AFTER residual

        return hidden_states


class TritterModel(nn.Module):
    """Main Tritter multimodal transformer model with embedding prediction support.

    Why (Embedding-Prediction Paradigm):
        This model operates in continuous embedding space. The forward() method supports
        two modes:
        - return_embeddings=False (default): Returns logits for token prediction training
        - return_embeddings=True: Returns continuous embeddings for embedding prediction

        Token prediction via logits is temporary scaffolding for training compatibility.
        Production inference uses embedding prediction with KNN/VQ rounding.

    Attributes:
        config: TritterConfig instance
        embed_tokens: Token embedding layer (entry point from discrete to continuous)
        layers: Stack of TritterLayer transformer blocks
        norm: Final LayerNorm before output
        lm_head: Output projection to logits (temporary scaffolding)

    Example:
        >>> config = TritterConfig(model_size="3B")
        >>> model = TritterModel(config)
        >>> # Token prediction mode (training)
        >>> logits = model(input_ids)  # (B, L, vocab_size)
        >>> # Embedding prediction mode
        >>> embeddings = model(input_ids, return_embeddings=True)  # (B, L, D)
    """

    def __init__(self, config: TritterConfig) -> None:
        """Initialize Tritter model.

        Args:
            config: Model configuration

        Why: Initializes all components for embedding-prediction architecture.
        The lm_head is temporary scaffolding for training - production inference
        will use get_embeddings() + KNN/VQ rounding instead.
        """
        super().__init__()
        self.config = config

        # Embeddings - Entry point from discrete tokens to continuous space
        # Why: This is the ONLY place where discrete tokens enter the model.
        # After this, all computation operates on continuous embedding vectors.
        self.embed_tokens = UnifiedEmbedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=0,
        )

        # Transformer layers - Core computation in continuous space
        # Why: Each layer transforms embeddings → embeddings without any
        # intermediate tokenization, consistent with Coconut/LCM paradigm.
        self.layers = nn.ModuleList([TritterLayer(config) for _ in range(config.num_layers)])

        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Output projection - Temporary scaffolding for token prediction training
        # Why: Cross-entropy training requires logits over vocabulary. This will
        # be bypassed in production when using embedding prediction + rounding.
        if config.use_bitnet:
            self.lm_head = TernaryWeight(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing to reduce memory usage during training.

        Why: Gradient checkpointing trades compute for memory by recomputing
        activations during backward pass instead of storing them. Reduces memory
        by ~60% for large models, enabling training on limited VRAM.
        """
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

    @property
    def is_gradient_checkpointing(self) -> bool:
        """Check if gradient checkpointing is enabled."""
        return getattr(self, "_gradient_checkpointing", False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        """Forward pass with optional embedding prediction.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            return_embeddings: If True, return embeddings instead of logits

        Returns:
            If return_embeddings=True: Hidden states (batch_size, seq_len, hidden_size)
            If return_embeddings=False: Logits (batch_size, seq_len, vocab_size)

        Why (Embedding-Prediction Context):
            This method supports both token prediction (training compatibility) and
            embedding prediction (continuous reasoning). The return_embeddings flag
            determines whether to apply the output projection or return raw embeddings.

            Token prediction mode computes cross-entropy loss against target tokens.
            Embedding prediction mode enables continuous reasoning and deferred
            discretization via KNN/VQ rounding.

        Note (Token Interface):
            input_ids is the entry point from discrete to continuous space.
            If return_embeddings=False, logits are the exit point back to discrete.
            If return_embeddings=True, caller is responsible for discretization.
        """
        # Entry point: tokens → embeddings
        # Why: This is the ONLY discretization boundary. After embedding lookup,
        # all subsequent computation operates in continuous space.
        hidden_states = self.embed_tokens(input_ids)  # (B, L, D)

        # Core computation: embeddings → embeddings
        # Why: No tokenization between layers. Each layer transforms continuous
        # representations, preserving semantic information without discretization loss.
        if self.is_gradient_checkpointing and self.training:
            # Use gradient checkpointing to reduce memory
            from torch.utils.checkpoint import checkpoint

            for layer in self.layers:
                hidden_states = checkpoint(
                    layer, hidden_states, attention_mask, use_reentrant=False
                )
        else:
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask)  # Still (B, L, D)

        # Final layer norm
        hidden_states = self.norm(hidden_states)  # (B, L, D)

        # Return embeddings for embedding prediction mode
        if return_embeddings:
            return hidden_states  # (B, L, D) - continuous representation

        # Exit point (temporary): embeddings → logits
        # Why: Token prediction scaffolding for training. Production will bypass
        # this and use KNN/VQ rounding on embeddings instead.
        logits = self.lm_head(hidden_states)  # (B, L, vocab_size)

        return logits

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract continuous embeddings without token projection.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask

        Returns:
            Hidden state embeddings of shape (batch_size, seq_len, hidden_size)

        Why (Embedding-Prediction Context):
            Convenience method for embedding extraction. Equivalent to
            forward(input_ids, return_embeddings=True) but with clearer intent.

            Use this for:
            - Semantic similarity computation
            - Embedding-based generation with KNN/VQ rounding
            - Continuous reasoning / latent chain-of-thought
            - Feature extraction for downstream tasks

        Example:
            >>> embeddings = model.get_embeddings(input_ids)
            >>> # Find nearest token via KNN
            >>> distances = torch.cdist(embeddings, model.embed_tokens.weight)
            >>> nearest_tokens = distances.argmin(dim=-1)
        """
        return self.forward(input_ids, attention_mask, return_embeddings=True)

    def get_target_embeddings(self, labels: torch.Tensor) -> torch.Tensor:
        """Get target embeddings for embedding prediction loss.

        Args:
            labels: Target token IDs of shape (batch_size, seq_len)

        Returns:
            Target embeddings of shape (batch_size, seq_len, hidden_size)

        Why (Embedding-Prediction Context):
            For hybrid or pure embedding prediction training, the target is the
            embedding of the next token rather than a one-hot distribution. This
            method retrieves the target embeddings for MSE loss computation.

            Hybrid loss: α * MSE(pred_emb, target_emb) + (1-α) * CE(logits, labels)
            Pure embedding loss: MSE(pred_emb, target_emb)

        Example:
            >>> pred_embeddings = model.get_embeddings(input_ids)
            >>> target_embeddings = model.get_target_embeddings(labels)
            >>> embedding_loss = F.mse_loss(pred_embeddings, target_embeddings)
        """
        return self.embed_tokens(labels)

    def embedding_prediction_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = 0.0,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute hybrid embedding + token prediction loss.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            labels: Target token IDs of shape (batch_size, seq_len)
            alpha: Weight for embedding loss (0.0 = pure token, 1.0 = pure embedding)
            attention_mask: Optional attention mask

        Returns:
            Dictionary containing:
            - 'loss': Combined loss value
            - 'token_loss': Cross-entropy token loss
            - 'embedding_loss': MSE embedding loss
            - 'alpha': Current alpha value (for logging)

        Why (Embedding-Prediction Context):
            Curriculum training starts with α=0 (pure token loss) and gradually
            increases to α=1 (pure embedding loss). This allows the model to learn
            token prediction first, then transition to embedding prediction.

            The hybrid approach provides:
            1. Training stability (token loss grounds initial learning)
            2. Smooth transition to embedding prediction
            3. Flexibility to tune the α schedule

        Example:
            >>> # Early training: mostly token loss
            >>> loss_dict = model.embedding_prediction_loss(input_ids, labels, alpha=0.1)
            >>> # Late training: mostly embedding loss
            >>> loss_dict = model.embedding_prediction_loss(input_ids, labels, alpha=0.9)
        """
        import torch.nn.functional as F

        # Get predicted embeddings and logits
        hidden_states = self.forward(input_ids, attention_mask, return_embeddings=True)
        logits = self.lm_head(hidden_states)  # (B, L, vocab_size)

        # Get target embeddings (embeddings of the next token)
        target_embeddings = self.get_target_embeddings(labels)  # (B, L, D)

        # Token prediction loss (cross-entropy)
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()  # (B, L-1, vocab_size)
        shift_labels = labels[..., 1:].contiguous()  # (B, L-1)
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,  # Ignore padding
        )

        # Embedding prediction loss (MSE)
        # Shift hidden states and target embeddings for next-embedding prediction
        shift_hidden = hidden_states[..., :-1, :].contiguous()  # (B, L-1, D)
        shift_target = target_embeddings[..., 1:, :].contiguous()  # (B, L-1, D)
        embedding_loss = F.mse_loss(shift_hidden, shift_target)

        # Combined loss
        loss = (1 - alpha) * token_loss + alpha * embedding_loss

        return {
            'loss': loss,
            'token_loss': token_loss,
            'embedding_loss': embedding_loss,
            'alpha': torch.tensor(alpha),
        }
