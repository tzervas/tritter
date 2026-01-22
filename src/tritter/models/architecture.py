"""
Tritter multimodal transformer architecture.

Implements a transformer-based model with BitNet quantization, multimodal support,
and optimizations for 128K context window processing.
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
            attention_mask: Optional attention mask. If None, creates causal mask for
                           autoregressive generation. Shape: (batch_size, 1, seq_len, seq_len)
                           or broadcastable. Values: 0 = attend, -inf = mask.

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)

        Why: Causal masking is essential for autoregressive (left-to-right) generation to
        prevent the model from attending to future tokens. Without it, the model would "cheat"
        during training by looking ahead. FlashAttention integration provides O(N) memory
        complexity instead of O(N²), critical for 128K context. When FlashAttention is enabled
        and no external mask is provided, we use is_causal=True which triggers the optimized
        causal kernel instead of manually creating an O(N²) mask. QK-Norm prevents attention
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
            # Optimization: When attention_mask is None, use is_causal=True with attn_mask=None
            # This allows FlashAttention-2 to use its optimized causal kernel (O(N) memory vs O(N²))
            # instead of manually creating an O(N²) mask. When attention_mask is provided,
            # we use is_causal=False with the provided mask.
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=(attention_mask is None),
            )
        else:
            # Fallback: Standard scaled dot-product attention
            # Create causal mask if not provided (for autoregressive generation)
            if attention_mask is None:
                # Causal mask: lower triangular matrix (can attend to past, not future)
                # Shape: (1, 1, seq_len, seq_len) for broadcasting
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device),
                    diagonal=1,
                )
                attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)

            if attention_mask is not None:
                scores = scores + attention_mask

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
    """Main Tritter multimodal transformer model."""

    def __init__(self, config: TritterConfig) -> None:
        """Initialize Tritter model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Embeddings
        self.embed_tokens = UnifiedEmbedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=0,
        )

        # Transformer layers
        self.layers = nn.ModuleList([TritterLayer(config) for _ in range(config.num_layers)])

        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Output projection
        if config.use_bitnet:
            self.lm_head = TernaryWeight(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        return logits
