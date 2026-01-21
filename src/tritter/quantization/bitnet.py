"""
BitNet 1.58-bit quantization for ternary weights.

Implements the quantization scheme from "The Era of 1-bit LLMs: All Large Language Models
are in 1.58 Bits" where weights are quantized to {-1, 0, 1}.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class TernaryWeight(nn.Module):
    """Ternary weight representation using BitNet 1.58-bit quantization.

    Weights are quantized to three values: {-1, 0, 1} with a scaling factor.
    This provides significant memory savings while maintaining model quality.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """Initialize ternary weight layer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: If True, adds a learnable bias

        Why: Full-precision shadow weights (self.weight) are maintained for training via
        straight-through estimator (STE). Quantization happens in forward pass only.
        The scale parameter allows per-channel adaptive scaling after quantization.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Full-precision weights for training
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        # Scaling factors for quantization (per output channel)
        self.scale = nn.Parameter(torch.ones(out_features, 1))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Cache for quantized weights during eval mode
        self.register_buffer("_quantized_weight_cache", None)
        self._cache_valid = False

    def quantize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize weights to ternary values {-1, 0, 1}.

        Args:
            weights: Full-precision weight tensor of shape (out_features, in_features)

        Returns:
            Quantized ternary weights

        Why: Per-channel quantization (computing alpha per output channel) provides better
        accuracy than global quantization, especially when weight magnitudes vary significantly
        across channels. This follows the BitNet b1.58 paper's recommendation for per-row/
        per-channel scaling. Each output channel gets its own threshold, allowing the model
        to adapt quantization granularity to each channel's weight distribution.
        """
        # Compute absolute mean per output channel (dim=1)
        # Shape: (out_features, 1) for broadcasting
        alpha = weights.abs().mean(dim=1, keepdim=True)

        # Quantize to {-1, 0, 1} using per-channel thresholds
        quantized = torch.where(
            weights > alpha,
            torch.ones_like(weights),
            torch.where(weights < -alpha, -torch.ones_like(weights), torch.zeros_like(weights)),
        )
        return quantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights.

        Args:
            x: Input tensor of shape (batch_size, in_features) or (batch_size, seq_len, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features) or (batch_size, seq_len, out_features)

        Why: Implements straight-through estimator (STE) for gradient flow. During forward pass,
        weights are quantized to {-1, 0, 1}, but gradients flow through as if quantization was
        identity function. This is achieved by detaching quantized weights and adding back the
        full-precision weights for autograd. In eval mode, quantized weights are cached to avoid
        repeated quantization overhead, significantly improving inference speed.
        """
        # Use cached quantized weights in eval mode for efficiency
        if not self.training:
            if not self._cache_valid or self._quantized_weight_cache is None:
                with torch.no_grad():
                    self._quantized_weight_cache = self.quantize_weights(self.weight)
                    self._cache_valid = True
            quantized_weight = self._quantized_weight_cache
        else:
            # Training mode: quantize on the fly
            quantized_weight = self.quantize_weights(self.weight)
            # Invalidate cache if weights change during training
            self._cache_valid = False

        # Implement straight-through estimator (STE)
        # Detach quantized weights from computation graph, then add gradient path
        # through full-precision weights. This allows gradients to flow during backprop
        # while using quantized weights during forward pass.
        if self.training:
            # STE: gradient flows through self.weight, not through quantization
            quantized_weight = quantized_weight.detach() + self.weight - self.weight.detach()

        # Apply per-channel scaling
        scaled_weight = quantized_weight * self.scale

        # Linear transformation
        output = F.linear(x, scaled_weight, self.bias)
        return output


class BitNetQuantizer:
    """Utility class for quantizing standard PyTorch models to BitNet format."""

    @staticmethod
    def quantize_linear(linear: nn.Linear) -> TernaryWeight:
        """Convert a standard linear layer to ternary weights.

        Args:
            linear: Standard PyTorch Linear layer

        Returns:
            TernaryWeight layer with quantized weights
        """
        ternary = TernaryWeight(
            linear.in_features, linear.out_features, bias=linear.bias is not None
        )

        # Copy weights
        with torch.no_grad():
            ternary.weight.copy_(linear.weight)
            if linear.bias is not None and ternary.bias is not None:
                ternary.bias.copy_(linear.bias)

        return ternary

    @staticmethod
    def quantize_model(model: nn.Module) -> nn.Module:
        """Recursively quantize all linear layers in a model.

        Args:
            model: PyTorch model to quantize

        Returns:
            Model with quantized linear layers

        Why: Converts all nn.Linear layers to TernaryWeight layers for BitNet quantization.
        Recursively processes all child modules to handle nested architectures. After
        quantizing a child, continues to the next sibling rather than returning immediately,
        ensuring all linear layers at all depths are converted.
        """
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                # Replace Linear layer with TernaryWeight
                setattr(model, name, BitNetQuantizer.quantize_linear(module))
            else:
                # Recursively quantize children and update if changed
                quantized_child = BitNetQuantizer.quantize_model(module)
                if quantized_child is not module:
                    setattr(model, name, quantized_child)
                # Continue to next sibling (don't return early)

        return model
