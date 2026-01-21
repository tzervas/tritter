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
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Full-precision weights for training
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        # Scaling factors for quantization
        self.scale = nn.Parameter(torch.ones(out_features, 1))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def quantize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize weights to ternary values {-1, 0, 1}.

        Args:
            weights: Full-precision weight tensor

        Returns:
            Quantized ternary weights
        """
        # Compute absolute mean for thresholding
        alpha = weights.abs().mean()

        # Quantize to {-1, 0, 1}
        quantized = torch.where(
            weights > alpha,
            torch.ones_like(weights),
            torch.where(weights < -alpha, -torch.ones_like(weights), torch.zeros_like(weights)),
        )
        return quantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Quantize weights during forward pass
        quantized_weight = self.quantize_weights(self.weight)

        # Apply scaling
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
        """
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                setattr(model, name, BitNetQuantizer.quantize_linear(module))
            else:
                BitNetQuantizer.quantize_model(module)

        return model
