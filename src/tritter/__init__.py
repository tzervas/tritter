"""
Tritter: Multimodal AI with BitNet 1.58-bit quantization.

A research-driven multimodal transformer optimized for RTX 5080 with 16GB GDDR7.
Supports any-to-any (text/code/image/audio) transformations with 128K context window.
"""

__version__ = "0.1.0"
__author__ = "Tyler Zervas"
__license__ = "MIT"

from tritter.core.config import TritterConfig
from tritter.models.architecture import TritterModel

__all__ = [
    "TritterConfig",
    "TritterModel",
    "__version__",
]
