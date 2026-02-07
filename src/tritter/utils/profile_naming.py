"""Profile naming helpers for accelerated training and inference.

Why: Tritter experiments combine multiple acceleration backends, quantization
methods, and training regimes. A consistent, machine-readable naming convention
reduces confusion when comparing runs and enables automated profile selection
and reporting.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from tritter.core.config import TritterConfig
from tritter.core.model_specs import MemoryEstimate, ModelSpec, estimate_memory, get_model_spec

_VALID_TRAINING_MODES = {"pretrain", "finetune", "lora", "qlora", "inference"}


@dataclass(frozen=True)
class ProfileNamingMetadata:
    """Structured profile naming metadata.

    Why: A structured representation keeps naming deterministic and enables
    stable downstream reporting or serialization without parsing strings.
    """

    model_size: str
    param_count_b: float
    training_mode: str
    quantization: str
    kv_cache: str
    attention: str
    attention_mode: str
    context_length: int
    sliding_window: str
    streaming: str
    layer_streaming: str
    optimizer: str
    vsa_mode: str
    accel_profile: str
    accel_features: tuple[str, ...]
    vram_raw_gb: float | None
    vram_loaded_gb: float | None

    def to_dict(self) -> dict[str, str | float | None | list[str]]:
        """Serialize metadata to a JSON-friendly dictionary.

        Why: Training artifacts and dashboards require structured payloads
        rather than parsing a formatted profile name.
        """
        return {
            "model_size": self.model_size,
            "param_count_b": self.param_count_b,
            "training_mode": self.training_mode,
            "quantization": self.quantization,
            "kv_cache": self.kv_cache,
            "attention": self.attention,
            "attention_mode": self.attention_mode,
            "context_length": self.context_length,
            "sliding_window": self.sliding_window,
            "streaming": self.streaming,
            "layer_streaming": self.layer_streaming,
            "optimizer": self.optimizer,
            "vsa_mode": self.vsa_mode,
            "accel_profile": self.accel_profile,
            "accel_features": list(self.accel_features),
            "vram_raw_gb": self.vram_raw_gb,
            "vram_loaded_gb": self.vram_loaded_gb,
        }


def resolve_accel_profile(preferred: str | None) -> str:
    """Resolve the acceleration profile name.

    Args:
        preferred: Preferred profile name or "auto" for automatic detection

    Returns:
        Resolved profile name

    Why: A deterministic resolution path makes profile naming consistent across
    CLI use, training scripts, and downstream tooling.
    """
    env_profile = os.getenv("TRITTER_ACCEL_PROFILE")
    if preferred and preferred.lower() not in {"auto", ""}:
        return preferred
    if env_profile:
        return env_profile
    if importlib.util.find_spec("tritter_accel") is not None:
        return "accelerated"
    return "baseline"


def resolve_accel_feature_flags(flags: Iterable[str] | None) -> tuple[str, ...]:
    """Resolve acceleration feature flags from config and environment.

    Args:
        flags: Preferred feature flags from configuration

    Returns:
        Normalized feature flag tuple

    Why: Feature flags enable rapid experimentation without changing configs.
    Environment overrides allow quick A/B tests on shared runners.
    """
    env_flags = os.getenv("TRITTER_ACCEL_FEATURES", "")
    env_parts = [part.strip() for part in env_flags.split(",") if part.strip()]
    combined = list(flags or []) + env_parts
    normalized = sorted({_sanitize_tag(flag) for flag in combined if flag})
    return tuple(normalized)


def normalize_training_mode(training_mode: str | None) -> str:
    """Normalize training mode to a supported value.

    Args:
        training_mode: Requested training mode

    Returns:
        Normalized training mode string

    Why: Ensures consistent naming tags regardless of user input casing or
    shorthand usage.
    """
    if not training_mode:
        return "pretrain"
    normalized = training_mode.strip().lower()
    if normalized not in _VALID_TRAINING_MODES:
        return "pretrain"
    return normalized


def build_profile_metadata(
    config: TritterConfig,
    training_mode: str | None = None,
    use_amp: bool = True,
    vram_raw_gb: float | None = None,
    vram_loaded_gb: float | None = None,
    optimizer_name: str | None = None,
    tag_overrides: Mapping[str, str | float | list[str] | tuple[str, ...]] | None = None,
) -> ProfileNamingMetadata:
    """Build profile metadata for deterministic naming.

    Args:
        config: Tritter configuration
        training_mode: Training mode tag (pretrain/finetune/lora/qlora)
        use_amp: Whether mixed precision is enabled (affects memory estimate)
        vram_raw_gb: Measured total GPU VRAM (if available)
        vram_loaded_gb: Measured loaded VRAM (if available)
        tag_overrides: Optional overrides for metadata fields

    Returns:
        ProfileNamingMetadata instance

    Why: Centralized metadata construction keeps the naming convention in one
    place and ensures overrides behave consistently.
    """
    spec = get_model_spec(config.model_size)
    params_b = spec.total_params_billions()
    memory = estimate_memory(spec)

    resolved_training = normalize_training_mode(training_mode)
    quantization = "bitnet1p58" if config.use_bitnet else "fp16"
    kv_cache = "int4" if config.int4_kv_cache else "fp16"
    attention = "flash" if config.use_flash_attention else "sdpa"
    attention_mode = config.attention_mode
    context_length = config.max_position_embeddings
    sliding_window = "off"
    if config.use_sliding_window:
        window_size = config.sliding_window_size or 0
        sliding_window = f"{window_size}"
    streaming = "on" if config.use_streaming_llm else "off"
    layer_streaming = "on" if config.use_layer_streaming else "off"
    optimizer = optimizer_name or "adamw"
    vsa_mode = "ternary" if config.use_vsa else "off"
    if config.use_hrr:
        vsa_mode = "hrr"
    accel_profile = resolve_accel_profile(config.accel_profile)
    accel_features = resolve_accel_feature_flags(config.accel_feature_flags)

    if vram_loaded_gb is None or vram_loaded_gb <= 0:
        vram_loaded_gb = _estimate_loaded_vram_gb(
            spec,
            memory,
            resolved_training,
            use_amp,
            quantization,
            kv_cache,
            context_length,
        )

    metadata = ProfileNamingMetadata(
        model_size=config.model_size,
        param_count_b=params_b,
        training_mode=resolved_training,
        quantization=quantization,
        kv_cache=kv_cache,
        attention=attention,
        attention_mode=attention_mode,
        context_length=context_length,
        sliding_window=sliding_window,
        streaming=streaming,
        layer_streaming=layer_streaming,
        optimizer=optimizer,
        vsa_mode=vsa_mode,
        accel_profile=accel_profile,
        accel_features=accel_features,
        vram_raw_gb=vram_raw_gb,
        vram_loaded_gb=vram_loaded_gb,
    )

    if tag_overrides:
        metadata = _apply_tag_overrides(metadata, tag_overrides)

    return metadata


def resolve_profile_name(
    config: TritterConfig,
    training_mode: str | None = None,
    use_amp: bool = True,
    vram_raw_gb: float | None = None,
    vram_loaded_gb: float | None = None,
    name_override: str | None = None,
    optimizer_name: str | None = None,
    tag_overrides: Mapping[str, str | float | list[str] | tuple[str, ...]] | None = None,
) -> tuple[str, ProfileNamingMetadata]:
    """Resolve the profile name and metadata.

    Args:
        config: Tritter configuration
        training_mode: Training mode tag
        use_amp: Whether mixed precision is enabled
        vram_raw_gb: Measured total VRAM
        vram_loaded_gb: Measured loaded VRAM
        name_override: Optional explicit name override
        tag_overrides: Optional metadata overrides

    Returns:
        Tuple of (profile_name, metadata)

    Why: Keeps the override and metadata logic in one place so downstream
    callers do not need to reimplement naming logic.
    """
    metadata = build_profile_metadata(
        config=config,
        training_mode=training_mode,
        use_amp=use_amp,
        vram_raw_gb=vram_raw_gb,
        vram_loaded_gb=vram_loaded_gb,
        optimizer_name=optimizer_name,
        tag_overrides=tag_overrides,
    )

    if name_override:
        return name_override, metadata

    return format_profile_name(metadata), metadata


def format_profile_name(metadata: ProfileNamingMetadata) -> str:
    """Format a human-readable profile name.

    Args:
        metadata: Profile naming metadata

    Returns:
        Formatted profile name string

    Why: Encodes model size, training mode, quantization, and VRAM metrics in a
    single identifier for easy experiment tracking.
    """
    model_tag = _sanitize_tag(metadata.model_size)
    params_tag = f"p{metadata.param_count_b:.2f}b"
    train_tag = f"train={_sanitize_tag(metadata.training_mode)}"
    quant_tag = f"quant={_sanitize_tag(metadata.quantization)}"
    kv_tag = f"kv={_sanitize_tag(metadata.kv_cache)}"
    attn_tag = f"attn={_sanitize_tag(metadata.attention)}"
    attn_mode_tag = f"attnmode={_sanitize_tag(metadata.attention_mode)}"
    ctx_tag = f"ctx={_format_context(metadata.context_length)}"
    slide_tag = f"slide={_sanitize_tag(metadata.sliding_window)}"
    stream_tag = f"stream={_sanitize_tag(metadata.streaming)}"
    layer_stream_tag = f"layerstream={_sanitize_tag(metadata.layer_streaming)}"
    optim_tag = f"optim={_sanitize_tag(metadata.optimizer)}"
    vsa_tag = f"vsa={_sanitize_tag(metadata.vsa_mode)}"
    accel_tag = f"accel={_sanitize_tag(metadata.accel_profile)}"

    features_tag = ""
    if metadata.accel_features:
        feature_blob = "+".join(_sanitize_tag(flag) for flag in metadata.accel_features)
        features_tag = f"-feat={feature_blob}"

    vram_raw = _format_vram(metadata.vram_raw_gb)
    vram_loaded = _format_vram(metadata.vram_loaded_gb)
    vram_tag = f"vram{vram_raw}g-load{vram_loaded}g"

    return (
        f"tritter-{model_tag}-{params_tag}-{train_tag}-{quant_tag}-{kv_tag}-"
        f"{attn_tag}-{attn_mode_tag}-{ctx_tag}-{slide_tag}-{stream_tag}-"
        f"{layer_stream_tag}-{optim_tag}-{vsa_tag}-{accel_tag}{features_tag}-{vram_tag}"
    )


def _estimate_loaded_vram_gb(
    spec: ModelSpec,
    memory: MemoryEstimate,
    training_mode: str,
    use_amp: bool,
    quantization: str,
    kv_cache: str,
    context_length: int,
) -> float:
    """Estimate loaded VRAM usage in GB.

    Args:
        memory: MemoryEstimate instance
        training_mode: Training mode tag
        use_amp: Whether mixed precision is enabled
        quantization: Quantization tag
        kv_cache: KV-cache tag

    Returns:
        Estimated loaded VRAM in GB

    Why: Provides a fallback estimate when measured logs are unavailable, keeping
    naming deterministic for automated pipelines.
    """
    # mypy: MemoryEstimate type is available but passed as object to avoid import cycle.
    kv_bytes = _estimate_kv_cache_bytes(spec, context_length, kv_cache)
    if training_mode in {"pretrain", "finetune"}:
        bytes_required = memory.training_bf16_mixed if use_amp else memory.training_fp32
    elif training_mode in {"lora", "qlora"}:
        if quantization == "bitnet1p58":
            weight_bytes = memory.weights_packed_ternary
        else:
            weight_bytes = memory.weights_fp16
        bytes_required = weight_bytes + kv_bytes
    else:
        if quantization == "bitnet1p58":
            weight_bytes = memory.weights_packed_ternary
        else:
            weight_bytes = memory.weights_fp16
        bytes_required = weight_bytes + kv_bytes

    return bytes_required / (1024**3)


def _apply_tag_overrides(
    metadata: ProfileNamingMetadata,
    overrides: Mapping[str, str | float | list[str] | tuple[str, ...]],
) -> ProfileNamingMetadata:
    """Apply override values to profile metadata.

    Args:
        metadata: Existing metadata
        overrides: Override mapping

    Returns:
        Updated metadata

    Why: Users may want to override specific tags without redefining the full
    naming payload.
    """
    data: dict[str, Any] = dict(metadata.to_dict())
    for key, value in overrides.items():
        if key in data:
            data[key] = value

    accel_features_raw = data.get("accel_features")
    if isinstance(accel_features_raw, (list, tuple)):
        resolved_features = tuple(str(feature) for feature in accel_features_raw)
    else:
        resolved_features = tuple(metadata.accel_features)

    return ProfileNamingMetadata(
        model_size=str(data.get("model_size", metadata.model_size)),
        param_count_b=float(data.get("param_count_b", metadata.param_count_b)),
        training_mode=str(data.get("training_mode", metadata.training_mode)),
        quantization=str(data.get("quantization", metadata.quantization)),
        kv_cache=str(data.get("kv_cache", metadata.kv_cache)),
        attention=str(data.get("attention", metadata.attention)),
        attention_mode=str(data.get("attention_mode", metadata.attention_mode)),
        context_length=int(data.get("context_length", metadata.context_length)),
        sliding_window=str(data.get("sliding_window", metadata.sliding_window)),
        streaming=str(data.get("streaming", metadata.streaming)),
        layer_streaming=str(data.get("layer_streaming", metadata.layer_streaming)),
        optimizer=str(data.get("optimizer", metadata.optimizer)),
        vsa_mode=str(data.get("vsa_mode", metadata.vsa_mode)),
        accel_profile=str(data.get("accel_profile", metadata.accel_profile)),
        accel_features=resolved_features,
        vram_raw_gb=_coerce_optional_float(data.get("vram_raw_gb", metadata.vram_raw_gb)),
        vram_loaded_gb=_coerce_optional_float(data.get("vram_loaded_gb", metadata.vram_loaded_gb)),
    )


def _coerce_optional_float(value: float | str | None) -> float | None:
    """Coerce optional float values for overrides.

    Args:
        value: Value to convert

    Returns:
        Float or None

    Why: Override maps may supply strings; coercion keeps metadata consistent.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except ValueError:
        return None


def _format_vram(value: float | None) -> str:
    """Format VRAM values for tags.

    Args:
        value: VRAM value in GB

    Returns:
        Formatted VRAM string

    Why: Keeps VRAM tags consistent and readable even when values are missing.
    """
    if value is None or value <= 0:
        return "na"
    return f"{value:.1f}"


def _sanitize_tag(value: str) -> str:
    """Sanitize tag values for profile names.

    Args:
        value: Raw tag value

    Returns:
        Sanitized tag string

    Why: Ensures profile names are safe for filenames and CLI usage.
    """
    return value.strip().lower().replace(" ", "-").replace("/", "-").replace("_", "-")


def _format_context(context_length: int) -> str:
    """Format context length tag.

    Args:
        context_length: Context length in tokens

    Returns:
        Context length tag string

    Why: Keeps context tags compact while preserving useful scale information.
    """
    if context_length >= 1024:
        return f"{context_length // 1024}k"
    return str(context_length)


def _estimate_kv_cache_bytes(spec: ModelSpec, context_length: int, kv_cache: str) -> int:
    """Estimate KV-cache bytes for the requested context length.

    Args:
        spec: ModelSpec instance
        context_length: Context length in tokens
        kv_cache: KV-cache precision tag

    Returns:
        Estimated KV-cache size in bytes

    Why: Context length directly impacts memory usage; this ensures profile
    names report VRAM requirements for the chosen window.
    """
    kv_heads = spec.effective_num_kv_heads
    head_dim = spec.effective_head_dim
    dtype_bytes = 0.5 if kv_cache == "int4" else 2.0
    per_layer = int(2 * context_length * kv_heads * head_dim * dtype_bytes)
    return int(spec.num_layers * per_layer)
