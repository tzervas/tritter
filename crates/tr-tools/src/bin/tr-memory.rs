//! tr-memory - Estimate memory requirements for Tritter models.
//!
//! Usage:
//!   tr-memory --model 3B
//!   tr-memory --model 7B --context 128k --detail
//!
//! Output formats: json (default), pretty table

use clap::{Parser, ValueEnum};
use serde::Serialize;
use std::io::{self, Write};

#[derive(Parser, Debug)]
#[command(name = "tr-memory")]
#[command(about = "Estimate memory requirements for Tritter models")]
#[command(version)]
struct Args {
    /// Model size (1B, 3B, 7B, 10B, 13B, 30B, 40B, 65B, 70B)
    #[arg(short, long)]
    model: String,

    /// Context length (e.g., 4k, 32k, 128k)
    #[arg(short, long, default_value = "32k")]
    context: String,

    /// Show detailed breakdown
    #[arg(short, long)]
    detail: bool,

    /// Output format
    #[arg(short, long, default_value = "json")]
    format: OutputFormat,
}

#[derive(Clone, Debug, ValueEnum)]
enum OutputFormat {
    Json,
    Pretty,
}

/// Model specification
#[derive(Debug, Clone)]
struct ModelSpec {
    hidden_size: u64,
    num_layers: u64,
    num_heads: u64,
    num_kv_heads: u64,
    total_params_b: f64,
}

/// Memory breakdown
#[derive(Debug, Serialize)]
struct MemoryBreakdown {
    model: String,
    context_tokens: u64,

    // Weight memory
    weights_fp32_gb: f64,
    weights_fp16_gb: f64,
    weights_packed_gb: f64,

    // KV-cache memory
    kv_cache_fp16_gb: f64,
    kv_cache_int4_gb: f64,

    // Training memory
    training_fp32_gb: f64,
    training_bf16_gb: f64,
    qlora_training_gb: f64,

    // Total estimates
    inference_total_gb: f64,
    training_total_gb: f64,

    // Hardware recommendations
    fits_8gb: bool,
    fits_16gb: bool,
    fits_24gb: bool,
    recommended_vram_gb: u64,
}

/// Detailed breakdown for --detail flag
#[derive(Debug, Serialize)]
struct DetailedBreakdown {
    #[serde(flatten)]
    basic: MemoryBreakdown,

    // Per-component details
    embedding_gb: f64,
    attention_gb: f64,
    ffn_gb: f64,
    head_dim: u64,
    gqa_ratio: f64,
}

/// Response wrapper
#[derive(Serialize)]
struct Response<T: Serialize> {
    ok: bool,
    data: T,
}

fn get_model_spec(model: &str) -> Result<ModelSpec, String> {
    match model.to_uppercase().as_str() {
        "1B" => Ok(ModelSpec {
            hidden_size: 2048,
            num_layers: 16,
            num_heads: 16,
            num_kv_heads: 16,
            total_params_b: 1.1,
        }),
        "3B" => Ok(ModelSpec {
            hidden_size: 2560,
            num_layers: 26,
            num_heads: 32,
            num_kv_heads: 32,
            total_params_b: 2.4,
        }),
        "7B" => Ok(ModelSpec {
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            total_params_b: 6.2,
        }),
        "10B" => Ok(ModelSpec {
            hidden_size: 4096,
            num_layers: 40,
            num_heads: 32,
            num_kv_heads: 8,
            total_params_b: 9.3,
        }),
        "13B" => Ok(ModelSpec {
            hidden_size: 5120,
            num_layers: 40,
            num_heads: 40,
            num_kv_heads: 8,
            total_params_b: 11.7,
        }),
        "30B" => Ok(ModelSpec {
            hidden_size: 6656,
            num_layers: 60,
            num_heads: 52,
            num_kv_heads: 8,
            total_params_b: 28.5,
        }),
        "40B" => Ok(ModelSpec {
            hidden_size: 8192,
            num_layers: 60,
            num_heads: 64,
            num_kv_heads: 8,
            total_params_b: 42.2,
        }),
        "65B" => Ok(ModelSpec {
            hidden_size: 8192,
            num_layers: 80,
            num_heads: 64,
            num_kv_heads: 8,
            total_params_b: 56.4,
        }),
        "70B" => Ok(ModelSpec {
            hidden_size: 8192,
            num_layers: 80,
            num_heads: 64,
            num_kv_heads: 8,
            total_params_b: 69.5,
        }),
        _ => Err(format!(
            "Unknown model size: {}. Valid: 1B, 3B, 7B, 10B, 13B, 30B, 40B, 65B, 70B",
            model
        )),
    }
}

fn parse_context(context: &str) -> Result<u64, String> {
    let ctx = context.to_lowercase();
    if let Some(num) = ctx.strip_suffix("k") {
        num.parse::<u64>()
            .map(|n| n * 1024)
            .map_err(|_| format!("Invalid context: {}", context))
    } else if let Some(num) = ctx.strip_suffix("m") {
        num.parse::<u64>()
            .map(|n| n * 1024 * 1024)
            .map_err(|_| format!("Invalid context: {}", context))
    } else {
        ctx.parse::<u64>()
            .map_err(|_| format!("Invalid context: {}", context))
    }
}

fn estimate_memory(spec: &ModelSpec, context: u64) -> MemoryBreakdown {
    let params = spec.total_params_b * 1e9;

    // Weight memory
    let weights_fp32_gb = params * 4.0 / 1e9;
    let weights_fp16_gb = params * 2.0 / 1e9;
    // Packed ternary: 2 bits per weight + scaling overhead
    let weights_packed_gb = (params * 0.25 + (params / 4096.0) * 4.0) / 1e9;

    // KV-cache memory per token
    let head_dim = spec.hidden_size / spec.num_heads;
    let kv_per_layer = 2 * spec.num_kv_heads * head_dim; // 2 for K and V

    // FP16 KV-cache
    let kv_cache_fp16_bytes = kv_per_layer as f64 * spec.num_layers as f64 * context as f64 * 2.0;
    let kv_cache_fp16_gb = kv_cache_fp16_bytes / 1e9;

    // INT4 KV-cache
    let kv_cache_int4_bytes = kv_per_layer as f64 * spec.num_layers as f64 * context as f64 * 0.5;
    let kv_cache_int4_gb = kv_cache_int4_bytes / 1e9;

    // Training memory (FP32 weights + gradients + optimizer states)
    // AdamW: weights + gradients + 2 momentum terms = 4x weights
    let training_fp32_gb = weights_fp32_gb * 4.0 + 2.0; // +2GB activations
    let training_bf16_gb = weights_fp16_gb * 2.0 + weights_fp32_gb * 2.0 + 2.0; // Mixed precision

    // QLoRA training: quantized base + LoRA adapters + optimizer for adapters
    // LoRA adds ~0.1% trainable params at rank 16
    let lora_params = params * 0.001;
    let qlora_training_gb = weights_packed_gb + (lora_params * 4.0 * 4.0) / 1e9 + kv_cache_int4_gb;

    // Total estimates
    let inference_total_gb = weights_packed_gb + kv_cache_int4_gb + 1.5; // +1.5GB overhead
    let training_total_gb = training_bf16_gb;

    // Hardware recommendations
    let fits_8gb = inference_total_gb <= 7.0;
    let fits_16gb = inference_total_gb <= 14.0;
    let fits_24gb = inference_total_gb <= 22.0;

    let recommended_vram_gb = if inference_total_gb <= 6.0 {
        8
    } else if inference_total_gb <= 14.0 {
        16
    } else if inference_total_gb <= 22.0 {
        24
    } else if inference_total_gb <= 38.0 {
        48
    } else {
        80
    };

    MemoryBreakdown {
        model: format!("{}B", spec.total_params_b),
        context_tokens: context,
        weights_fp32_gb,
        weights_fp16_gb,
        weights_packed_gb,
        kv_cache_fp16_gb,
        kv_cache_int4_gb,
        training_fp32_gb,
        training_bf16_gb,
        qlora_training_gb,
        inference_total_gb,
        training_total_gb,
        fits_8gb,
        fits_16gb,
        fits_24gb,
        recommended_vram_gb,
    }
}

fn format_gb(gb: f64) -> String {
    if gb < 1.0 {
        format!("{:.0} MB", gb * 1024.0)
    } else {
        format!("{:.2} GB", gb)
    }
}

fn format_pretty(breakdown: &MemoryBreakdown) -> String {
    let mut lines = vec![];
    lines.push(format!("Memory Estimate: {} @ {} tokens", breakdown.model, breakdown.context_tokens));
    lines.push("─".repeat(50));

    lines.push("\nWeights:".to_string());
    lines.push(format!("  FP32:           {}", format_gb(breakdown.weights_fp32_gb)));
    lines.push(format!("  FP16:           {}", format_gb(breakdown.weights_fp16_gb)));
    lines.push(format!("  Packed Ternary: {}", format_gb(breakdown.weights_packed_gb)));

    lines.push("\nKV-Cache:".to_string());
    lines.push(format!("  FP16:           {}", format_gb(breakdown.kv_cache_fp16_gb)));
    lines.push(format!("  INT4:           {}", format_gb(breakdown.kv_cache_int4_gb)));

    lines.push("\nTraining:".to_string());
    lines.push(format!("  Full FP32:      {}", format_gb(breakdown.training_fp32_gb)));
    lines.push(format!("  Mixed BF16:     {}", format_gb(breakdown.training_bf16_gb)));
    lines.push(format!("  QLoRA:          {}", format_gb(breakdown.qlora_training_gb)));

    lines.push("\nTotal Estimates:".to_string());
    lines.push(format!("  Inference:      {}", format_gb(breakdown.inference_total_gb)));
    lines.push(format!("  Training:       {}", format_gb(breakdown.training_total_gb)));

    lines.push("\nHardware Fit:".to_string());
    lines.push(format!("  8GB:  {}", if breakdown.fits_8gb { "✓" } else { "✗" }));
    lines.push(format!("  16GB: {}", if breakdown.fits_16gb { "✓" } else { "✗" }));
    lines.push(format!("  24GB: {}", if breakdown.fits_24gb { "✓" } else { "✗" }));
    lines.push(format!("  Recommended: {}GB", breakdown.recommended_vram_gb));

    lines.join("\n")
}

fn main() {
    let args = Args::parse();

    let spec = match get_model_spec(&args.model) {
        Ok(s) => s,
        Err(e) => {
            let response = serde_json::json!({
                "ok": false,
                "error": {
                    "code": "INVALID_MODEL",
                    "message": e,
                }
            });
            eprintln!("{}", serde_json::to_string_pretty(&response).unwrap());
            std::process::exit(1);
        }
    };

    let context = match parse_context(&args.context) {
        Ok(c) => c,
        Err(e) => {
            let response = serde_json::json!({
                "ok": false,
                "error": {
                    "code": "INVALID_CONTEXT",
                    "message": e,
                }
            });
            eprintln!("{}", serde_json::to_string_pretty(&response).unwrap());
            std::process::exit(1);
        }
    };

    let breakdown = estimate_memory(&spec, context);

    let stdout = io::stdout();
    let mut handle = stdout.lock();

    match args.format {
        OutputFormat::Json => {
            let response = Response {
                ok: true,
                data: breakdown,
            };
            writeln!(handle, "{}", serde_json::to_string_pretty(&response).unwrap()).unwrap();
        }
        OutputFormat::Pretty => {
            writeln!(handle, "{}", format_pretty(&breakdown)).unwrap();
        }
    }
}
