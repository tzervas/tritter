//! tr-config - Generate optimized TritterConfig for model + hardware.
//!
//! Usage:
//!   tr-config --model 3B --vram 16
//!   tr-config --model 7B --mode qlora --format toml
//!
//! Output formats: json (default), toml, python

use clap::{Parser, ValueEnum};
use serde::Serialize;
use std::io::{self, Write};

#[derive(Parser, Debug)]
#[command(name = "tr-config")]
#[command(about = "Generate optimized TritterConfig for model + hardware")]
#[command(version)]
struct Args {
    /// Model size (1B, 3B, 7B, 10B, 13B, 30B, 40B, 65B, 70B)
    #[arg(short, long)]
    model: String,

    /// Available VRAM in GB (default: 16)
    #[arg(short, long, default_value = "16")]
    vram: f64,

    /// Training mode
    #[arg(long, default_value = "inference")]
    mode: Mode,

    /// Output format
    #[arg(short, long, default_value = "json")]
    format: OutputFormat,

    /// Pretty print output
    #[arg(long)]
    pretty: bool,

    /// Context length (default: auto based on VRAM)
    #[arg(short, long)]
    context: Option<u64>,
}

#[derive(Clone, Debug, ValueEnum)]
enum Mode {
    Inference,
    Train,
    Qlora,
    Lora,
}

#[derive(Clone, Debug, ValueEnum)]
enum OutputFormat {
    Json,
    Toml,
    Python,
}

/// Model specification from tritter's model_specs.py
#[derive(Debug, Clone, Serialize)]
struct ModelSpec {
    hidden_size: u64,
    num_layers: u64,
    num_heads: u64,
    num_kv_heads: u64,
    intermediate_size: u64,
    vocab_size: u64,
    total_params_b: f64,
    packed_weights_mb: f64,
}

/// Generated configuration
#[derive(Debug, Clone, Serialize)]
struct TritterConfig {
    model_size: String,
    hidden_size: u64,
    num_layers: u64,
    num_heads: u64,
    num_kv_heads: u64,
    intermediate_size: u64,
    vocab_size: u64,
    max_position_embeddings: u64,
    use_bitnet: bool,
    use_flash_attention: bool,
    int4_kv_cache: bool,
    use_layer_streaming: bool,
    layer_group_size: Option<u64>,
    gpu_memory_budget_gb: f64,
}

/// Response wrapper
#[derive(Serialize)]
struct Response {
    ok: bool,
    data: TritterConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    notes: Option<Vec<String>>,
}

fn get_model_spec(model: &str) -> Result<ModelSpec, String> {
    // Model specs from tritter/core/model_specs.py
    match model.to_uppercase().as_str() {
        "1B" => Ok(ModelSpec {
            hidden_size: 2048,
            num_layers: 16,
            num_heads: 16,
            num_kv_heads: 16,
            intermediate_size: 5632,
            vocab_size: 65536,
            total_params_b: 1.1,
            packed_weights_mb: 261.0,
        }),
        "3B" => Ok(ModelSpec {
            hidden_size: 2560,
            num_layers: 26,
            num_heads: 32,
            num_kv_heads: 32,
            intermediate_size: 6912,
            vocab_size: 65536,
            total_params_b: 2.4,
            packed_weights_mb: 574.0,
        }),
        "7B" => Ok(ModelSpec {
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_size: 14336,
            vocab_size: 65536,
            total_params_b: 6.2,
            packed_weights_mb: 1450.0,
        }),
        "10B" => Ok(ModelSpec {
            hidden_size: 4096,
            num_layers: 40,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_size: 14336,
            vocab_size: 65536,
            total_params_b: 9.3,
            packed_weights_mb: 2160.0,
        }),
        "13B" => Ok(ModelSpec {
            hidden_size: 5120,
            num_layers: 40,
            num_heads: 40,
            num_kv_heads: 8,
            intermediate_size: 13824,
            vocab_size: 65536,
            total_params_b: 11.7,
            packed_weights_mb: 2730.0,
        }),
        "30B" => Ok(ModelSpec {
            hidden_size: 6656,
            num_layers: 60,
            num_heads: 52,
            num_kv_heads: 8,
            intermediate_size: 17920,
            vocab_size: 65536,
            total_params_b: 28.5,
            packed_weights_mb: 6660.0,
        }),
        "40B" => Ok(ModelSpec {
            hidden_size: 8192,
            num_layers: 60,
            num_heads: 64,
            num_kv_heads: 8,
            intermediate_size: 22016,
            vocab_size: 65536,
            total_params_b: 42.2,
            packed_weights_mb: 9870.0,
        }),
        "65B" => Ok(ModelSpec {
            hidden_size: 8192,
            num_layers: 80,
            num_heads: 64,
            num_kv_heads: 8,
            intermediate_size: 22016,
            vocab_size: 65536,
            total_params_b: 56.4,
            packed_weights_mb: 13200.0,
        }),
        "70B" => Ok(ModelSpec {
            hidden_size: 8192,
            num_layers: 80,
            num_heads: 64,
            num_kv_heads: 8,
            intermediate_size: 28672,
            vocab_size: 65536,
            total_params_b: 69.5,
            packed_weights_mb: 16300.0,
        }),
        _ => Err(format!(
            "Unknown model size: {}. Valid: 1B, 3B, 7B, 10B, 13B, 30B, 40B, 65B, 70B",
            model
        )),
    }
}

fn estimate_kv_cache_mb(spec: &ModelSpec, context: u64, int4: bool) -> f64 {
    let bytes_per_element = if int4 { 0.5 } else { 2.0 };
    let head_dim = spec.hidden_size / spec.num_heads;
    let per_layer = 2.0 * context as f64 * spec.num_kv_heads as f64 * head_dim as f64 * bytes_per_element;
    (spec.num_layers as f64 * per_layer) / (1024.0 * 1024.0)
}

fn generate_config(args: &Args) -> Result<(TritterConfig, Vec<String>), String> {
    let spec = get_model_spec(&args.model)?;
    let mut notes = Vec::new();

    // Determine optimal context length based on VRAM
    let vram_mb = args.vram * 1024.0;
    let weights_mb = spec.packed_weights_mb;
    let overhead_mb = 2048.0; // Activations, overhead

    let available_for_kv = vram_mb - weights_mb - overhead_mb;

    // Calculate max context with INT4 KV-cache
    let context = args.context.unwrap_or_else(|| {
        let head_dim = spec.hidden_size / spec.num_heads;
        let bytes_per_token = 2.0 * spec.num_layers as f64 * spec.num_kv_heads as f64 * head_dim as f64 * 0.5;
        let max_tokens = (available_for_kv * 1024.0 * 1024.0) / bytes_per_token;
        let max_tokens = max_tokens.max(1024.0) as u64;

        // Round to power of 2
        let ctx = [4096u64, 8192, 16384, 32768, 65536, 131072]
            .iter()
            .rev()
            .find(|&&c| c <= max_tokens)
            .copied()
            .unwrap_or(4096);

        ctx
    });

    // Determine if layer streaming is needed
    let kv_mb = estimate_kv_cache_mb(&spec, context, true);
    let total_mb = weights_mb + kv_mb + overhead_mb;
    let use_layer_streaming = total_mb > vram_mb;

    if use_layer_streaming {
        notes.push(format!(
            "Layer streaming enabled: model ({:.0} MB) + KV ({:.0} MB) > VRAM ({:.0} MB)",
            weights_mb, kv_mb, vram_mb
        ));
    }

    // Determine layer group size for streaming
    let layer_group_size = if use_layer_streaming {
        Some(4)
    } else {
        None
    };

    // Training mode adjustments
    let gpu_budget = match args.mode {
        Mode::Inference => args.vram - 1.0, // 1GB headroom
        Mode::Train => args.vram - 2.0,     // 2GB headroom for gradients
        Mode::Qlora | Mode::Lora => args.vram - 1.5,
    };

    if matches!(args.mode, Mode::Train) && spec.total_params_b > 3.0 {
        notes.push("Full training of 7B+ requires >40GB VRAM. Consider QLoRA.".to_string());
    }

    let config = TritterConfig {
        model_size: args.model.to_uppercase(),
        hidden_size: spec.hidden_size,
        num_layers: spec.num_layers,
        num_heads: spec.num_heads,
        num_kv_heads: spec.num_kv_heads,
        intermediate_size: spec.intermediate_size,
        vocab_size: spec.vocab_size,
        max_position_embeddings: context,
        use_bitnet: true,
        use_flash_attention: true,
        int4_kv_cache: true,
        use_layer_streaming,
        layer_group_size,
        gpu_memory_budget_gb: gpu_budget.max(1.0),
    };

    Ok((config, notes))
}

fn format_output(config: &TritterConfig, notes: Vec<String>, format: &OutputFormat, pretty: bool) -> String {
    match format {
        OutputFormat::Json => {
            let response = Response {
                ok: true,
                data: config.clone(),
                notes: if notes.is_empty() { None } else { Some(notes) },
            };
            if pretty {
                serde_json::to_string_pretty(&response).unwrap()
            } else {
                serde_json::to_string(&response).unwrap()
            }
        }
        OutputFormat::Toml => {
            toml::to_string_pretty(config).unwrap()
        }
        OutputFormat::Python => {
            let mut lines = vec![
                "from tritter import TritterConfig".to_string(),
                "".to_string(),
                "config = TritterConfig(".to_string(),
            ];
            lines.push(format!("    model_size=\"{}\",", config.model_size));
            lines.push(format!("    hidden_size={},", config.hidden_size));
            lines.push(format!("    num_layers={},", config.num_layers));
            lines.push(format!("    num_heads={},", config.num_heads));
            lines.push(format!("    num_kv_heads={},", config.num_kv_heads));
            lines.push(format!("    intermediate_size={},", config.intermediate_size));
            lines.push(format!("    vocab_size={},", config.vocab_size));
            lines.push(format!("    max_position_embeddings={},", config.max_position_embeddings));
            lines.push(format!("    use_bitnet={},", if config.use_bitnet { "True" } else { "False" }));
            lines.push(format!("    use_flash_attention={},", if config.use_flash_attention { "True" } else { "False" }));
            lines.push(format!("    int4_kv_cache={},", if config.int4_kv_cache { "True" } else { "False" }));
            lines.push(format!("    use_layer_streaming={},", if config.use_layer_streaming { "True" } else { "False" }));
            if let Some(group_size) = config.layer_group_size {
                lines.push(format!("    layer_group_size={},", group_size));
            }
            lines.push(format!("    gpu_memory_budget_gb={:.1},", config.gpu_memory_budget_gb));
            lines.push(")".to_string());

            if !notes.is_empty() {
                lines.push("".to_string());
                lines.push("# Notes:".to_string());
                for note in notes {
                    lines.push(format!("# - {}", note));
                }
            }

            lines.join("\n")
        }
    }
}

fn main() {
    let args = Args::parse();

    match generate_config(&args) {
        Ok((config, notes)) => {
            let output = format_output(&config, notes, &args.format, args.pretty);
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            writeln!(handle, "{}", output).unwrap();
        }
        Err(e) => {
            let response = serde_json::json!({
                "ok": false,
                "error": {
                    "code": "INVALID_MODEL",
                    "message": e,
                    "suggestion": "Use one of: 1B, 3B, 7B, 10B, 13B, 30B, 40B, 65B, 70B"
                }
            });
            eprintln!("{}", serde_json::to_string_pretty(&response).unwrap());
            std::process::exit(1);
        }
    }
}
