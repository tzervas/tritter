//! tr-profile - Hardware profile detection and model fit checking.
//!
//! Usage:
//!   tr-profile                    # Auto-detect GPU
//!   tr-profile --check 7B         # Check if 7B fits
//!   tr-profile --metadata > p.json # Export profile metadata

use clap::Parser;
use serde::Serialize;
use std::io::{self, Write};

#[derive(Parser, Debug)]
#[command(name = "tr-profile")]
#[command(about = "Hardware profile detection and model fit checking")]
#[command(version)]
struct Args {
    /// Check if a specific model fits
    #[arg(short, long)]
    check: Option<String>,

    /// Output detailed metadata
    #[arg(long)]
    metadata: bool,

    /// Assume specific VRAM (GB) instead of detecting
    #[arg(long)]
    vram: Option<f64>,
}

#[derive(Debug, Serialize)]
struct GpuInfo {
    name: String,
    vram_total_gb: f64,
    vram_available_gb: f64,
    compute_capability: String,
    driver_version: String,
}

#[derive(Debug, Serialize)]
struct ProfileMetadata {
    gpu: GpuInfo,
    recommended_models: Vec<String>,
    max_context_by_model: Vec<ModelContext>,
}

#[derive(Debug, Serialize)]
struct ModelContext {
    model: String,
    max_context: u64,
    inference_fits: bool,
    qlora_fits: bool,
    full_training_fits: bool,
}

#[derive(Debug, Serialize)]
struct FitCheck {
    model: String,
    vram_gb: f64,
    fits: bool,
    inference_gb: f64,
    qlora_gb: f64,
    training_gb: f64,
    notes: Vec<String>,
}

/// Response wrapper
#[derive(Serialize)]
struct Response<T: Serialize> {
    ok: bool,
    data: T,
}

fn detect_gpu() -> GpuInfo {
    // In a real implementation, this would use CUDA/NVML
    // For now, return a placeholder based on common hardware
    GpuInfo {
        name: "NVIDIA RTX 5080".to_string(),
        vram_total_gb: 16.0,
        vram_available_gb: 14.5, // Account for OS overhead
        compute_capability: "12.0".to_string(),
        driver_version: "570.0".to_string(),
    }
}

fn get_model_memory(model: &str) -> Option<(f64, f64, f64)> {
    // Returns (inference_gb, qlora_gb, training_gb)
    match model.to_uppercase().as_str() {
        "1B" => Some((1.9, 1.9, 12.2)),
        "3B" => Some((3.1, 3.1, 27.4)),
        "7B" => Some((5.5, 3.7, 60.8)),
        "10B" => Some((7.5, 4.5, 91.2)),
        "13B" => Some((9.5, 5.1, 111.8)),
        "30B" => Some((18.0, 8.5, 260.0)),
        "40B" => Some((24.0, 13.7, 397.8)),
        "65B" => Some((38.0, 18.0, 520.0)),
        "70B" => Some((45.0, 20.2, 652.1)),
        _ => None,
    }
}

fn check_model_fit(model: &str, vram_gb: f64) -> Result<FitCheck, String> {
    let (inference_gb, qlora_gb, training_gb) = get_model_memory(model)
        .ok_or_else(|| format!("Unknown model: {}", model))?;

    let fits = inference_gb <= vram_gb - 1.0; // 1GB headroom
    let mut notes = Vec::new();

    if !fits {
        notes.push(format!(
            "Inference requires {:.1}GB, you have {:.1}GB",
            inference_gb, vram_gb
        ));
        notes.push("Consider using layer streaming for larger models".to_string());
    }

    if qlora_gb <= vram_gb - 1.0 {
        notes.push(format!("QLoRA training fits ({:.1}GB)", qlora_gb));
    } else {
        notes.push(format!(
            "QLoRA requires {:.1}GB (use gradient checkpointing)",
            qlora_gb
        ));
    }

    if training_gb <= vram_gb - 2.0 {
        notes.push("Full training possible".to_string());
    } else {
        notes.push(format!(
            "Full training requires {:.1}GB - use QLoRA instead",
            training_gb
        ));
    }

    Ok(FitCheck {
        model: model.to_uppercase(),
        vram_gb,
        fits,
        inference_gb,
        qlora_gb,
        training_gb,
        notes,
    })
}

fn generate_metadata(vram_gb: f64) -> ProfileMetadata {
    let gpu = GpuInfo {
        name: "NVIDIA RTX 5080".to_string(),
        vram_total_gb: vram_gb,
        vram_available_gb: vram_gb - 1.5,
        compute_capability: "12.0".to_string(),
        driver_version: "570.0".to_string(),
    };

    let models = ["1B", "3B", "7B", "10B", "13B", "30B", "40B", "65B", "70B"];
    let mut recommended = Vec::new();
    let mut model_contexts = Vec::new();

    for model in &models {
        if let Some((inf_gb, qlora_gb, train_gb)) = get_model_memory(model) {
            let inference_fits = inf_gb <= vram_gb - 1.0;
            let qlora_fits = qlora_gb <= vram_gb - 1.0;
            let training_fits = train_gb <= vram_gb - 2.0;

            if inference_fits {
                recommended.push(model.to_string());
            }

            // Estimate max context based on remaining VRAM
            let remaining = vram_gb - inf_gb;
            let max_ctx = if remaining > 8.0 {
                131072
            } else if remaining > 4.0 {
                65536
            } else if remaining > 2.0 {
                32768
            } else if remaining > 1.0 {
                16384
            } else {
                4096
            };

            model_contexts.push(ModelContext {
                model: model.to_string(),
                max_context: max_ctx,
                inference_fits,
                qlora_fits,
                full_training_fits: training_fits,
            });
        }
    }

    ProfileMetadata {
        gpu,
        recommended_models: recommended,
        max_context_by_model: model_contexts,
    }
}

fn main() {
    let args = Args::parse();

    let vram_gb = args.vram.unwrap_or_else(|| {
        let gpu = detect_gpu();
        gpu.vram_available_gb
    });

    let stdout = io::stdout();
    let mut handle = stdout.lock();

    if let Some(model) = &args.check {
        match check_model_fit(model, vram_gb) {
            Ok(fit) => {
                let response = Response { ok: true, data: fit };
                writeln!(handle, "{}", serde_json::to_string_pretty(&response).unwrap()).unwrap();
            }
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
        }
    } else if args.metadata {
        let metadata = generate_metadata(vram_gb);
        let response = Response { ok: true, data: metadata };
        writeln!(handle, "{}", serde_json::to_string_pretty(&response).unwrap()).unwrap();
    } else {
        // Default: show summary
        let metadata = generate_metadata(vram_gb);
        writeln!(handle, "Hardware Profile").unwrap();
        writeln!(handle, "{}", "─".repeat(40)).unwrap();
        writeln!(handle, "GPU:       {}", metadata.gpu.name).unwrap();
        writeln!(handle, "VRAM:      {:.1} GB total, {:.1} GB available",
            metadata.gpu.vram_total_gb, metadata.gpu.vram_available_gb).unwrap();
        writeln!(handle, "Compute:   SM {}", metadata.gpu.compute_capability).unwrap();
        writeln!(handle).unwrap();
        writeln!(handle, "Recommended models: {}", metadata.recommended_models.join(", ")).unwrap();
        writeln!(handle).unwrap();
        writeln!(handle, "Model  | Inference | QLoRA | Full Train | Max Context").unwrap();
        writeln!(handle, "{}", "─".repeat(60)).unwrap();
        for mc in &metadata.max_context_by_model {
            let inf = if mc.inference_fits { "✓" } else { "✗" };
            let qlora = if mc.qlora_fits { "✓" } else { "✗" };
            let train = if mc.full_training_fits { "✓" } else { "✗" };
            let ctx = if mc.max_context >= 1024 {
                format!("{}K", mc.max_context / 1024)
            } else {
                format!("{}", mc.max_context)
            };
            writeln!(handle, "{:6} |     {}     |   {}   |      {}     | {}",
                mc.model, inf, qlora, train, ctx).unwrap();
        }
    }
}
