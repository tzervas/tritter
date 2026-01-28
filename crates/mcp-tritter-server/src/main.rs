//! MCP (Model Context Protocol) server for Tritter.
//!
//! Exposes Tritter tools to Claude Code via JSON-RPC over stdio.
//!
//! Tools:
//! - tritter_config: Generate optimized TritterConfig
//! - tritter_memory: Estimate memory requirements
//! - tritter_fits: Check if model fits hardware
//! - vsa_compress: Compress gradients using VSA
//! - curate_sample: Analyze code quality

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use tracing::{debug, info};

// ============================================================================
// JSON-RPC Types
// ============================================================================

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<serde_json::Value>,
    method: String,
    #[serde(default)]
    params: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<serde_json::Value>,
}

// ============================================================================
// MCP Protocol Types
// ============================================================================

#[derive(Debug, Serialize)]
struct ServerInfo {
    name: String,
    version: String,
}

#[derive(Debug, Serialize)]
struct ServerCapabilities {
    tools: ToolsCapability,
}

#[derive(Debug, Serialize)]
struct ToolsCapability {
    #[serde(rename = "listChanged")]
    list_changed: bool,
}

#[derive(Debug, Serialize)]
struct InitializeResult {
    #[serde(rename = "protocolVersion")]
    protocol_version: String,
    capabilities: ServerCapabilities,
    #[serde(rename = "serverInfo")]
    server_info: ServerInfo,
}

#[derive(Debug, Serialize)]
struct Tool {
    name: String,
    description: String,
    #[serde(rename = "inputSchema")]
    input_schema: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct ToolListResult {
    tools: Vec<Tool>,
}

#[derive(Debug, Serialize)]
struct ToolCallResult {
    content: Vec<ToolContent>,
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    is_error: Option<bool>,
}

#[derive(Debug, Serialize)]
struct ToolContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

// ============================================================================
// Model Specs
// ============================================================================

struct ModelSpec {
    hidden_size: u64,
    num_layers: u64,
    num_heads: u64,
    num_kv_heads: u64,
    intermediate_size: u64,
    total_params_b: f64,
    packed_weights_mb: f64,
}

fn get_model_spec(model: &str) -> Option<ModelSpec> {
    match model.to_uppercase().as_str() {
        "1B" => Some(ModelSpec {
            hidden_size: 2048,
            num_layers: 16,
            num_heads: 16,
            num_kv_heads: 16,
            intermediate_size: 5632,
            total_params_b: 1.1,
            packed_weights_mb: 261.0,
        }),
        "3B" => Some(ModelSpec {
            hidden_size: 2560,
            num_layers: 26,
            num_heads: 32,
            num_kv_heads: 32,
            intermediate_size: 6912,
            total_params_b: 2.4,
            packed_weights_mb: 574.0,
        }),
        "7B" => Some(ModelSpec {
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_size: 14336,
            total_params_b: 6.2,
            packed_weights_mb: 1450.0,
        }),
        "10B" => Some(ModelSpec {
            hidden_size: 4096,
            num_layers: 40,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_size: 14336,
            total_params_b: 9.3,
            packed_weights_mb: 2160.0,
        }),
        "13B" => Some(ModelSpec {
            hidden_size: 5120,
            num_layers: 40,
            num_heads: 40,
            num_kv_heads: 8,
            intermediate_size: 13824,
            total_params_b: 11.7,
            packed_weights_mb: 2730.0,
        }),
        _ => None,
    }
}

// ============================================================================
// Tool Implementations
// ============================================================================

fn tool_tritter_config(params: &serde_json::Value) -> Result<serde_json::Value, String> {
    let model = params.get("model")
        .and_then(|v| v.as_str())
        .ok_or("Missing 'model' parameter")?;

    let vram_gb = params.get("vram_gb")
        .and_then(|v| v.as_f64())
        .unwrap_or(16.0);

    let _mode = params.get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("inference");

    let spec = get_model_spec(model)
        .ok_or_else(|| format!("Unknown model: {}. Use: 1B, 3B, 7B, 10B, 13B", model))?;

    // Estimate context based on VRAM
    let vram_mb = vram_gb * 1024.0;
    let overhead_mb = 2048.0;
    let available = vram_mb - spec.packed_weights_mb - overhead_mb;

    let head_dim = spec.hidden_size / spec.num_heads;
    let bytes_per_token = 2.0 * spec.num_layers as f64 * spec.num_kv_heads as f64 * head_dim as f64 * 0.5;
    let max_tokens = (available * 1024.0 * 1024.0 / bytes_per_token).max(4096.0) as u64;

    let context = [4096u64, 8192, 16384, 32768, 65536, 131072]
        .iter()
        .rev()
        .find(|&&c| c <= max_tokens)
        .copied()
        .unwrap_or(4096);

    let use_layer_streaming = spec.packed_weights_mb + overhead_mb > vram_mb;

    let config = serde_json::json!({
        "ok": true,
        "data": {
            "model_size": model.to_uppercase(),
            "hidden_size": spec.hidden_size,
            "num_layers": spec.num_layers,
            "num_heads": spec.num_heads,
            "num_kv_heads": spec.num_kv_heads,
            "intermediate_size": spec.intermediate_size,
            "vocab_size": 65536,
            "max_position_embeddings": context,
            "use_bitnet": true,
            "use_flash_attention": true,
            "int4_kv_cache": true,
            "use_layer_streaming": use_layer_streaming,
            "gpu_memory_budget_gb": vram_gb - 1.5,
        }
    });

    Ok(config)
}

fn tool_tritter_memory(params: &serde_json::Value) -> Result<serde_json::Value, String> {
    let model = params.get("model")
        .and_then(|v| v.as_str())
        .ok_or("Missing 'model' parameter")?;

    let context = params.get("context")
        .and_then(|v| v.as_u64())
        .unwrap_or(32768);

    let spec = get_model_spec(model)
        .ok_or_else(|| format!("Unknown model: {}", model))?;

    let params_f = spec.total_params_b * 1e9;
    let weights_packed_gb = spec.packed_weights_mb / 1024.0;
    let weights_fp16_gb = params_f * 2.0 / 1e9;

    let head_dim = spec.hidden_size / spec.num_heads;
    let kv_per_layer = 2 * spec.num_kv_heads * head_dim;
    let kv_int4_gb = (kv_per_layer as f64 * spec.num_layers as f64 * context as f64 * 0.5) / 1e9;

    let inference_total = weights_packed_gb + kv_int4_gb + 1.5;
    let qlora_total = weights_packed_gb + (params_f * 0.001 * 16.0) / 1e9 + kv_int4_gb;
    let training_total = weights_fp16_gb * 4.0 + 2.0;

    let result = serde_json::json!({
        "ok": true,
        "data": {
            "model": model.to_uppercase(),
            "context_tokens": context,
            "weights_packed_gb": weights_packed_gb,
            "kv_cache_int4_gb": kv_int4_gb,
            "inference_total_gb": inference_total,
            "qlora_training_gb": qlora_total,
            "full_training_gb": training_total,
            "fits_16gb": inference_total <= 14.0,
            "fits_24gb": inference_total <= 22.0,
        }
    });

    Ok(result)
}

fn tool_tritter_fits(params: &serde_json::Value) -> Result<serde_json::Value, String> {
    let model = params.get("model")
        .and_then(|v| v.as_str())
        .ok_or("Missing 'model' parameter")?;

    let vram_gb = params.get("vram_gb")
        .and_then(|v| v.as_f64())
        .unwrap_or(16.0);

    let spec = get_model_spec(model)
        .ok_or_else(|| format!("Unknown model: {}", model))?;

    let weights_gb = spec.packed_weights_mb / 1024.0;
    let overhead_gb = 2.5;
    let min_required = weights_gb + overhead_gb;

    let fits = min_required <= vram_gb;
    let mut notes = Vec::new();

    if fits {
        notes.push(format!("Model fits with {:.1}GB headroom", vram_gb - min_required));
    } else {
        notes.push(format!("Need {:.1}GB, have {:.1}GB", min_required, vram_gb));
        notes.push("Consider layer streaming or smaller model".to_string());
    }

    let result = serde_json::json!({
        "ok": true,
        "data": {
            "model": model.to_uppercase(),
            "vram_gb": vram_gb,
            "fits": fits,
            "min_required_gb": min_required,
            "notes": notes,
        }
    });

    Ok(result)
}

fn tool_curate_sample(params: &serde_json::Value) -> Result<serde_json::Value, String> {
    let code = params.get("code")
        .and_then(|v| v.as_str())
        .ok_or("Missing 'code' parameter")?;

    let lang = params.get("lang")
        .and_then(|v| v.as_str())
        .unwrap_or("python");

    // Simple quality heuristics
    let lines: Vec<&str> = code.lines().collect();
    let total_lines = lines.len();

    // Check for docstrings/comments
    let comment_lines = lines.iter()
        .filter(|l| l.trim().starts_with('#') || l.trim().starts_with("//") || l.trim().starts_with("///"))
        .count();
    let doc_ratio = comment_lines as f64 / total_lines.max(1) as f64;

    // Check for hardcoded secrets
    let has_secrets = code.contains("password=") || code.contains("api_key=") ||
        code.contains("SECRET") || code.contains("Bearer ");

    // Check for security issues
    let has_security_issues = code.contains("eval(") || code.contains("exec(") ||
        code.contains("shell=True") || code.contains("unsafe {");

    let mut issues = Vec::new();
    let mut quality_score: f64 = 0.7;

    if has_secrets {
        issues.push("Contains potential hardcoded secrets".to_string());
        quality_score -= 0.5;
    }
    if has_security_issues {
        issues.push("Contains potential security issues".to_string());
        quality_score -= 0.3;
    }
    if doc_ratio < 0.05 {
        issues.push("Low documentation coverage".to_string());
        quality_score -= 0.1;
    }
    if total_lines < 5 {
        issues.push("Very short code sample".to_string());
        quality_score -= 0.1;
    }

    quality_score = quality_score.clamp(0.0, 1.0);

    let label = if has_secrets {
        "reject"
    } else if quality_score >= 0.5 {
        "positive"
    } else {
        "negative"
    };

    let result = serde_json::json!({
        "ok": true,
        "data": {
            "language": lang,
            "lines": total_lines,
            "quality_score": quality_score,
            "quality_label": label,
            "issues": issues,
        }
    });

    Ok(result)
}

// ============================================================================
// Tool Registry
// ============================================================================

fn get_tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "tritter_config".to_string(),
            description: "Generate optimized TritterConfig for model + hardware".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model size (1B, 3B, 7B, 10B, 13B)",
                        "enum": ["1B", "3B", "7B", "10B", "13B"]
                    },
                    "vram_gb": {
                        "type": "number",
                        "description": "Available VRAM in GB (default: 16)",
                        "default": 16
                    },
                    "mode": {
                        "type": "string",
                        "description": "Training mode",
                        "enum": ["inference", "train", "qlora"],
                        "default": "inference"
                    }
                },
                "required": ["model"]
            }),
        },
        Tool {
            name: "tritter_memory".to_string(),
            description: "Estimate memory requirements for a Tritter model".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model size (1B, 3B, 7B, 10B, 13B)"
                    },
                    "context": {
                        "type": "integer",
                        "description": "Context length in tokens (default: 32768)",
                        "default": 32768
                    }
                },
                "required": ["model"]
            }),
        },
        Tool {
            name: "tritter_fits".to_string(),
            description: "Check if a Tritter model fits in available VRAM".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model size (1B, 3B, 7B, 10B, 13B)"
                    },
                    "vram_gb": {
                        "type": "number",
                        "description": "Available VRAM in GB (default: 16)",
                        "default": 16
                    }
                },
                "required": ["model"]
            }),
        },
        Tool {
            name: "curate_sample".to_string(),
            description: "Analyze code sample for quality and security issues".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Code sample to analyze"
                    },
                    "lang": {
                        "type": "string",
                        "description": "Programming language",
                        "default": "python"
                    }
                },
                "required": ["code"]
            }),
        },
    ]
}

fn call_tool(name: &str, params: &serde_json::Value) -> Result<serde_json::Value, String> {
    match name {
        "tritter_config" => tool_tritter_config(params),
        "tritter_memory" => tool_tritter_memory(params),
        "tritter_fits" => tool_tritter_fits(params),
        "curate_sample" => tool_curate_sample(params),
        _ => Err(format!("Unknown tool: {}", name)),
    }
}

// ============================================================================
// Request Handlers
// ============================================================================

fn handle_initialize(_params: &serde_json::Value) -> serde_json::Value {
    serde_json::to_value(InitializeResult {
        protocol_version: "2024-11-05".to_string(),
        capabilities: ServerCapabilities {
            tools: ToolsCapability { list_changed: false },
        },
        server_info: ServerInfo {
            name: "mcp-tritter-server".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        },
    }).unwrap()
}

fn handle_tools_list() -> serde_json::Value {
    serde_json::to_value(ToolListResult { tools: get_tools() }).unwrap()
}

fn handle_tools_call(params: &serde_json::Value) -> (serde_json::Value, bool) {
    let name = params.get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let arguments = params.get("arguments")
        .cloned()
        .unwrap_or(serde_json::Value::Object(Default::default()));

    match call_tool(name, &arguments) {
        Ok(result) => {
            let text = serde_json::to_string_pretty(&result).unwrap();
            let content = vec![ToolContent {
                content_type: "text".to_string(),
                text,
            }];
            (serde_json::to_value(ToolCallResult { content, is_error: None }).unwrap(), false)
        }
        Err(e) => {
            let content = vec![ToolContent {
                content_type: "text".to_string(),
                text: format!("Error: {}", e),
            }];
            (serde_json::to_value(ToolCallResult { content, is_error: Some(true) }).unwrap(), true)
        }
    }
}

fn handle_request(request: &JsonRpcRequest) -> Option<JsonRpcResponse> {
    debug!("Handling method: {}", request.method);

    let (result, error) = match request.method.as_str() {
        "initialize" => (Some(handle_initialize(&request.params)), None),
        "initialized" => return None, // Notification, no response
        "tools/list" => (Some(handle_tools_list()), None),
        "tools/call" => {
            let (result, _is_error) = handle_tools_call(&request.params);
            (Some(result), None)
        }
        "notifications/cancelled" => return None,
        "ping" => (Some(serde_json::json!({})), None),
        _ => (
            None,
            Some(JsonRpcError {
                code: -32601,
                message: format!("Method not found: {}", request.method),
                data: None,
            }),
        ),
    };

    Some(JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id: request.id.clone(),
        result,
        error,
    })
}

// ============================================================================
// Main Loop
// ============================================================================

fn main() -> anyhow::Result<()> {
    // Initialize logging to stderr
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_writer(io::stderr)
        .init();

    info!("Starting mcp-tritter-server v{}", env!("CARGO_PKG_VERSION"));

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout_handle = stdout.lock();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }

        debug!("Received: {}", line);

        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(req) => req,
            Err(e) => {
                let error_response = JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: None,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("Parse error: {}", e),
                        data: None,
                    }),
                };
                let response_json = serde_json::to_string(&error_response)?;
                writeln!(stdout_handle, "{}", response_json)?;
                stdout_handle.flush()?;
                continue;
            }
        };

        if let Some(response) = handle_request(&request) {
            let response_json = serde_json::to_string(&response)?;
            debug!("Sending: {}", response_json);
            writeln!(stdout_handle, "{}", response_json)?;
            stdout_handle.flush()?;
        }
    }

    info!("Server shutting down");
    Ok(())
}
