//! Shared utilities for tr-tools CLI suite.
//!
//! This module provides common functionality used across multiple CLI tools.

use serde::Serialize;

/// Standard response wrapper for JSON output.
#[derive(Serialize)]
pub struct Response<T: Serialize> {
    pub ok: bool,
    pub data: T,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<Vec<String>>,
}

/// Standard error response.
#[derive(Serialize)]
pub struct ErrorResponse {
    pub ok: bool,
    pub error: ErrorDetails,
}

#[derive(Serialize)]
pub struct ErrorDetails {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
}

impl ErrorResponse {
    pub fn new(code: &str, message: &str) -> Self {
        Self {
            ok: false,
            error: ErrorDetails {
                code: code.to_string(),
                message: message.to_string(),
                suggestion: None,
            },
        }
    }

    pub fn with_suggestion(code: &str, message: &str, suggestion: &str) -> Self {
        Self {
            ok: false,
            error: ErrorDetails {
                code: code.to_string(),
                message: message.to_string(),
                suggestion: Some(suggestion.to_string()),
            },
        }
    }
}

/// Model sizes supported by Tritter.
pub const MODEL_SIZES: &[&str] = &["1B", "3B", "7B", "10B", "13B", "30B", "40B", "65B", "70B"];

/// Format bytes as human-readable string.
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Parse context length string (e.g., "32k", "128K", "1m").
pub fn parse_context(s: &str) -> Result<u64, String> {
    let s = s.to_lowercase();
    if let Some(num) = s.strip_suffix('k') {
        num.parse::<u64>()
            .map(|n| n * 1024)
            .map_err(|_| format!("Invalid context: {}", s))
    } else if let Some(num) = s.strip_suffix('m') {
        num.parse::<u64>()
            .map(|n| n * 1024 * 1024)
            .map_err(|_| format!("Invalid context: {}", s))
    } else {
        s.parse::<u64>()
            .map_err(|_| format!("Invalid context: {}", s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_context() {
        assert_eq!(parse_context("32k").unwrap(), 32768);
        assert_eq!(parse_context("128K").unwrap(), 131072);
        assert_eq!(parse_context("1m").unwrap(), 1048576);
        assert_eq!(parse_context("4096").unwrap(), 4096);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536 * 1024), "1.50 MB");
        assert_eq!(format_bytes(2 * 1024 * 1024 * 1024), "2.00 GB");
    }
}
