# Tritter Rust Acceleration Crates

This directory contains Rust crates for accelerating Tritter operations and providing Claude Code integration.

## Crates

### tritter-accel

PyO3 bindings exposing high-performance Rust operations to Python.

**Functions:**
- `pack_ternary_weights` - Pack ternary weights into 2-bit format (4x memory reduction)
- `unpack_ternary_weights` - Unpack ternary weights
- `ternary_matmul` - Matrix multiply with packed ternary weights (20-25x speedup)
- `quantize_weights_absmean` - AbsMean quantization for BitNet
- `compress_gradients_vsa` - VSA gradient compression (~90% reduction)
- `decompress_gradients_vsa` - VSA gradient decompression

**Build:**
```bash
cd tritter-accel
maturin develop --release
```

### tr-tools

CLI tools for configuration, memory estimation, and hardware profiling.

**Commands:**
- `tr-config` - Generate optimized TritterConfig for model + hardware
- `tr-memory` - Estimate memory requirements
- `tr-profile` - Hardware profile detection and model fit checking

**Build:**
```bash
cargo build --release -p tr-tools
```

**Usage:**
```bash
# Generate config for 1B model on 16GB GPU
./target/release/tr-config --model 1B --vram 16

# Estimate memory for 3B model
./target/release/tr-memory --model 3B --format pretty

# Check hardware profile
./target/release/tr-profile --vram 16
```

### mcp-tritter-server

MCP (Model Context Protocol) server exposing Tritter tools to Claude Code.

**Tools:**
- `tritter_config` - Generate optimized TritterConfig
- `tritter_memory` - Estimate memory requirements
- `tritter_fits` - Check if model fits hardware
- `curate_sample` - Analyze code quality

**Build:**
```bash
cargo build --release -p mcp-tritter-server
```

**Claude Code Integration:**

Add to `~/.claude/mcp.json`:
```json
{
  "mcpServers": {
    "tritter": {
      "command": "/path/to/tritter/crates/target/release/mcp-tritter-server"
    }
  }
}
```

## Dependencies

All crates use the rust-ai ecosystem from crates.io (github.com/tzervas):
- `rust-ai-core` - Device selection, config validation
- `bitnet-quantize` - BitNet 1.58-bit quantization
- `trit-vsa` - Ternary arithmetic, VSA operations
- `vsa-optim-rs` - Gradient compression

## Development

```bash
# Check all crates
cargo check

# Build all crates
cargo build --release

# Run tests
cargo test
```
