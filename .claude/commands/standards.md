# Development Standards Reminder

Quick reference for Tritter development standards.

## Instructions

Remind the user of critical development standards from DEVELOPMENT_STANDARDS.md:

### Documentation Requirements
1. **Google-style docstrings with "Why" section** - Explain design decisions, not just what code does
2. **Tensor shapes in comments** - Always: `x = proj(hidden)  # (B, L, D)`
3. **Module docstrings** - Every module needs purpose + "Why" explanation

### Testing Requirements
1. **Use config values** - Never hardcode `vocab_size=1000`, use `config.vocab_size`
2. **Bounds checking** - Parameter counts need ranges, not just `> 0`
3. **Gradient verification** - Check magnitude, not just existence
4. **Test docstrings** - Every test explains what it validates

### Architecture Requirements
1. **Embedding-prediction paradigm** - Acknowledge in model/tokenization code
2. **Symmetric operations** - encode/decode must round-trip
3. **Memory constraints** - Respect RTX 5080 16GB budget
4. **`__all__` matches imports** - Verify with `from module import *`

### BitNet-Specific
1. **Squared ReLU** - Required activation for stability
2. **QK-Norm** - Query-key normalization prevents score explosion
3. **Post-FFN LayerNorm** - Chameleon-style placement
4. **vocab_size >= 264** - Minimum for byte-level encoding

### Anti-Patterns
- Hardcoded magic numbers in tests
- Weak validation (`assert params > 0`)
- Missing "Why" in docstrings
- Manual causal mask (use `is_causal=True`)
- Asymmetric encode/decode

Run `/validate` before committing to catch issues early.
