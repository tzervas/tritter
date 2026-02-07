#!/usr/bin/env bash
# Claude Code Prompt Submit Hook for Tritter
# Analyzes user prompts and suggests optimal model/effort level
# Based on task complexity

set -euo pipefail

# Get the user's prompt from stdin or first argument
USER_PROMPT="${1:-$(cat)}"

# Normalize prompt to lowercase for matching
PROMPT_LOWER=$(echo "$USER_PROMPT" | tr '[:upper:]' '[:lower:]')

# Default model and effort
SUGGESTED_MODEL="sonnet"
SUGGESTED_EFFORT="medium"
REASON=""

# Check for Opus-worthy tasks (architecture, complex optimization)
if echo "$PROMPT_LOWER" | grep -qE "(architect|design|optimize|memory budget|quantization strategy|distributed|performance critical|security audit)"; then
    SUGGESTED_MODEL="opus"
    SUGGESTED_EFFORT="high"
    REASON="Complex architecture or optimization task"

# Check for Haiku-worthy tasks (simple formatting, documentation)
elif echo "$PROMPT_LOWER" | grep -qE "(format|lint|typo|fix spelling|add comment|update readme|simple test)"; then
    SUGGESTED_MODEL="haiku"
    SUGGESTED_EFFORT="low"
    REASON="Simple, deterministic task"

# Default to Sonnet for general development
else
    SUGGESTED_MODEL="sonnet"
    SUGGESTED_EFFORT="medium"
    REASON="General development task"
fi

# Output suggestion (Claude Code will parse this)
# Format: model|effort|reason
echo "${SUGGESTED_MODEL}|${SUGGESTED_EFFORT}|${REASON}"

exit 0
