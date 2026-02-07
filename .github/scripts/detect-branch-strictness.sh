#!/usr/bin/env bash
# Tritter CI - Branch Strictness Detection
# Determines CI strictness level based on branch name and config
# Enhanced version with better YAML parsing and error handling

set -euo pipefail

# Configuration file location
CONFIG_FILE="${CI_CONFIG_FILE:-.github/ci-config.yml}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse YAML (simple grep-based parser for basic YAML)
get_yaml_value() {
    local key="$1"
    local file="${2:-$CONFIG_FILE}"
    grep "^  ${key}:" "$file" | sed 's/.*: //' | tr -d '"' | tr -d "'" | head -1
}

get_yaml_patterns() {
    local level="$1"
    local file="${2:-$CONFIG_FILE}"
    grep "^  ${level}:" "$file" | sed 's/.*: //' | tr -d '"' | tr -d "'" | head -1
}

# Get branch name (supports multiple CI systems)
get_branch() {
    # GitHub Actions
    if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
        echo "$GITHUB_BASE_REF"
    elif [[ -n "${GITHUB_REF:-}" ]]; then
        echo "${GITHUB_REF#refs/heads/}"
    # GitLab CI
    elif [[ -n "${CI_MERGE_REQUEST_TARGET_BRANCH_NAME:-}" ]]; then
        echo "$CI_MERGE_REQUEST_TARGET_BRANCH_NAME"
    elif [[ -n "${CI_COMMIT_BRANCH:-}" ]]; then
        echo "$CI_COMMIT_BRANCH"
    # Bitbucket Pipelines
    elif [[ -n "${BITBUCKET_BRANCH:-}" ]]; then
        echo "$BITBUCKET_BRANCH"
    # Jenkins
    elif [[ -n "${GIT_BRANCH:-}" ]]; then
        echo "${GIT_BRANCH#origin/}"
    # CircleCI
    elif [[ -n "${CIRCLE_BRANCH:-}" ]]; then
        echo "$CIRCLE_BRANCH"
    # Local git
    else
        git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown"
    fi
}

# Match branch against pattern (supports glob patterns)
branch_matches_pattern() {
    local branch="$1"
    local patterns="$2"

    IFS=',' read -ra PATTERNS <<< "$patterns"
    for pattern in "${PATTERNS[@]}"; do
        pattern=$(echo "$pattern" | xargs)  # Trim whitespace

        # Convert glob pattern to regex
        # * becomes .* (match any characters)
        # ? becomes . (match single character)
        pattern_regex="${pattern//\*/.*}"
        pattern_regex="${pattern_regex//\?/.}"

        if [[ "$branch" =~ ^${pattern_regex}$ ]]; then
            return 0
        fi
    done
    return 1
}

# Determine strictness level
get_strictness_level() {
    local branch="$1"

    # Get patterns from config
    local feature_patterns
    local dev_patterns
    local release_patterns
    local prod_patterns

    feature_patterns=$(get_yaml_patterns "feature")
    dev_patterns=$(get_yaml_patterns "development")
    release_patterns=$(get_yaml_patterns "release")
    prod_patterns=$(get_yaml_patterns "production")

    # Match branch against patterns (order: production > release > development > feature)
    if branch_matches_pattern "$branch" "$prod_patterns"; then
        echo "3"
    elif branch_matches_pattern "$branch" "$release_patterns"; then
        echo "2"
    elif branch_matches_pattern "$branch" "$dev_patterns"; then
        echo "1"
    elif branch_matches_pattern "$branch" "$feature_patterns"; then
        echo "0"
    else
        # Default to feature level for unknown branches
        echo "0"
    fi
}

# Get level name
get_level_name() {
    case "$1" in
        0) echo "FEATURE" ;;
        1) echo "DEVELOPMENT" ;;
        2) echo "RELEASE" ;;
        3) echo "PRODUCTION" ;;
        *) echo "UNKNOWN" ;;
    esac
}

# Get level config key
get_level_config_key() {
    case "$1" in
        0) echo "feature" ;;
        1) echo "development" ;;
        2) echo "release" ;;
        3) echo "production" ;;
        *) echo "feature" ;;
    esac
}

# Get check configuration for level
get_check_config() {
    local level="$1"
    local check="$2"
    local level_key
    level_key=$(get_level_config_key "$level")

    # Read from config file under strictness_levels section
    # First find strictness_levels:, then find the level within it
    awk -v level_key="$level_key" -v check="$check" '
        /^strictness_levels:/ { in_strictness=1; next }
        in_strictness && $0 ~ "^  " level_key ":" { in_level=1; next }
        in_level && $0 ~ "^    " check ":" {
            sub(/.*: /, "");
            gsub(/"/, "");
            gsub(/'\''/, "");
            gsub(/#.*/, "");  # Remove comments
            gsub(/^[[:space:]]+/, "");  # Trim leading whitespace
            gsub(/[[:space:]]+$/, "");  # Trim trailing whitespace
            print;
            exit
        }
        in_level && /^  [a-z]/ { exit }
        in_strictness && /^[a-z]/ { exit }
    ' "$CONFIG_FILE"
}

# Main execution
main() {
    # Check if config exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}" >&2
        echo "Please create ci-config.yml or set CI_CONFIG_FILE environment variable" >&2
        exit 1
    fi

    local branch
    branch=$(get_branch)

    local level
    level=$(get_strictness_level "$branch")

    local level_name
    level_name=$(get_level_name "$level")

    # Output as environment variables (can be eval'd by caller)
    echo "export BRANCH='$branch'"
    echo "export STRICTNESS_LEVEL=$level"
    echo "export STRICTNESS_NAME='$level_name'"

    # Output all check configurations
    for check in format lint typecheck coverage_threshold todos debug_code pedantic_lint security_audit auto_fix gpu_tests slow_tests; do
        local config
        config=$(get_check_config "$level" "$check")
        if [[ -n "$config" ]]; then
            local var_name="CHECK_${check^^}"
            echo "export ${var_name}='${config}'"
        fi
    done
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
