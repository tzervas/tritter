#!/usr/bin/env bash
# Tritter CI - Main CI Runner
# Executes progressive CI checks based on branch strictness level
# Enhanced with Python-specific checks and Tritter validations

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CI_CONFIG_FILE:-.github/ci-config.yml}"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Track overall status
EXIT_CODE=0
CHECKS_RUN=0
CHECKS_PASSED=0
CHECKS_WARNED=0
CHECKS_FAILED=0

# Load configuration
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Detect strictness level
if [[ -f "$SCRIPT_DIR/detect-branch-strictness.sh" ]]; then
    eval "$(bash "$SCRIPT_DIR/detect-branch-strictness.sh")"
else
    echo -e "${RED}Error: detect-branch-strictness.sh not found${NC}"
    exit 1
fi

# Print header
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${BLUE}  Tritter CI - Progressive Quality Gates${NC}"
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "Project: ${CYAN}Tritter${NC}"
echo -e "Branch: ${GREEN}${BRANCH}${NC}"
echo -e "Strictness Level: ${YELLOW}Level ${STRICTNESS_LEVEL} (${STRICTNESS_NAME})${NC}"
echo -e "Config: ${CYAN}${CONFIG_FILE}${NC}"
echo ""

# Get project language
PROJECT_LANGUAGE=$(awk '/^  language:/ { sub(/.*: /, ""); gsub(/"/, ""); gsub(/'\''/, ""); print; exit }' "$CONFIG_FILE")

# Get command for specific check
get_command() {
    local check="$1"
    awk -v lang="  ${PROJECT_LANGUAGE}:" -v check="    $check:" '
        $0 ~ lang { found=1; next }
        found && $0 ~ check {
            sub(/.*: /, "");
            gsub(/"/, "");
            gsub(/'\''/, "");
            print;
            exit
        }
        found && /^  [a-z]/ && $0 !~ check { exit }
    ' "$CONFIG_FILE"
}

# Execute check based on strictness level
run_check() {
    local check_name="$1"
    local check_config_var="$2"
    local check_command="$3"

    ((CHECKS_RUN++))

    local check_level="${!check_config_var:-skip}"

    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}Check: ${check_name}${NC}"
    echo -e "Config: ${YELLOW}${check_level}${NC}"

    if [[ "$check_level" == "skip" ]]; then
        echo -e "${YELLOW}⏭️  Skipped (configured to skip for ${STRICTNESS_NAME} level)${NC}"
        echo ""
        return 0
    fi

    if [[ -z "$check_command" ]] || [[ "$check_command" == "echo"* ]]; then
        echo -e "${YELLOW}⚠️  No command configured${NC}"
        echo ""
        return 0
    fi

    echo -e "Command: ${CYAN}${check_command}${NC}"
    echo ""

    local output
    local exit_code=0

    # Run command and capture output
    cd "$PROJECT_ROOT"
    output=$(eval "$check_command" 2>&1) || exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}✅ Passed${NC}"
        [[ -n "$output" ]] && echo "$output"
        ((CHECKS_PASSED++))
        echo ""
        return 0
    else
        if [[ "$check_level" == "warn" ]]; then
            echo -e "${YELLOW}${BOLD}⚠️  Warning${NC} (not blocking for ${STRICTNESS_NAME} level)"
            [[ -n "$output" ]] && echo "$output"
            ((CHECKS_WARNED++))
            echo ""
            return 0
        else
            echo -e "${RED}${BOLD}❌ Failed${NC}"
            [[ -n "$output" ]] && echo "$output"
            ((CHECKS_FAILED++))
            EXIT_CODE=1
            echo ""
            return 1
        fi
    fi
}

# Run coverage check
run_coverage_check() {
    ((CHECKS_RUN++))

    local coverage_command
    coverage_command=$(get_command "coverage")

    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}Check: Test Coverage${NC}"
    echo -e "Required: ${YELLOW}${CHECK_COVERAGE_THRESHOLD}%${NC}"

    if [[ -z "$coverage_command" ]] || [[ "$coverage_command" == "echo"* ]]; then
        echo -e "${YELLOW}⚠️  No coverage command configured${NC}"
        echo ""
        return 0
    fi

    echo -e "Command: ${CYAN}${coverage_command}${NC}"
    echo ""

    cd "$PROJECT_ROOT"
    eval "$coverage_command" || true

    # Parse coverage (Python-specific: coverage.json)
    local coverage_percent=0
    if [[ -f "coverage.json" ]]; then
        coverage_percent=$(python3 -c "import json; data=json.load(open('coverage.json')); print(f\"{data['totals']['percent_covered']:.2f}\")" 2>/dev/null || echo "0")
    elif [[ -f ".coverage" ]]; then
        # Try using coverage CLI if JSON not available
        coverage_percent=$(coverage report --precision=2 2>/dev/null | grep TOTAL | awk '{print $NF}' | sed 's/%//' || echo "0")
    fi

    echo -e "Coverage: ${BOLD}${coverage_percent}%${NC}"

    # Validate prerequisites for coverage comparison
    if [[ -z "${CHECK_COVERAGE_THRESHOLD:-}" ]]; then
        echo -e "${YELLOW}⚠️  Coverage threshold not configured, skipping check${NC}"
        ((CHECKS_WARNED++))
        echo ""
        return 0
    fi

    # Check if bc is available for floating point comparison
    if ! command -v bc &> /dev/null; then
        echo -e "${YELLOW}⚠️  bc command not found, cannot compare coverage (install: apt-get install bc)${NC}"
        ((CHECKS_WARNED++))
        echo ""
        return 0
    fi

    # Compare coverage (safe now - tools and threshold verified)
    if (( $(echo "$coverage_percent < ${CHECK_COVERAGE_THRESHOLD}" | bc -l) )); then
        if [[ "$STRICTNESS_LEVEL" == "0" ]]; then
            echo -e "${YELLOW}${BOLD}⚠️  Warning${NC}: Coverage below ${CHECK_COVERAGE_THRESHOLD}% (acceptable for ${STRICTNESS_NAME} level)"
            ((CHECKS_WARNED++))
            echo ""
            return 0
        else
            echo -e "${RED}${BOLD}❌ Failed${NC}: Coverage below required ${CHECK_COVERAGE_THRESHOLD}%"
            ((CHECKS_FAILED++))
            EXIT_CODE=1
            echo ""
            return 1
        fi
    else
        echo -e "${GREEN}${BOLD}✅ Passed${NC}: Coverage meets threshold"
        ((CHECKS_PASSED++))
        echo ""
        return 0
    fi
}

# Run custom checks from config
run_custom_checks() {
    # Check if custom_checks section exists
    if ! grep -q "^custom_checks:" "$CONFIG_FILE"; then
        return 0
    fi

    # Parse custom checks (simplified YAML parsing)
    local in_custom_checks=0
    local current_check=""
    local current_command=""
    local current_level_config=""

    while IFS= read -r line; do
        if [[ "$line" =~ ^custom_checks: ]]; then
            in_custom_checks=1
            continue
        fi

        if [[ $in_custom_checks -eq 1 ]]; then
            # Stop if we hit a new top-level section
            if [[ "$line" =~ ^[a-z_]+: ]] && [[ ! "$line" =~ ^[[:space:]] ]]; then
                break
            fi

            # New check definition
            if [[ "$line" =~ ^[[:space:]]{2}[a-z_]+: ]] && [[ ! "$line" =~ command: ]]; then
                # Save previous check if exists
                if [[ -n "$current_check" ]] && [[ -n "$current_command" ]]; then
                    run_custom_check "$current_check" "$current_command" "$current_level_config"
                fi

                current_check=$(echo "$line" | sed 's/^[[:space:]]*//;s/:.*//')
                current_command=""
                current_level_config=""
            elif [[ "$line" =~ command: ]]; then
                current_command=$(echo "$line" | sed 's/.*command:[[:space:]]*//;s/"//g;s/'\''//g')
            elif [[ "$line" =~ [[:space:]]{4}${STRICTNESS_LEVEL}: ]] || [[ "$line" =~ [[:space:]]{4}$(get_level_config_key "$STRICTNESS_LEVEL"): ]]; then
                current_level_config=$(echo "$line" | sed 's/.*:[[:space:]]*//;s/"//g;s/'\''//g')
            fi
        fi
    done < "$CONFIG_FILE"

    # Run last check if exists
    if [[ -n "$current_check" ]] && [[ -n "$current_command" ]]; then
        run_custom_check "$current_check" "$current_command" "$current_level_config"
    fi
}

# Run a single custom check
run_custom_check() {
    local check_name="$1"
    local check_command="$2"
    local check_level="${3:-skip}"

    ((CHECKS_RUN++))

    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}Custom Check: ${check_name}${NC}"
    echo -e "Config: ${YELLOW}${check_level}${NC}"

    if [[ "$check_level" == "skip" ]]; then
        echo -e "${YELLOW}⏭️  Skipped${NC}"
        echo ""
        return 0
    fi

    echo -e "Command: ${CYAN}${check_command}${NC}"
    echo ""

    local output
    local exit_code=0

    cd "$PROJECT_ROOT"
    output=$(eval "$check_command" 2>&1) || exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}✅ Passed${NC}"
        [[ -n "$output" ]] && echo "$output"
        ((CHECKS_PASSED++))
        echo ""
        return 0
    else
        if [[ "$check_level" == "warn" ]]; then
            echo -e "${YELLOW}${BOLD}⚠️  Warning${NC}"
            [[ -n "$output" ]] && echo "$output"
            ((CHECKS_WARNED++))
            echo ""
            return 0
        else
            echo -e "${RED}${BOLD}❌ Failed${NC}"
            [[ -n "$output" ]] && echo "$output"
            ((CHECKS_FAILED++))
            EXIT_CODE=1
            echo ""
            return 1
        fi
    fi
}

# Helper to get level config key
get_level_config_key() {
    case "$1" in
        0) echo "feature" ;;
        1) echo "development" ;;
        2) echo "release" ;;
        3) echo "production" ;;
        *) echo "feature" ;;
    esac
}

# Main execution
main() {
    # Standard checks
    run_check "Format Check" "CHECK_FORMAT" "$(get_command "format")" || true
    run_check "Lint Check" "CHECK_LINT" "$(get_command "lint")" || true
    run_check "Type Check" "CHECK_TYPECHECK" "$(get_command "typecheck")" || true

    # Pedantic lint (if not skip)
    if [[ "${CHECK_PEDANTIC_LINT:-skip}" != "skip" ]]; then
        run_check "Pedantic Lint" "CHECK_PEDANTIC_LINT" "$(get_command "lint_pedantic")" || true
    fi

    # Tests (skip GPU and slow tests based on config)
    if [[ "${CHECK_GPU_TESTS:-skip}" == "skip" ]] && [[ "${CHECK_SLOW_TESTS:-skip}" == "skip" ]]; then
        run_check "Unit Tests" "CHECK_TESTS" "$(get_command "test")" || true
    else
        run_check "All Tests" "CHECK_TESTS" "$(get_command "test_all")" || true
    fi

    # Coverage
    run_coverage_check || true

    # Import validation
    run_check "Import Check" "CHECK_IMPORTS" "$(get_command "import_check")" || true

    # Custom checks (Tritter-specific)
    run_custom_checks || true

    # Print summary
    echo ""
    echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}  Summary${NC}"
    echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "Checks Run: ${BOLD}${CHECKS_RUN}${NC}"
    echo -e "Passed: ${GREEN}${BOLD}${CHECKS_PASSED}${NC}"
    echo -e "Warnings: ${YELLOW}${BOLD}${CHECKS_WARNED}${NC}"
    echo -e "Failed: ${RED}${BOLD}${CHECKS_FAILED}${NC}"
    echo ""

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}✅ All checks passed for ${STRICTNESS_NAME} level${NC}"
    else
        echo -e "${RED}${BOLD}❌ Some checks failed for ${STRICTNESS_NAME} level${NC}"
    fi
    echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════${NC}"

    exit $EXIT_CODE
}

main "$@"
