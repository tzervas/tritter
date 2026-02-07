#!/usr/bin/env bash
# Claude Code Pre-Commit Hook for Tritter
# Runs before git commit to enforce quality gates
# Integrates with progressive CI system

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo -e "${YELLOW}Running pre-commit quality checks...${NC}"

# Detect branch strictness
if [[ -f ".github/scripts/detect-branch-strictness.sh" ]]; then
    eval "$(bash .github/scripts/detect-branch-strictness.sh)"
else
    echo -e "${RED}Error: Strictness detection script not found${NC}"
    exit 1
fi

echo -e "Branch: ${GREEN}${BRANCH}${NC}"
echo -e "Strictness: ${YELLOW}${STRICTNESS_NAME} (Level ${STRICTNESS_LEVEL})${NC}"
echo ""

# Quick format check
echo -e "Checking code format..."
if ruff format --check . &> /dev/null; then
    echo -e "${GREEN}✅ Format OK${NC}"
else
    echo -e "${YELLOW}⚠️  Code needs formatting${NC}"
    if [[ "${CHECK_FORMAT:-error}" == "error" ]]; then
        echo -e "${RED}❌ Format check failed (required for ${STRICTNESS_NAME} level)${NC}"
        echo ""
        echo -e "Fix with: ${YELLOW}ruff format .${NC}"
        exit 1
    else
        echo -e "   ${YELLOW}Continuing (warnings allowed for ${STRICTNESS_NAME} level)${NC}"
    fi
fi

# Quick lint check
echo -e "Checking for critical lint errors..."
if ruff check . --select E,F &> /dev/null; then
    echo -e "${GREEN}✅ Lint OK${NC}"
else
    echo -e "${YELLOW}⚠️  Lint issues found${NC}"
    if [[ "${CHECK_LINT:-warn}" == "error" ]]; then
        echo -e "${RED}❌ Lint check failed (required for ${STRICTNESS_NAME} level)${NC}"
        echo ""
        echo -e "Fix with: ${YELLOW}ruff check --fix .${NC}"
        exit 1
    else
        echo -e "   ${YELLOW}Continuing (warnings allowed for ${STRICTNESS_NAME} level)${NC}"
    fi
fi

# Check for hardcoded secrets (always enforced)
echo -e "Checking for hardcoded secrets..."
if python -m tritter.curation.secrets scan . --quiet 2>/dev/null; then
    echo -e "${GREEN}✅ No secrets detected${NC}"
else
    echo -e "${RED}❌ SECURITY: Hardcoded secrets detected${NC}"
    echo -e "${RED}   This check always fails regardless of strictness level${NC}"
    echo ""
    echo -e "Review files and remove secrets before committing."
    exit 1
fi

# Type check (for release and production only)
if [[ "$STRICTNESS_LEVEL" -ge 2 ]] && [[ "${CHECK_TYPECHECK:-warn}" == "error" ]]; then
    echo -e "Checking types (strict mode)..."
    if mypy src/tritter --no-error-summary &> /dev/null; then
        echo -e "${GREEN}✅ Type check OK${NC}"
    else
        echo -e "${RED}❌ Type check failed (required for ${STRICTNESS_NAME} level)${NC}"
        echo ""
        echo -e "Run: ${YELLOW}mypy src/tritter${NC} for details"
        exit 1
    fi
fi

# Success
echo ""
echo -e "${GREEN}✅ All pre-commit checks passed for ${STRICTNESS_NAME} level${NC}"
echo ""

exit 0
