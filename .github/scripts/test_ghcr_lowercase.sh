#!/usr/bin/env bash
# Regression test for GHCR_REPOSITORY lowercase normalization

set -euo pipefail

test_case() {
  local input="$1"
  local expected="$2"

  GHCR_REPOSITORY="$input"
  GHCR_REPOSITORY="${GHCR_REPOSITORY,,}"

  if [[ "$GHCR_REPOSITORY" != "$expected" ]]; then
    printf 'FAIL: input=%s expected=%s actual=%s\n' "$input" "$expected" "$GHCR_REPOSITORY"
    exit 1
  fi
}

# Mixed case (the bug scenario)
test_case "ghcr.io/OuyCC/ai-infra-bench-task-envs" "ghcr.io/ouycc/ai-infra-bench-task-envs"

# Already lowercase (no-op)
test_case "ghcr.io/ouycc/ai-infra-bench-task-envs" "ghcr.io/ouycc/ai-infra-bench-task-envs"

# All uppercase
test_case "GHCR.IO/OUYCC/AI-INFRA-BENCH-TASK-ENVS" "ghcr.io/ouycc/ai-infra-bench-task-envs"

printf 'All GHCR lowercase normalization tests passed\n'
