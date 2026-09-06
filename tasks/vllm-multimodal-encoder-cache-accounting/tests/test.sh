#!/usr/bin/env bash
set -uo pipefail
mkdir -p /logs/verifier
if cd /app && python3 -I /tests/verify_encoder_cache.py; then
  printf '1\n' > /logs/verifier/reward.txt
else
  printf '0\n' > /logs/verifier/reward.txt
fi

