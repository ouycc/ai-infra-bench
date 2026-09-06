#!/usr/bin/env bash
set -uo pipefail
mkdir -p /logs/verifier
cd /workspace/repo || exit 1

# Preflight: verify local model config fixture
echo "=== Preflight: verify opt-125m config fixture ===" > /logs/verifier/preflight.log
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 - <<'PY' >> /logs/verifier/preflight.log 2>&1
from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    "/tests/fixtures/opt-125m",
    local_files_only=True,
)
assert config.model_type == "opt"
assert config.hidden_size == 768
assert config.num_hidden_layers == 12
print("LOCAL_OPT_CONFIG=PASS")
PY

if [ $? -ne 0 ]; then
  echo "ERROR: opt-125m fixture preflight failed" >&2
  cat /logs/verifier/preflight.log >&2
  printf '0\n' > /logs/verifier/reward.txt
  exit 1
fi
cat /logs/verifier/preflight.log

config_rc=0
scheduler_rc=0
basic_rc=0
reordered_rc=0
integrated_rc=0

python3 /tests/verify_async_pp.py --check-config \
  > /logs/verifier/config.log 2>&1 || config_rc=$?

timeout --signal=TERM --kill-after=30s 300s \
  python3 -m torch.distributed.run --nnodes=1 --nproc-per-node=2 \
    --master-addr=127.0.0.1 --master-port=29618 \
    /tests/verify_async_pp.py --scenario basic \
  > /logs/verifier/nccl-basic.log 2>&1 || basic_rc=$?

timeout --signal=TERM --kill-after=30s 300s \
  python3 -m torch.distributed.run --nnodes=1 --nproc-per-node=2 \
    --master-addr=127.0.0.1 --master-port=29619 \
    /tests/verify_async_pp.py --scenario reordered \
  > /logs/verifier/nccl-reordered.log 2>&1 || reordered_rc=$?

timeout --signal=TERM --kill-after=30s 300s \
  python3 -m torch.distributed.run --nnodes=1 --nproc-per-node=2 \
    --master-addr=127.0.0.1 --master-port=29620 \
    /tests/verify_async_pp.py --scenario integrated \
  > /logs/verifier/nccl-integrated.log 2>&1 || integrated_rc=$?

python3 /tests/verify_async_pp.py --check-scheduler \
  > /logs/verifier/scheduler.log 2>&1 || scheduler_rc=$?

cat /logs/verifier/config.log
cat /logs/verifier/nccl-basic.log
cat /logs/verifier/nccl-reordered.log
cat /logs/verifier/nccl-integrated.log
cat /logs/verifier/scheduler.log
printf '{"config":%d,"scheduler":%d,"nccl_basic":%d,"nccl_reordered":%d,"nccl_integrated":%d}\n' \
  "$config_rc" "$scheduler_rc" "$basic_rc" "$reordered_rc" \
  "$integrated_rc" \
  > /logs/verifier/stages.json

if (( config_rc == 0 && scheduler_rc == 0 \
      && basic_rc == 0 && reordered_rc == 0 && integrated_rc == 0 )); then
  printf '1\n' > /logs/verifier/reward.txt
else
  printf '0\n' > /logs/verifier/reward.txt
fi
