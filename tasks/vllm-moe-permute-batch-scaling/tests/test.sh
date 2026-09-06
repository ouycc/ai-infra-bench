#!/usr/bin/env bash
set -euo pipefail
mkdir -p /logs/verifier

fail_provenance() {
  local stage="$1"
  printf 'provenance_failed stage=%s\n' "${stage}" \
    > /logs/verifier/failure-stage.txt
  printf '0\n' > /logs/verifier/reward.txt
  exit 0
}

repo=/app
base_commit="$(git -c safe.directory="${repo}" -C "${repo}" rev-parse HEAD)" \
  || fail_provenance base_commit
candidate_patch_sha256="$(
  git -c safe.directory="${repo}" -C "${repo}" diff --binary HEAD \
    | sha256sum | awk '{print $1}'
)" || fail_provenance candidate_patch
candidate_status="$(git -c safe.directory="${repo}" -C "${repo}" status --short)" \
  || fail_provenance candidate_status
{
  printf 'base_commit=%s\n' "${base_commit}"
  printf 'candidate_patch_sha256=%s\n' "${candidate_patch_sha256}"
  printf '%s\n' "${candidate_status}"
} > /logs/verifier/candidate-provenance.txt
set +e
sh /tests/rebuild_for_verification.sh \
  > /logs/verifier/native-build.stdout.log \
  2> /logs/verifier/native-build.stderr.log
build_status=$?
set -e
if [[ ${build_status} -ne 0 ]]; then
  printf 'build_failed exit=%s\n' "${build_status}" \
    > /logs/verifier/failure-stage.txt
  printf '0\n' > /logs/verifier/reward.txt
  exit 0
fi
printf 'build_passed\n' > /logs/verifier/build-stage.txt
correctness_rc=0
performance_rc=0
cd /app
python3 -I /tests/verify_moe_permute.py --stage correctness \
  > /logs/verifier/correctness.log 2>&1 || correctness_rc=$?
if [[ ${correctness_rc} -eq 0 ]]; then
  python3 -I /tests/verify_moe_permute.py --stage performance \
    > /logs/verifier/performance.log 2>&1 || performance_rc=$?
else
  performance_rc=125
  printf 'skipped: correctness failed\n' > /logs/verifier/performance.log
fi
cat /logs/verifier/correctness.log
cat /logs/verifier/performance.log
printf '{"build":0,"correctness":%d,"performance":%d}\n' \
  "${correctness_rc}" "${performance_rc}" > /logs/verifier/stages.json
if [[ ${correctness_rc} -eq 0 && ${performance_rc} -eq 0 ]]; then
  printf '1\n' > /logs/verifier/reward.txt
else
  printf 'correctness=%s performance=%s\n' \
    "${correctness_rc}" "${performance_rc}" \
    > /logs/verifier/failure-stage.txt
  printf '0\n' > /logs/verifier/reward.txt
fi
