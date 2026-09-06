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

repo=/workspace/repo
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

# Verify frozen reference is present and unmodified
frozen_ref=/opt/ai-infra-bench/reference-int8/reference_int8_kernel.py
if [[ ! -f "${frozen_ref}" ]]; then
  printf 'frozen_reference_missing\n' > /logs/verifier/failure-stage.txt
  printf '0\n' > /logs/verifier/reward.txt
  exit 0
fi

expected_sha256=36406a44b95e54cf99988105d0fe9a69645a0d2fcbfe2e60b1982d3ac9fdcff3
actual_sha256="$(sha256sum "${frozen_ref}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
  printf 'frozen_reference_tampered expected=%s actual=%s\n' \
    "${expected_sha256}" "${actual_sha256}" \
    > /logs/verifier/failure-stage.txt
  printf '0\n' > /logs/verifier/reward.txt
  exit 0
fi

# Rebuild native extension
set +e
bash /tests/rebuild_for_verification.sh \
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

# Run trusted parent/worker verifier (writes reward.txt itself)
cd /workspace/repo
exec python3 -I /tests/verify_int8_quant.py > /logs/verifier/verification.log 2>&1
