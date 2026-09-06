#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/repo
cd "${repo}"

source_digest="$({
  find csrc vllm/model_executor/layers/quantization/utils \
    -type f \( -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' \
      -o -name '*.h' -o -name '*.py' \) -print0 \
    | sort -z \
    | xargs -0 sha256sum
  sha256sum setup.py CMakeLists.txt
} | sha256sum | awk '{print $1}')"

setup_backup="$(mktemp /tmp/vllm-setup.XXXXXX)"
cp -p setup.py "${setup_backup}"
restore_setup() {
  mv "${setup_backup}" setup.py
}
trap restore_setup EXIT HUP INT TERM

python3 - setup.py <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
injection = 'ext_modules = [CMakeExtension(name="vllm._C")]\n\n'
marker = "package_data = {"
assert text.count(marker) == 1
if injection not in text:
    path.write_text(text.replace(marker, injection + marker))
PY

printf 'candidate_source_sha256=%s\n' "${source_digest}"
printf 'build_command=python3 -m pip install --no-build-isolation --no-deps -e %s\n' "${repo}"
python3 -m pip install --no-build-isolation --no-deps -e "${repo}"

restore_setup
trap - EXIT HUP INT TERM

native="$(find "${repo}/vllm" -maxdepth 1 -type f -name '_C*.so' -print -quit)"
test -n "${native}"
native_digest="$(sha256sum "${native}" | awk '{print $1}')"
printf 'candidate_native_path=%s\n' "${native}"
printf 'candidate_native_sha256=%s\n' "${native_digest}"

cd /tmp
python3 -I - "${repo}" "${native}" "${native_digest}" <<'PY'
import hashlib
import importlib.util
import pathlib
import sys

import torch
import vllm

repo = pathlib.Path(sys.argv[1]).resolve()
expected_native = pathlib.Path(sys.argv[2]).resolve()
expected_digest = sys.argv[3]
source = pathlib.Path(vllm.__file__).resolve()
spec = importlib.util.find_spec("vllm._C")
assert source.is_relative_to(repo), source
assert spec and spec.origin
loaded = pathlib.Path(spec.origin).resolve()
assert loaded == expected_native, (loaded, expected_native)
actual_digest = hashlib.sha256(loaded.read_bytes()).hexdigest()
assert actual_digest == expected_digest, (actual_digest, expected_digest)
torch.ops.load_library(str(loaded))
print(f"cold_import_source={source}")
print(f"cold_import_native={loaded}")
print(f"cold_import_native_sha256={actual_digest}")
PY
