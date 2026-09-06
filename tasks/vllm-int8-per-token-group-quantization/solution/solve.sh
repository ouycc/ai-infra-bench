#!/usr/bin/env bash
set -euo pipefail
cd /workspace/repo
git apply /solution/oracle.patch

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
marker = "package_data = {"
assert text.count(marker) == 1
path.write_text(text.replace(marker, 'ext_modules = [CMakeExtension(name="vllm._C")]\n\n' + marker))
PY
python3 -m pip install --no-build-isolation --no-deps -e /workspace/repo
restore_setup
trap - EXIT HUP INT TERM
