#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
archive="$script_dir/catalystflow-linux-x86_64.tar.gz"
target="${1:-$script_dir/.portable_env}"

if [[ ! -f "$archive" ]]; then
  echo "Missing portable environment archive: $archive" >&2
  exit 1
fi
if [[ -x "$target/bin/python" ]]; then
  if "$target/bin/python" -c \
    "import numpy, scipy, pandas, sklearn, rdkit; print('Portable CatalystFlow environment ready')"; then
    echo "Environment path: $target"
    exit 0
  fi
  echo "Existing portable environment failed the import check: $target" >&2
  echo "Move or delete this generated cache, then rerun the launcher to extract a clean copy." >&2
  exit 1
fi
if [[ -e "$target" ]] && [[ -n "$(find "$target" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Target exists and is not an initialized portable environment: $target" >&2
  exit 1
fi

mkdir -p "$target"
tar -xzf "$archive" -C "$target"
"$target/bin/conda-unpack"
"$target/bin/python" -c \
  "import numpy, scipy, pandas, sklearn, rdkit; print('Portable CatalystFlow environment ready')"
echo "Environment path: $target"
