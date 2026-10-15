#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_root="$(cd "$script_dir/.." && pwd)"
bash "$script_dir/prepare_openscience_workspace.sh"
engine="$(bash "$script_dir/bootstrap_openscience.sh")"

cd "$package_root"
cat <<'EOF'
Configure the competition DeepSeek credential.

Open Science's generic DeepSeek credential screen may test the key against the
official DeepSeek service and display "Authentication Fails". Competition keys
are valid only for the custom endpoint declared in this project's
openscience.json. The endpoint-specific AUTH_OK test below is authoritative for
this submission.
EOF

set +e
"$engine" keys add deepseek
generic_status=$?
set -e

if [[ "$generic_status" -ne 0 ]]; then
  echo "Generic provider validation did not pass; testing the configured competition endpoint..." >&2
fi

bash "$script_dir/verify_deepseek.sh"
