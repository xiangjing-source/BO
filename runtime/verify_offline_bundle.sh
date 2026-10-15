#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
environment_dir="$script_dir/.portable_env"

bash "$script_dir/bootstrap_portable_env.sh" "$environment_dir"
"$environment_dir/bin/python" "$script_dir/smoke_test_environment.py"
bash "$script_dir/verify_openscience_runtime.sh"

echo "Offline numerical and Open Science runtime verification: PASS"
