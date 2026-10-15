#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
runtime_root="$project_root/runtime/.portable_env"
runtime_python="$runtime_root/bin/python"

if [[ ! -x "$runtime_python" ]]; then
  bash "$project_root/runtime/bootstrap_portable_env.sh" "$runtime_root" >&2
fi

exec "$runtime_python" "$project_root/agent/mcp_server.py"
