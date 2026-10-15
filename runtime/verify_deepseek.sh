#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_root="$(cd "$script_dir/.." && pwd)"
bash "$script_dir/prepare_openscience_workspace.sh"
engine="$(bash "$script_dir/bootstrap_openscience.sh")"

cd "$package_root"
export OPENSCIENCE_DISABLE_AUTOUPDATE=1
export OPENSCIENCE_DISABLE_MODELS_FETCH=1
export OPENSCIENCE_DISABLE_LSP_DOWNLOAD=1

set +e
output="$($engine run \
  --agent catalystflow \
  --model deepseek/deepseek-v4-flash \
  --format json \
  --title catalystflow-auth-check \
  '只调用 catalystflow_agent_status，然后只回复 AUTH_OK，不执行其他操作。' 2>&1)"
status=$?
set -e

if [[ "$status" -eq 0 ]] && grep -q '"text":"AUTH_OK"' <<<"$output"; then
  echo "Competition DeepSeek endpoint authentication: PASS"
  echo "CatalystFlow Agent and MCP status call: PASS"
  exit 0
fi

echo "Competition DeepSeek endpoint authentication: FAIL" >&2
echo "Check the saved key, network, quota, and the base URL in openscience.json." >&2
echo "$output" >&2
exit 1
