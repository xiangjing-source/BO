#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_root="$(cd "$script_dir/.." && pwd)"
engine="$(bash "$script_dir/bootstrap_openscience.sh")"
audit_root="$(mktemp -d "${TMPDIR:-/tmp}/catalystflow-openscience-audit.XXXXXX")"
mkdir -p "$audit_root/home" "$audit_root/cache" "$audit_root/config" "$audit_root/data"
cleanup() {
  rm -rf -- "$audit_root"
}
trap cleanup EXIT

for required in \
  "$package_root/openscience.json" \
  "$package_root/.openscience/agent/catalystflow.md" \
  "$package_root/skill/catalystflow/SKILL.md" \
  "$package_root/agent/mcp_server.py"; do
  [[ -f "$required" ]] || {
    echo "Missing CatalystFlow integration file: $required" >&2
    exit 4
  }
done

cd "$package_root"
run_isolated() {
  HOME="$audit_root/home" \
  XDG_CACHE_HOME="$audit_root/cache" \
  XDG_CONFIG_HOME="$audit_root/config" \
  OPENSCIENCE_DATA_DIR="$audit_root/data" \
  OPENSCIENCE_CONFIG_DIR="$audit_root/config" \
  OPENSCIENCE_DISABLE_AUTOUPDATE=1 \
  OPENSCIENCE_DISABLE_MODELS_FETCH=1 \
  OPENSCIENCE_DISABLE_LSP_DOWNLOAD=1 \
    "$engine" "$@"
}

run_isolated agent list >"$audit_root/agents.txt"

grep -F "catalystflow (primary)" "$audit_root/agents.txt" >/dev/null || {
  echo "Bundled Open Science did not discover the CatalystFlow primary agent." >&2
  exit 4
}

run_isolated debug skill >"$audit_root/skills.json"
grep -F "\"name\": \"catalystflow\"" "$audit_root/skills.json" >/dev/null || {
  echo "Bundled Open Science did not load the CatalystFlow project Skill." >&2
  exit 4
}
grep -F "$package_root/skill/catalystflow/SKILL.md" "$audit_root/skills.json" >/dev/null || {
  echo "CatalystFlow Skill was not loaded from the submission directory." >&2
  exit 4
}

run_isolated debug agent catalystflow >"$audit_root/agent.json"
grep -F '"providerID": "deepseek"' "$audit_root/agent.json" >/dev/null || {
  echo "CatalystFlow agent does not resolve to the DeepSeek provider." >&2
  exit 4
}
grep -F '"modelID": "deepseek-v4-flash"' "$audit_root/agent.json" >/dev/null || {
  echo "CatalystFlow agent does not resolve to deepseek-v4-flash." >&2
  exit 4
}

echo "Open Science runtime verification: PASS"
echo "Version: $($engine --version)"
echo "CatalystFlow primary agent: discovered"
echo "CatalystFlow project Skill and DeepSeek model binding: verified"
echo "Isolation: clean temporary HOME, configuration, and cache"
echo "DeepSeek API call: not executed (credential and network are intentionally external)"
