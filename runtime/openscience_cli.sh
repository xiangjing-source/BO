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
exec "$engine" "$@"
