#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_root="$(cd "$script_dir/.." && pwd)"
bootstrap_dir="$script_dir/.micromamba_linux_aarch64"
environment_dir="$script_dir/.platform_env_linux_aarch64"
data_root="$package_root/data"
results_root="$package_root/reproduced_results"
results_root_set=0

usage() {
  cat <<EOF
Usage:
  bash runtime/run_reproduction_linux_arm64.sh
  bash runtime/run_reproduction_linux_arm64.sh [RESULTS_ROOT]
  bash runtime/run_reproduction_linux_arm64.sh [--data-root PATH] [--results-root PATH]

The positional RESULTS_ROOT form is retained for backward compatibility.
For the submitted benchmark, use the zero-argument form above.
EOF
}

reject_placeholder_path() {
  local label="$1"
  local value="$2"
  case "$value" in
    /path|/path/*|*/path/to/*|*'<'*|*'>'*)
      echo "$label is an example placeholder, not an executable path: $value" >&2
      echo "Run the launcher with no arguments to use the packaged data and output directory." >&2
      exit 2
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      [[ $# -ge 2 ]] && [[ -n "$2" ]] || { echo "--data-root requires a path" >&2; exit 2; }
      data_root="$2"
      shift 2
      ;;
    --data-root=*)
      data_root="${1#*=}"
      [[ -n "$data_root" ]] || { echo "--data-root requires a path" >&2; exit 2; }
      shift
      ;;
    --results-root)
      [[ $# -ge 2 ]] && [[ -n "$2" ]] || { echo "--results-root requires a path" >&2; exit 2; }
      [[ "$results_root_set" -eq 0 ]] || { echo "Results root was specified more than once" >&2; exit 2; }
      results_root="$2"
      results_root_set=1
      shift 2
      ;;
    --results-root=*)
      [[ "$results_root_set" -eq 0 ]] || { echo "Results root was specified more than once" >&2; exit 2; }
      results_root="${1#*=}"
      [[ -n "$results_root" ]] || { echo "--results-root requires a path" >&2; exit 2; }
      results_root_set=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      [[ "$results_root_set" -eq 0 ]] || { echo "Results root was specified more than once" >&2; exit 2; }
      results_root="$1"
      results_root_set=1
      shift
      ;;
  esac
done

reject_placeholder_path "Data root" "$data_root"
reject_placeholder_path "Results root" "$results_root"
if [[ "$data_root" != /* ]]; then
  data_root="$package_root/$data_root"
fi
if [[ "$results_root" != /* ]]; then
  results_root="$package_root/$results_root"
fi

if [[ ! -d "$data_root" ]]; then
  echo "Data root is not a directory: $data_root" >&2
  exit 2
fi

if ! mkdir -p "$results_root"; then
  echo "Cannot create results root: $results_root" >&2
  exit 2
fi

echo "Competition data: $data_root"
echo "Reproduction output: $results_root"

if [[ "$(uname -s)" != "Linux" ]] || [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This launcher is for Linux aarch64. Linux x86_64 uses run_reproduction.sh." >&2
  exit 1
fi

micromamba="$bootstrap_dir/bin/micromamba"
if [[ ! -x "$micromamba" ]]; then
  mkdir -p "$bootstrap_dir"
  archive="$bootstrap_dir/micromamba.tar.bz2"
  curl -fL "https://micro.mamba.pm/api/micromamba/linux-aarch64/latest" \
    -o "$archive"
  tar -xjf "$archive" -C "$bootstrap_dir" bin/micromamba
  chmod +x "$micromamba"
fi

if [[ ! -x "$environment_dir/bin/python" ]]; then
  "$micromamba" create -y -p "$environment_dir" \
    -f "$package_root/code/environment-runtime.yml"
fi

"$micromamba" run -p "$environment_dir" python \
  "$script_dir/smoke_test_environment.py"
"$micromamba" run -p "$environment_dir" python \
  "$package_root/code/main/run_selected.py" \
  --data-root "$data_root" \
  --results-root "$results_root" \
  --dataset all --rounds 40 --seeds 20

echo "Linux aarch64 reproduction complete: $results_root"
