#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_root="$(cd "$script_dir/.." && pwd)"
environment_dir="$script_dir/.portable_env"
data_root="$package_root/data"
results_root="$package_root/reproduced_results"
results_root_set=0

usage() {
  cat <<EOF
Usage:
  bash runtime/run_reproduction.sh
  bash runtime/run_reproduction.sh [RESULTS_ROOT]
  bash runtime/run_reproduction.sh [--data-root PATH] [--results-root PATH]

Defaults:
  --data-root     $package_root/data
  --results-root  $package_root/reproduced_results

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
      echo "Run 'bash runtime/run_reproduction.sh' with no arguments to use the packaged data and output directory." >&2
      exit 2
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      if [[ $# -lt 2 ]] || [[ -z "$2" ]]; then
        echo "--data-root requires a path" >&2
        exit 2
      fi
      data_root="$2"
      shift 2
      ;;
    --data-root=*)
      data_root="${1#*=}"
      if [[ -z "$data_root" ]]; then
        echo "--data-root requires a path" >&2
        exit 2
      fi
      shift
      ;;
    --results-root)
      if [[ $# -lt 2 ]] || [[ -z "$2" ]]; then
        echo "--results-root requires a path" >&2
        exit 2
      fi
      if [[ "$results_root_set" -eq 1 ]]; then
        echo "Results root was specified more than once" >&2
        exit 2
      fi
      results_root="$2"
      results_root_set=1
      shift 2
      ;;
    --results-root=*)
      if [[ "$results_root_set" -eq 1 ]]; then
        echo "Results root was specified more than once" >&2
        exit 2
      fi
      results_root="${1#*=}"
      if [[ -z "$results_root" ]]; then
        echo "--results-root requires a path" >&2
        exit 2
      fi
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
      if [[ "$results_root_set" -eq 1 ]]; then
        echo "Results root was specified more than once" >&2
        exit 2
      fi
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
  echo "Run the zero-argument command from a writable extracted submission directory." >&2
  exit 2
fi

echo "Competition data: $data_root"
echo "Reproduction output: $results_root"

if [[ ! -x "$environment_dir/bin/python" ]]; then
  bash "$script_dir/bootstrap_portable_env.sh" "$environment_dir"
fi

"$environment_dir/bin/python" "$package_root/code/main/run_selected.py" \
  --data-root "$data_root" \
  --results-root "$results_root" \
  --dataset all --rounds 40 --seeds 20

echo "Reproduction complete: $results_root"
