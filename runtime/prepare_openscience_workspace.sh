#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_root="$(cd "$script_dir/.." && pwd)"

# Open Science identifies a project by its nearest Git worktree.  If the
# submission is unpacked inside another repository, sessions would otherwise
# inherit the parent project and miss this package's openscience.json.  A local
# repository boundary keeps the Web and CLI front ends on the submission root.
if [[ ! -e "$package_root/.git" ]]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "Git is required to isolate the Open Science project workspace." >&2
    exit 1
  fi
  git -C "$package_root" init --quiet
  git -C "$package_root" config gc.auto 0
  git -C "$package_root" config core.excludesFile \
    "$script_dir/openscience_project_excludes"
  git -C "$package_root" config user.name CatalystFlow
  git -C "$package_root" config user.email catalystflow@local.invalid
  git -C "$package_root" commit --quiet --allow-empty --no-gpg-sign \
    -m "Initialize Open Science project boundary"
  echo "Open Science project boundary initialized: $package_root" >&2
fi
