#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
archive="$script_dir/openscience/openscience-linux-x64-2.0.23.tgz"
target="$script_dir/.openscience_runtime"
binary="$target/bin/openscience"
archive_sha256="d3b0e541e2a08579243694509e2141eb56fee1825f33b7c6ecaf62a823b3278e"
binary_sha256="733132330d9e0e7e48ca93bc370b7734e7c1920e5773e8514c690753cc65836e"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "Bundled Open Science runtime supports Linux x86_64 only." >&2
  exit 2
fi

if ! command -v sha256sum >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1; then
  echo "sha256sum and tar are required to unpack the bundled runtime." >&2
  exit 2
fi

verify_binary() {
  [[ -x "$binary" ]] || return 1
  [[ "$(sha256sum "$binary" | awk '{print $1}')" == "$binary_sha256" ]] || return 1
  [[ "$($binary --version)" == "2.0.23" ]] || return 1
}

if [[ -e "$target" ]]; then
  if verify_binary; then
    printf '%s\n' "$binary"
    exit 0
  fi
  echo "Existing runtime cache is incomplete or has a hash mismatch: $target" >&2
  echo "Move that cache aside and rerun; the submission archive was not modified." >&2
  exit 3
fi

[[ -f "$archive" ]] || {
  echo "Bundled Open Science archive is missing: $archive" >&2
  exit 2
}

actual_archive_sha256="$(sha256sum "$archive" | awk '{print $1}')"
if [[ "$actual_archive_sha256" != "$archive_sha256" ]]; then
  echo "Bundled Open Science archive failed SHA-256 verification." >&2
  exit 3
fi

temporary="$(mktemp -d "$script_dir/.openscience_runtime.tmp.XXXXXX")"
cleanup() {
  rm -rf -- "$temporary"
}
trap cleanup EXIT

tar -xzf "$archive" -C "$temporary"
[[ -f "$temporary/package/bin/openscience" ]] || {
  echo "Official archive does not contain package/bin/openscience." >&2
  exit 3
}
chmod +x "$temporary/package/bin/openscience"

candidate_sha256="$(sha256sum "$temporary/package/bin/openscience" | awk '{print $1}')"
if [[ "$candidate_sha256" != "$binary_sha256" ]]; then
  echo "Extracted Open Science binary failed SHA-256 verification." >&2
  exit 3
fi

if [[ "$("$temporary/package/bin/openscience" --version)" != "2.0.23" ]]; then
  echo "Extracted Open Science version is not 2.0.23." >&2
  exit 3
fi

mv "$temporary/package" "$target"
printf '%s\n' "$binary"
