#!/usr/bin/env bash
# Build, check and publish. Exit code is the result; nothing prints success for a skipped step.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGET="${1:-staging}"
case "$TARGET" in
  staging|production) ;;
  *) echo "unknown target: $TARGET" >&2; exit 2 ;;
esac

if [ -n "$(git status --porcelain)" ]; then
  echo "working tree is dirty, refusing to publish" >&2
  exit 3
fi

REVISION="$(git rev-parse --short HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="dist/${TARGET}-${REVISION}-${STAMP}.tar.gz"

mkdir -p dist
tar --exclude='.git' --exclude='node_modules' -czf "$ARCHIVE" src public package.json
SIZE="$(stat -c%s "$ARCHIVE")"
if [ "$SIZE" -gt 20971520 ]; then
  echo "archive is ${SIZE} bytes, over the 20 MiB ceiling" >&2
  exit 4
fi
echo "built $ARCHIVE (${SIZE} bytes) for $TARGET at revision $REVISION"
