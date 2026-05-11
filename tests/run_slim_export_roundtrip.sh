#!/bin/sh
set -eu

SPARQY_BIN="$1"
SLIM_BIN="$2"
CONFIG_PATH="$3"

TMPDIR="$(mktemp -d)"
cleanup() {
    rm -rf "$TMPDIR"
}
trap cleanup EXIT INT TERM

PREFIX="$TMPDIR/sparqy_export_roundtrip"

"$SPARQY_BIN" --config "$CONFIG_PATH" --export-slim "$PREFIX" >/dev/null

test -f "$PREFIX.txt"
test -f "$PREFIX.slim"

"$SLIM_BIN" "$PREFIX.slim" >"$TMPDIR/slim.out" 2>"$TMPDIR/slim.err"

grep -q '^sparqy_import_ok' "$TMPDIR/slim.out"
