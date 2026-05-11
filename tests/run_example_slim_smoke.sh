#!/bin/sh

set -eu

slim_bin=$1
examples_dir=$2

found_example=0
for slim_path in "$examples_dir"/*.slim; do
    [ -e "$slim_path" ] || continue
    found_example=1
    "$slim_bin" "$slim_path" >/dev/null
done

if [ "$found_example" -eq 0 ]; then
    echo "no .slim examples found under $examples_dir" >&2
    exit 1
fi
