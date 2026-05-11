#!/bin/sh

set -eu

sparqy_bin=$1
examples_dir=$2

found_example=0
for config_path in "$examples_dir"/*.sparqy; do
    [ -e "$config_path" ] || continue
    found_example=1
    "$sparqy_bin" --config "$config_path" >/dev/null
done

if [ "$found_example" -eq 0 ]; then
    echo "no .sparqy examples found under $examples_dir" >&2
    exit 1
fi

"$sparqy_bin" 10 100 1e-7 0.01 -0.01 5 2 0.5 42 1 --stats=mean_fitness >/dev/null

if "$sparqy_bin" 10 100 1e-7 0.01 -0.01 5 abc 0.5 42 1 --stats=mean_fitness >/dev/null 2>&1; then
    echo "legacy CLI accepted a non-numeric out_interval" >&2
    exit 1
fi

if "$sparqy_bin" 10 100 1e-7 0.01 -0.01 5 0 0.5 42 1 --stats=mean_fitness >/dev/null 2>&1; then
    echo "legacy CLI accepted out_interval=0" >&2
    exit 1
fi
