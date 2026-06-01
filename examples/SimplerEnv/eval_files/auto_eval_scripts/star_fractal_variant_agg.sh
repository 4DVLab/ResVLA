#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FRACTAL_SETTING=variant_agg

exec "${SCRIPT_DIR}/star_fractal.sh" "$@"
