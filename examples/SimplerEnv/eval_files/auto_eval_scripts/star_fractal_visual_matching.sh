#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FRACTAL_SETTING=visual_matching

exec "${SCRIPT_DIR}/star_fractal.sh" "$@"
