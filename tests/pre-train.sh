#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
exec "${PYTHON:-python}" -m pytest -q tests "$@"
