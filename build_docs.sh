#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sphinx-build -M clean "$ROOT/docs" "$ROOT/docs/_build"
sphinx-build -M html "$ROOT/docs" "$ROOT/docs/_build"
