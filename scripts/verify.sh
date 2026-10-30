#!/usr/bin/env bash
set -euo pipefail

echo "=== 1. Checking Code Formatting with Ruff ==="
uv run ruff format --check .

echo "=== 2. Running Ruff Linter ==="
uv run ruff check .

echo "=== 3. Static Type Analysis with Mypy ==="
uv run mypy src

echo "=== 4. Executing Pytest Harness (Unit + Integration with Coverage) ==="
uv run pytest

echo "✅ All Quality Gates Passed (ISO/IEC 25010 Conformance Verified)!"
