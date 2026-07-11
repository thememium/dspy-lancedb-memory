#!/bin/bash
set -euo pipefail

# Run tests with coverage and capture output
OUTPUT=$(uv run pytest tests/ -v --cov=dspy_lancedb_memory --cov-report=term-missing 2>&1)

# Extract coverage percentage from the TOTAL line
COVERAGE=$(echo "$OUTPUT" | grep "^TOTAL" | awk '{print $NF}' | tr -d '%')

# Count tests
TEST_COUNT=$(echo "$OUTPUT" | grep -c "PASSED" || true)

# Extract test time
TEST_TIME=$(echo "$OUTPUT" | grep "passed in" | sed 's/.*in \([0-9.]*\)s.*/\1/' || echo "0")

# Output structured metrics
echo "METRIC coverage_pct=$COVERAGE"
echo "METRIC test_count=$TEST_COUNT"
echo "METRIC test_time_s=$TEST_TIME"

# Show the coverage table for analysis
echo "---COVERAGE_TABLE---"
echo "$OUTPUT" | grep -A 100 "^Name" | head -20
