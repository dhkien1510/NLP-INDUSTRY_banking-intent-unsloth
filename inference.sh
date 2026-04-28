#!/bin/bash
# inference.sh
# Entry point for running standalone inference.

PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
export PROJECT_ROOT=$PROJECT_ROOT

echo "=========================================================="
echo " Starting Banking Intent Inference"
echo " Project Root: $PROJECT_ROOT"
echo "=========================================================="

MODE=$1
QUERY=$2

# Interactive mode (default)
if [ -z "$MODE" ]; then
    python "$PROJECT_ROOT/scripts/inference.py"
# Evaluate entire test set
elif [ "$MODE" == "test" ]; then
    python "$PROJECT_ROOT/scripts/inference.py" --eval_test
# Single query mode
elif [ "$MODE" == "query" ]; then
    python "$PROJECT_ROOT/scripts/inference.py" --query "$QUERY"
else
    echo "Usage:"
    echo "  bash inference.sh             (Interactive mode)"
    echo "  bash inference.sh test        (Evaluate test set)"
    echo "  bash inference.sh query \"msg\" (Classify a single message)"
fi
