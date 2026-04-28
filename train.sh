#!/bin/bash
# train.sh
# Entry point for training the model.

# Get the absolute path to the project root
PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
export PROJECT_ROOT=$PROJECT_ROOT

echo "=========================================================="
echo " Starting Banking Intent Training Pipeline"
echo " Project Root: $PROJECT_ROOT"
echo "=========================================================="

# Check if an environment argument is passed (local vs cloud)
ENV=${1:-"cloud"}

if [ "$ENV" == "local" ]; then
    echo "Running in LOCAL mode..."
    echo "[WARNING] Ensure your local machine has an NVIDIA GPU and CUDA installed."
    echo "[WARNING] Training via Unsloth on a pure CPU will fail or be extremely slow."
    
    # We use train_kaggle.py as the main robust script. 
    # If you have a specific local train.py, you can change this.
    python "$PROJECT_ROOT/scripts/train_kaggle.py"
else
    echo "Running in CLOUD (Vast.ai / Kaggle) mode..."
    python "$PROJECT_ROOT/scripts/train_kaggle.py"
fi
