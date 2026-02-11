#!/bin/bash
# Preprocessing Annotations Pipeline Runner
#
# This script runs the annotation pipeline from the project root.
# All relative paths are resolved from the project directory.

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to project directory
cd "$SCRIPT_DIR"

# Load environment variables if .env exists
if [ -f .env ]; then
    echo "Loading environment from .env..."
    set -a
    source .env
    set +a
fi

# Run the pipeline
python main.py "$@"
