#!/usr/bin/env python3
"""
Entry point for running the preprocessing annotations pipeline.

This file allows the pipeline to be run directly from the project root:
    python main.py --input ./data --output ./results
"""

import sys
from pathlib import Path

# Add current directory to path so absolute imports work
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run the pipeline
if __name__ == "__main__":
    # Import here to ensure path is set
    from pipeline import main
    main()
