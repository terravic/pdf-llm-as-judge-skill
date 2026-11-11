"""Runner script for Multi-Agent PDF Extraction and Consensus Evaluator."""

import sys
from pathlib import Path

# Ensure project root directory is in Python module search path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from pdf_consensus_evaluator.cli import main

if __name__ == "__main__":
    sys.exit(main())
