"""Compatibility entry point: corrected shared Phase 1 pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.phase1 import main

if __name__ == "__main__":
    main()
