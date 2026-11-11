"""Run all checks, or only one phase's checks for an isolated environment."""
import argparse
import sys
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['1','2','all'],default='all')
    args=parser.parse_args()
    pattern='test_*.py' if args.phase=='all' else f'test_phase{args.phase}.py'
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
