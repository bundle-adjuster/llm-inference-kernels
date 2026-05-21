import os
import sys

# Make `reference`, `benchmarks`, etc. importable when running pytest.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
