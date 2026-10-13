import os
import sys

# Make the repository root importable (model/, utils.py, scripts/).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
