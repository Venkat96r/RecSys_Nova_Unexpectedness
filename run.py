"""
Convenience launcher — runs main.py with the correct Python environment.
Just execute:  py -3.12 run.py  (or python run.py if 3.12 is default)
Pass any main.py args through, e.g.:  py -3.12 run.py --epochs 20 --device cpu
"""
import sys
import os

# Make sure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import main
from config import Config

cfg = Config()

# Simple CLI override
args = sys.argv[1:]
i = 0
while i < len(args):
    key = args[i].lstrip('-')
    val = args[i + 1] if i + 1 < len(args) else None
    if val and not val.startswith('--'):
        try:
            v = int(val)
        except ValueError:
            try:
                v = float(val)
            except ValueError:
                v = val
        setattr(cfg, key, v)
        i += 2
    else:
        i += 1

main(cfg)
