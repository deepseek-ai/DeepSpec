# The repo has no packaging (no pyproject.toml/setup.py), so put the repo
# root on sys.path to keep ``deepspec`` importable under a bare ``pytest``
# from any cwd. Remove once packaging or a root conftest lands.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
