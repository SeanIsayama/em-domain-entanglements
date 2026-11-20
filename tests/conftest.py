import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURE_WORDS = ["QZXUSERQ", "QZXREPLYQ", "QZXREADQ"]  # synthetic text; must never appear in committed files
