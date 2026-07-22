"""Adapters around the authors' ORIGINAL watermark repos. By default the
clones are expected next to the pilot_study checkout
(<parent>/lm-watermarking and <parent>/SemStamp); override the locations with
the LM_WATERMARKING_DIR / SEMSTAMP_DIR environment variables.
Import wm.official.kgw / wm.official.semstamp; they call add_repo_paths()
themselves."""

import os
import sys
from pathlib import Path

REPOS_DIR = Path(__file__).resolve().parents[4]  # parent of the repo checkout
LM_WATERMARKING_DIR = Path(os.environ.get("LM_WATERMARKING_DIR", REPOS_DIR / "lm-watermarking"))
SEMSTAMP_DIR = Path(os.environ.get("SEMSTAMP_DIR", REPOS_DIR / "SemStamp"))

_CLONE_HINTS = {
    "lm-watermarking": ("LM_WATERMARKING_DIR", "https://github.com/jwkirchenbauer/lm-watermarking"),
    "SemStamp": ("SEMSTAMP_DIR", "https://github.com/bohanhou14/SemStamp"),
}


def add_repo_paths():
    for p in (LM_WATERMARKING_DIR, SEMSTAMP_DIR):
        if not p.is_dir():
            env_var, url = _CLONE_HINTS[p.name]
            raise FileNotFoundError(
                f"Official upstream repo not found at {p}. The --impl official "
                f"runs need a clone of {url}: either clone it there or point "
                f"the {env_var} environment variable at an existing clone."
            )
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
