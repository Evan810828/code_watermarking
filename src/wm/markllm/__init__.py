"""MarkLLM-package reproduction (vLLM inference). Self-contained subpackage:
the runners here are executed from this directory (they cannot live in src/
directly, because MarkLLM ships its own top-level `utils` package which would
clash with src/utils.py on sys.path — see adapter.load_pilot_utils)."""
