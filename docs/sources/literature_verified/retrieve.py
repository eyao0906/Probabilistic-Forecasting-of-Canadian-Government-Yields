"""Safe compatibility entry point: explicit offline/live manifest replay.

Original implementation is preserved in docs/archive_pre_resumption/. Use
python -m src.acquisition --help for the canonical interface. No network is
used without --live; existing source bytes are never overwritten.
"""
from pathlib import Path
import sys,json


def main():
    root=Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(root))
    from src.acquisition import main as replay
    ids=[x['id'] for x in json.loads((root/'config/sources.json').read_text(encoding='utf-8'))['sources'] if x['id'].startswith('lit_')]
    return replay(['--only',*ids,*sys.argv[1:]])


if __name__=='__main__':raise SystemExit(main())

