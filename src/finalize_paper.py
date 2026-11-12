"""Finalize paper after full experiment completion."""
from __future__ import annotations

if __name__ == "__main__":
    import sys as _legacy_sys
    from pathlib import Path as _LegacyPath
    _legacy_sys.path.insert(0, str(_LegacyPath(__file__).resolve().parent))
    from legacy_guard import require_legacy_opt_in as _require_legacy_opt_in
    _require_legacy_opt_in()

from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(".")
PAPER = ROOT / "paper" / "MRP_Draft.qmd"
RESULTS = ROOT / "results" / "full"
FIGURES = ROOT / "figures"

def generate_figures() -> None:
    subprocess.run([sys.executable, str(ROOT / "src" / "generate_figures.py")],
                   cwd=ROOT, check=True)

def update_paper_results() -> None:
    # This is a placeholder for a more robust update.
    # For now, we rely on the paper's embedded Python chunks to read results directly.
    pass

def render_paper() -> None:
    subprocess.run(["quarto", "render", str(PAPER), "--to", "pdf"],
                   cwd=ROOT, check=True)

def main() -> None:
    if not (RESULTS / "run_metadata.json").exists():
        print("Results not ready.")
        return
    meta = json.loads((RESULTS / "run_metadata.json").read_text())
    print(f"Finalizing paper for {meta['out_of_sample_origins']} origins...")
    generate_figures()
    update_paper_results()
    render_paper()
    print("Paper finalized.")

if __name__ == "__main__":
    main()
