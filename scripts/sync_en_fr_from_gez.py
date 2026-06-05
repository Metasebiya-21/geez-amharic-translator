#!/usr/bin/env python3
"""Clone gez_to_amh_blstm.ipynb → en_fr_baseline.ipynb with EN–FR defaults."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "gez_to_amh_blstm.ipynb"
DST = ROOT / "en_fr_baseline.ipynb"


def main():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    # Apply patches via same logic as initial sync — run this file after editing gez notebook
    text = json.dumps(nb)
    assert len(nb["cells"]) > 40
    DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Cloned structure only — run full patch pass from repo maintainer script.")
    print("Prefer: keep en_fr_baseline.ipynb in git; edit Config/Data cells manually after gez changes.")


if __name__ == "__main__":
    main()
