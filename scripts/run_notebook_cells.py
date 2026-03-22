import argparse
import json
import os
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute notebook code cells sequentially with plain Python.")
    parser.add_argument("notebook", type=Path)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Directory to switch into before execution. Defaults to notebook parent.",
    )
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    workdir = (args.workdir or notebook_path.parent).resolve()

    # Avoid GUI backends during headless execution.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl-config")
    os.chdir(workdir)

    nb = json.loads(notebook_path.read_text())
    namespace = {"__name__": "__main__"}

    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        print(f"===== CELL {i} START =====", flush=True)
        try:
            exec(compile(source, f"{notebook_path.name}:cell_{i}", "exec"), namespace)
        except Exception as exc:
            print(f"FAILED CELL {i}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            return 1
        print(f"===== CELL {i} OK =====", flush=True)

    print("ALL CODE CELLS EXECUTED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
