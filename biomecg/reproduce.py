"""Execute the paper notebook, requiring every cell to succeed."""

import argparse
import os
import sys
from jupyter_client import KernelManager
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from biomecg.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts/paper")
    args = parser.parse_args()
    os.environ["BIOMECG_ARTIFACTS"] = str(args.artifacts.resolve())
    os.environ["MPLBACKEND"] = "Agg"
    out = ROOT / "outputs/paper"
    out.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(
        ROOT / "notebooks/paper_figs_and_charts.ipynb", as_version=4
    )
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    try:
        NotebookClient(
            notebook,
            km=manager,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.write(notebook, out / "executed.ipynb")
    print("Figures and tables written to outputs/paper")


if __name__ == "__main__":
    main()
