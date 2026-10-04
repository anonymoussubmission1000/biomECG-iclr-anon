"""Load the aggregate results used in the paper."""

import pandas as pd
import os
from pathlib import Path
from biomecg.paths import ROOT


def snapshot(name):
    path = (
        Path(os.environ.get("BIOMECG_ARTIFACTS", str(ROOT / "artifacts/paper")))
        / f"{name}.csv"
    )
    frame = pd.read_csv(path, dtype={"cohort_label": str, "cohort": str})
    if frame.empty:
        raise ValueError(f"Empty required paper artifact: {name}")
    return frame
