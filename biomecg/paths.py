"""Repository paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def data_paths(site):
    if site not in ("mimic", "code15"):
        raise ValueError(f"Unknown cohort: {site}")
    base = ROOT / "data" / site / "processed"
    return base / "labels.csv", base / "ecg.h5"
