"""Prepare all CODE-15 exams with the local MIMIC signal preprocessing.

Selection/follow-up adapted from Cavalab Code15_Process_042225.py (see README).
Run explicitly; importing this module performs no data processing.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import h5py
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class ExcludeECG(ValueError):
    """An individual signal cannot enter the output cohort."""


def binary(series, name):
    values = (
        series.astype(str)
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
    )
    if values.isna().any():
        raise ValueError(f"Invalid/missing binary values in {name}")
    return values.astype(np.int8)


def load_labels(path):
    labels = pd.read_csv(path)
    labels = labels.drop(
        columns=[
            column
            for column in labels
            if column.lower() in {"train", "test", "val", "validation", "split"}
            or column.lower().startswith("test_train_split")
        ]
    )
    required = {
        "exam_id",
        "patient_id",
        "age",
        "is_male",
        "death",
        "timey",
        "trace_file",
    }
    if missing := (required - set(labels)):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for column in ("exam_id", "patient_id"):
        values = pd.to_numeric(labels[column], errors="raise")
        if not np.isfinite(values).all() or (values % 1 != 0).any():
            raise ValueError(f"Invalid {column}")
        labels[column] = values.astype(np.int64)
    if labels.exam_id.duplicated().any():
        raise ValueError("Duplicate exam_id in exams.csv")
    timey = pd.to_numeric(labels.timey, errors="raise")
    has_survival = timey.notna()
    if not has_survival.equals(labels.death.notna()):
        raise ValueError("death and timey must be either both present or both missing")
    if labels.loc[has_survival, "patient_id"].duplicated().any():
        raise ValueError("Multiple survival-labelled exams for the same patient")
    if not np.isfinite(timey.loc[has_survival]).all():
        raise ValueError("Nonfinite follow-up")
    selected = labels.copy()
    age = pd.to_numeric(selected.age, errors="raise")
    if not np.isfinite(age).all() or (age < 0).any():
        raise ValueError("Invalid age")
    selected["Age"] = age
    selected["Is_Male"] = binary(selected.is_male, "is_male")
    selected["E_Mortality"] = pd.Series(pd.NA, index=selected.index, dtype="Int8")
    selected.loc[has_survival, "E_Mortality"] = binary(
        selected.loc[has_survival, "death"], "death"
    )
    selected["TTE_original_years"] = timey
    selected["TTE_Mortality"] = timey.clip(lower=1 / 365) * 365
    selected["TTE_clipped_to_one_day"] = (timey < 1 / 365).astype("boolean")
    selected.loc[~has_survival, "TTE_clipped_to_one_day"] = pd.NA
    selected["Has_Survival_Label"] = has_survival
    selected["Is_First_ECG"] = has_survival
    selected["ECG_Order"] = pd.Series(pd.NA, index=selected.index, dtype="Int64")
    selected.loc[has_survival, "ECG_Order"] = 0
    selected["ECG_Session_Index"] = selected.groupby(
        "patient_id", sort=False
    ).cumcount()
    selected = selected.rename(columns={"exam_id": "SID", "patient_id": "PID"})
    if selected.trace_file.isna().any():
        raise ValueError("Missing trace_file")
    for name in selected.trace_file.unique():
        if (
            Path(name).name != name
            or not name.startswith("exams_part")
            or (not name.endswith(".hdf5"))
        ):
            raise ValueError(f"Invalid source filename: {name}")
    return (labels, selected)


def preprocess_signal(raw):
    """Remove source zero padding, resample 400->500 Hz, clean like MIMIC.

    Zero padding is inferred only from contiguous all-lead-zero edges. No
    internal zero samples are removed and signal duration is never stretched.
    """
    if raw.shape != (4096, 12):
        raise ValueError(f"Unexpected CODE signal shape: {raw.shape}")
    if not np.isfinite(raw).all():
        raise ExcludeECG("nonfinite_raw_signal")
    active = np.flatnonzero(np.any(raw != 0, axis=1))
    if not len(active):
        raise ExcludeECG("all_zero_signal")
    start, stop = (int(active[0]), int(active[-1]) + 1)
    signal = raw[start:stop]
    output_length = (len(signal) * 5 + 3) // 4
    if output_length < 4096:
        raise ExcludeECG("shorter_than_4096_at_500hz")
    if len(signal) < 2:
        raise ExcludeECG("insufficient_signal_samples")
    signal = resample_poly(signal.astype(np.float64), up=5, down=4, axis=0)
    cleaned = np.empty_like(signal)
    for lead in range(12):
        padded = np.pad(signal[:, lead], (500, 500), mode="reflect")
        filtered = nk.ecg_clean(padded, sampling_rate=500, method="neurokit")
        cleaned[:, lead] = filtered[500:-500]
    crop_start = (len(cleaned) - 4096) // 2
    cleaned = cleaned[crop_start : crop_start + 4096]
    result = cleaned.T.astype(np.float32)
    if not np.isfinite(result).all():
        raise ExcludeECG("nonfinite_processed_signal")
    return (
        result,
        {
            "source_start_sample": start,
            "source_stop_sample": stop,
            "source_valid_samples": stop - start,
            "resampled_samples": len(signal),
        },
    )


def file_info(path):
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/code15/raw")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/code15/processed"
    )
    args = parser.parse_args()
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(
            f"Output already exists; choose a new directory: {output}"
        )
    labels, selected = load_labels(source / "exams.csv")
    locations = {}
    inputs = [file_info(source / "exams.csv")]
    for name, group in selected.groupby("trace_file", sort=True):
        path = source / name
        inputs.append(file_info(path))
        with h5py.File(path, "r") as handle:
            ids = handle["exam_id"][:]
            if handle["tracings"].shape != (len(ids), 4096, 12):
                raise ValueError(f"Unexpected tracings shape in {path}")
            valid = ids[ids != 0]
            if len(np.unique(valid)) != len(valid):
                raise ValueError(f"Duplicate nonzero exam_id in {path}")
            lookup = {int(sid): i for i, sid in enumerate(ids) if sid != 0}
            missing = set(group.SID) - lookup.keys()
            if missing:
                raise ValueError(f"{len(missing)} selected exams missing from {path}")
            locations[name] = lookup
    metadata = {
        "status": "incomplete",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "exams_csv_sha256": hashlib.sha256(
            (source / "exams.csv").read_bytes()
        ).hexdigest(),
        "input_files": inputs,
        "input_rows": len(labels),
        "input_patients": int(labels.patient_id.nunique()),
        "exams_to_process": len(selected),
        "first_exams_with_followup": int(selected.Has_Survival_Label.sum()),
        "patients_without_followup": int(
            labels.patient_id.nunique()
            - selected.loc[selected.Has_Survival_Label, "PID"].nunique()
        ),
        "source_hz": 400,
        "output_hz": 500,
        "output_samples": 4096,
        "lead_order": LEADS,
        "short_policy": "exclude",
        "resampling": "scipy.signal.resample_poly(up=5, down=4), default parameters",
        "filter": "500-sample reflection, nk.ecg_clean(500 Hz, neurokit), remove reflection, centre crop",
        "normalization": "none; downstream per-ECG/per-lead z-score remains separate",
        "tte": "max(timey, 1/365) * 365 days, no integer rounding",
        "split": "none; downstream patient splits defined later",
        "selection": "all exams passing signal checks; survival targets only where provided",
        "ecg_order": "0 for survival-labelled first exam, missing otherwise",
        "session_index": "within-patient original CSV order; not chronological",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pandas", "h5py", "neurokit2")
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "preprocessing.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    kept, excluded = ([], [])
    hdf_path = output / "ecg.h5"
    try:
        with h5py.File(hdf_path, "w") as dest:
            ecgs = dest.create_dataset(
                "ECG",
                (0, 12, 4096),
                maxshape=(None, 12, 4096),
                dtype="float32",
                chunks=(1, 12, 4096),
                compression="gzip",
            )
            pids = dest.create_dataset("PID", (0,), maxshape=(None,), dtype="int64")
            sids = dest.create_dataset("SID", (0,), maxshape=(None,), dtype="int64")
            dest.attrs["sampling_rate"] = 500
            dest.attrs["lead_order"] = json.dumps(LEADS)
            dest.attrs["complete"] = False
            with tqdm(total=len(selected), desc="CODE-15 ECGs", unit="ECG") as progress:
                for name, group in selected.groupby("trace_file", sort=True):
                    with h5py.File(source / name, "r") as handle:
                        for row in group.to_dict("records"):
                            source_row = locations[name][row["SID"]]
                            raw = handle["tracings"][source_row]
                            try:
                                signal, details = preprocess_signal(raw)
                            except ExcludeECG as error:
                                excluded.append(
                                    {
                                        "PID": row["PID"],
                                        "SID": row["SID"],
                                        "trace_file": name,
                                        "Has_Survival_Label": row["Has_Survival_Label"],
                                        "reason": str(error),
                                    }
                                )
                            else:
                                index = len(kept)
                                ecgs.resize(index + 1, axis=0)
                                pids.resize(index + 1, axis=0)
                                sids.resize(index + 1, axis=0)
                                ecgs[index], pids[index], sids[index] = (
                                    signal,
                                    row["PID"],
                                    row["SID"],
                                )
                                row.update(
                                    details, source_row=source_row, hdf5_row=index
                                )
                                kept.append(row)
                            progress.update(1)
            if not kept:
                raise ValueError("No ECG retained")
            pd.DataFrame(kept).to_csv(output / "labels.csv", index=False)
            pd.DataFrame(
                excluded,
                columns=["PID", "SID", "trace_file", "Has_Survival_Label", "reason"],
            ).to_csv(output / "excluded_exams.csv", index=False)
            metadata.update(
                status="complete",
                retained=len(kept),
                excluded=len(excluded),
                retained_patients=len({row["PID"] for row in kept}),
                retained_survival_exams=sum(
                    (row["Has_Survival_Label"] for row in kept)
                ),
                retained_patients_with_multiple_ecgs=int(
                    (pd.Series([row["PID"] for row in kept]).value_counts() >= 2).sum()
                ),
                exclusion_counts=pd.Series([x["reason"] for x in excluded], dtype=str)
                .value_counts()
                .to_dict(),
            )
            dest.attrs["complete"] = True
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    except BaseException as error:
        metadata.update(
            status="failed", error=repr(error), retained_before_failure=len(kept)
        )
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
        raise
    print(f"Complete: {len(kept)} ECGs, {len(excluded)} excluded. Outputs: {output}")


if __name__ == "__main__":
    main()
