"""Stream the documented MIMIC preprocessing into aligned HDF5 and CSV files."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import h5py
import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from biomecg.paths import ROOT

LEADS = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]


def metadata(source):
    """Use record-list timestamps, recorded death, and last discharge + 365 days."""
    patients = pd.read_csv(source / "patients.csv.gz")
    admissions = pd.read_csv(
        source / "admissions.csv.gz", usecols=["subject_id", "dischtime"]
    )
    records = pd.read_csv(source / "record_list.csv")
    measurements = pd.read_csv(source / "machine_measurements.csv", low_memory=False)
    records["ecg_time"] = pd.to_datetime(records.ecg_time, errors="raise")
    patients["dod"] = pd.to_datetime(patients.dod, errors="raise")
    admissions["dischtime"] = pd.to_datetime(admissions.dischtime, errors="raise")
    if (
        patients.subject_id.duplicated().any()
        or records.study_id.duplicated().any()
        or measurements.study_id.duplicated().any()
    ):
        raise ValueError("Duplicate patient or ECG identifiers in source tables")
    measurements = measurements.drop(columns=["ecg_time"])
    frame = measurements.merge(
        records, on=["subject_id", "study_id"], validate="one_to_one"
    )
    frame = frame.merge(patients, on="subject_id", validate="many_to_one")
    last_ecg = records.groupby("subject_id").ecg_time.max().rename("last_ecg")
    last_discharge = (
        admissions.groupby("subject_id").dischtime.max().rename("last_discharge")
    )
    frame = frame.join(last_ecg, on="subject_id").join(last_discharge, on="subject_id")
    censor_date = pd.concat(
        [frame.last_ecg, frame.last_discharge + pd.Timedelta(days=365)], axis=1
    ).max(axis=1)
    frame["E_Mortality"] = frame.dod.notna()
    frame["TTE_Mortality"] = (censor_date - frame.ecg_time).dt.days
    died = frame.E_Mortality
    frame.loc[died, "TTE_Mortality"] = (
        frame.loc[died, "dod"] - frame.loc[died, "ecg_time"].dt.normalize()
    ).dt.days
    frame["Age"] = frame.ecg_time.dt.year - frame.anchor_year + frame.anchor_age
    if not frame.gender.isin(["M", "F"]).all():
        raise ValueError("Missing or invalid sex")
    frame["Is_Male"] = frame.gender.eq("M")
    frame["ECG_Order"] = -1
    for _, group in frame.groupby("subject_id", sort=True):
        ordered = group.sort_values("ecg_time")
        frame.loc[ordered.index, "ECG_Order"] = np.arange(len(ordered))
    frame = frame.rename(columns={"subject_id": "PID", "study_id": "SID"})
    if not np.isfinite(frame[["Age", "TTE_Mortality", "ECG_Order"]].to_numpy()).all():
        raise ValueError("Missing or nonfinite downstream target metadata")
    return frame


def source_signal(record):
    """Preserve the WFDB channel positions used by the reference MIMIC export."""
    if record.fs != 500:
        raise ValueError(f"Expected 500 Hz, got {record.fs!r} Hz")
    names = record.sig_name
    if names != LEADS:
        raise ValueError(f"Expected MIMIC source lead order {LEADS}, got {names!r}")
    signal = record.p_signal
    if signal is None or signal.shape != (5000, len(names)):
        shape = None if signal is None else signal.shape
        raise ValueError(f"Expected (5000, 12) signal, got {shape}")
    return signal


def exclusion_reason(signal):
    """Apply the raw-signal exclusions from process_ecg.py."""
    if np.isnan(signal).any():
        return "nan_raw_signal"
    if not np.isfinite(signal).all():
        raise ValueError("Infinite raw signal; cannot reproduce reference filtering")
    if np.any(np.ptp(signal, axis=0) < 1e-6) or np.any(signal.std(axis=0) < 1e-6):
        return "null_raw_lead"
    return None


def clean_signal(signal):
    if signal.shape != (5000, 12):
        raise ValueError(f"Expected (5000, 12), got {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError("Nonfinite signal must be logged as excluded before cleaning")
    result = np.stack(
        [
            nk.ecg_clean(
                np.pad(signal[:, lead], (500, 500), mode="reflect"),
                sampling_rate=500,
                method="neurokit",
            )[952:-952]
            for lead in range(12)
        ]
    ).astype(np.float32)
    if result.shape != (12, 4096) or not np.isfinite(result).all():
        raise ValueError("Invalid cleaned ECG")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/mimic/raw")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/mimic/processed"
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}. "
            "Move it aside or choose --output-dir with a new path. "
            "Incomplete exports cannot be resumed."
        )
    frame = metadata(args.input_dir)
    args.output_dir.mkdir(parents=True)
    journal = args.output_dir / "preprocessing.json"
    info = dict(
        status="incomplete",
        input_ecgs=len(frame),
        frequency=500,
        samples=4096,
        leads=LEADS,
        row_order="source",
        raw_exclusions=["nan_raw_signal", "null_raw_lead"],
        versions={
            k: importlib.metadata.version(k)
            for k in ("numpy", "pandas", "neurokit2", "wfdb", "h5py")
        },
    )
    journal.write_text(json.dumps(info, indent=2) + "\n")
    retained, excluded = [], []
    temporary = args.output_dir / "signals.partial.h5"
    with h5py.File(temporary, "w") as handle:
        ecgs = handle.create_dataset(
            "ECG",
            shape=(0, 12, 4096),
            maxshape=(None, 12, 4096),
            dtype="float32",
            chunks=(1, 12, 4096),
            compression="gzip",
        )
        for row in tqdm(
            frame.itertuples(index=False), total=len(frame), desc="MIMIC ECGs"
        ):
            record = wfdb.rdrecord(str(args.input_dir / row.path))
            try:
                raw_signal = source_signal(record)
            except ValueError as exc:
                raise ValueError(f"Invalid ECG {row.SID}: {exc}") from exc
            reason = exclusion_reason(raw_signal)
            if reason is not None:
                excluded.append(
                    dict(PID=row.PID, SID=row.SID, reason=reason)
                )
            else:
                signal = clean_signal(raw_signal)
                index = len(retained)
                ecgs.resize(index + 1, axis=0)
                ecgs[index] = signal
                retained.append(
                    dict(
                        PID=row.PID,
                        SID=row.SID,
                        Age=row.Age,
                        Is_Male=row.Is_Male,
                        E_Mortality=row.E_Mortality,
                        TTE_Mortality=row.TTE_Mortality,
                        ECG_Order=row.ECG_Order,
                        original_row=index,
                    )
                )
    labels = pd.DataFrame(retained)
    if labels.empty:
        raise ValueError("No retained ECGs")
    # Preserve source row order, as in the reference process_ecg.py export.
    with (
        h5py.File(temporary, "r") as src,
        h5py.File(args.output_dir / "ecg.h5", "w") as dst,
    ):
        dst.attrs["complete"] = False
        dst.attrs["sampling_rate"] = 500
        dst.attrs["lead_order"] = LEADS
        signals = dst.create_dataset(
            "ECG",
            shape=(len(labels), 12, 4096),
            dtype="float32",
            chunks=(1, 12, 4096),
            compression="gzip",
        )
        for index, old in enumerate(tqdm(labels.original_row, desc="Aligning ECGs")):
            signals[index] = src["ECG"][int(old)]
        dst.create_dataset("PID", data=labels.PID.to_numpy(dtype=np.int64))
        dst.create_dataset("SID", data=labels.SID.to_numpy(dtype=np.int64))
        labels = labels.drop(columns="original_row")
        labels.to_csv(args.output_dir / "labels.csv", index=False)
        dst.attrs["complete"] = True
    pd.DataFrame(excluded, columns=["PID", "SID", "reason"]).to_csv(
        args.output_dir / "excluded_ecgs.csv", index=False
    )
    info.update(
        status="complete",
        retained_ecgs=len(labels),
        retained_patients=int(labels.PID.nunique()),
        excluded_ecgs=len(excluded),
    )
    journal.write_text(json.dumps(info, indent=2) + "\n")
    temporary.unlink()
    print(f"Complete: {len(labels)} ECGs; {len(excluded)} exclusions recorded")


if __name__ == "__main__":
    main()
