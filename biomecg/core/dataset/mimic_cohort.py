"""MIMIC/CODE-15 biometric manifests from saved downstream training patients.

Keep the original MIMIC selection and fingerprints. CODE-15 uses its administrative
session index, never missing survival labels, to order sessions. Fixed train/val
and test counts are required; no smaller cohort is silently substituted.
"""

import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path
import tempfile

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm


def _fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def configure_pretraining_cohort(config, n_train_val=None):
    """Resolve size-specific checkpoint, inference and manifest names.

    Size counts train + validation patients, not sessions or pairs. Each size
    uses the first N eligible patients in the same saved downstream order, so
    train/validation pools are nested (their random partitions need not be).
    The next n_test eligible patients form that size's biometric test: biometric
    test sets differ across sizes, and are never the downstream mortality test.
    """
    config = deepcopy(config)
    settings = config["biometrics"]
    size = settings["n_train_val"] if n_train_val is None else n_train_val
    if isinstance(size, bool) or not isinstance(size, int) or size < 10:
        raise ValueError("n_train_val must be an integer >= 10")
    rate = config["training"]["rate_validation"]
    if not 0 < rate < 1 or min(int(size * (1 - rate)), int(np.ceil(size * rate))) < 2:
        raise ValueError("Train and validation each need at least two patients for negative pairs")
    settings["n_train_val"] = size
    base = re.sub(r"(?:_cohort\d+)?_seed\d+$", "", config["name"])
    base = re.sub(r"_cohort\d+$", "", base)
    suffix = f"_cohort{size}"
    config["name"] = f"{base}{suffix}_seed{config['seed']}"
    # Isolate sizes even when the supplied config was already resolved for
    # another experiment size.
    manifest = Path(settings["manifest_path"])
    stem = re.sub(r"_cohort(?:\d+|\{n_train_val\})$", "", manifest.stem)
    settings["manifest_path"] = str(manifest.with_name(f"{stem}{suffix}{manifest.suffix}"))
    return config


def prepare_cohort(config, project_root):
    """Select the configured train/val and test sizes with >=2 valid sessions.

    All patients come from the saved downstream train order. The first eligible
    patients form train/val; the following eligible patients form biometric test.
    """
    root = Path(project_root)
    source = config["dir_input"][config["run_from"]][config["DB"]]
    ecg_path, labels_path = Path(source["ecg"]), Path(source["labels"])
    settings = config["biometrics"]
    split_path = root / settings["split_path"].format(seed=config["seed"], n_train_val=settings["n_train_val"])
    manifest_path = root / settings["manifest_path"].format(seed=config["seed"], n_train_val=settings["n_train_val"])
    n_pool, n_test = settings["n_train_val"], settings["n_test"]
    provenance = {
        "version": 1,
        "seed": config["seed"],
        "n_train_val": n_pool,
        "n_test": n_test,
        "rate_validation": config["training"]["rate_validation"],
        "min_valid_sessions": 2,
        "validity": "all 12 leads finite and std > 1e-6 over full strip",
        "ecg": _fingerprint(ecg_path),
        "labels": _fingerprint(labels_path),
        "split": _fingerprint(split_path),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    }
    with np.load(split_path, allow_pickle=False) as split:
        order = split["train_patient_order"].astype(str).tolist()
        downstream_val = set(split["val_patients"].astype(str))
        downstream_test = set(split["test_patients"].astype(str))
    if len(set(order)) != len(order):
        raise ValueError("Duplicate patients in downstream train order")
    if set(order) & (downstream_val | downstream_test) or downstream_val & downstream_test:
        raise ValueError("Overlapping downstream patient splits")

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["provenance"] != provenance:
            raise ValueError(f"Cohort inputs changed; use a new manifest_path: {manifest_path}")
        _validate_manifest(manifest, order, n_pool, n_test)
        return manifest, manifest_path

    labels = pd.read_csv(labels_path, dtype={"PID": str, "SID": str})
    order_column = "ECG_Session_Index" if config["DB"] == "CODE15" else "ECG_Order"
    if labels[["PID", "SID", order_column]].isna().any().any() or labels["SID"].duplicated().any():
        raise ValueError("Labels need non-missing PID/SID/session index and unique SID")
    labels = labels.sort_values(["PID", order_column, "SID"], kind="stable")
    records = labels.groupby("PID", sort=False)["SID"].agg(list).to_dict()
    selected, valid_sids = [], {}
    with h5py.File(ecg_path, "r") as handle:
        ecgs = handle["ECG"]
        sids, pids = handle["SID"][:], handle["PID"][:]
        if ecgs.ndim != 3 or ecgs.shape[1] != 12 or ecgs.shape[2] < 4096:
            raise ValueError("Expected ECG shape (N, 12, samples >= 4096)")
        if len(ecgs) != len(sids) or len(ecgs) != len(pids) or len(np.unique(sids)) != len(sids):
            raise ValueError("Invalid HDF5 ECG/PID/SID alignment")
        row_by_sid = {str(sid): row for row, sid in enumerate(sids)}
        with tqdm(total=n_pool + n_test, desc=f"Selecting eligible {config['DB']} patients") as progress:
            for pid in order:
                patient_sids = records.get(pid, [])
                if len(patient_sids) < 2:
                    continue
                valid = []
                for sid in patient_sids:
                    row = row_by_sid[sid]
                    if str(pids[row]) != pid:
                        raise ValueError(f"PID mismatch for SID {sid}")
                    arr = ecgs[row]
                    if np.isfinite(arr).all() and np.all(arr.std(axis=1) > 1e-6):
                        valid.append(sid)
                if len(valid) >= 2:
                    selected.append(pid)
                    valid_sids[pid] = valid
                    progress.update(1)
                if len(selected) == n_pool + n_test:
                    break
    if len(selected) != n_pool + n_test:
        raise ValueError(f"Only {len(selected)} eligible patients; need {n_pool + n_test}")

    pool = selected[:n_pool]
    # Identical splitter to excluded source; every MIMIC patient belongs to the multi stratum.
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=config["training"]["rate_validation"],
        random_state=config["seed"],
    )
    train_idx, val_idx = next(splitter.split(pool, np.ones(len(pool), dtype=int)))
    manifest = {
        "provenance": provenance,
        "train_val_patients": pool,
        "train_patients": [pool[i] for i in train_idx],
        "val_patients": [pool[i] for i in val_idx],
        "test_patients": selected[n_pool:],
        "valid_sids": valid_sids,
        "downstream_train_prefix_required": order.index(selected[-1]) + 1,
        "downstream_cohort_overlap": {
            str(n): {
                group: len(set(patients) & set(order[:n]))
                for group, patients in {
                    "biometric_train_val": pool, "biometric_test": selected[n_pool:]
                }.items()
            }
            for n in [500, 1000, 2000, 4000, 8000, 16000, 50000, 100000, len(order)]
            if n <= len(order)
        },
    }
    _validate_manifest(manifest, order, n_pool, n_test)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete file atomically, without replacing a concurrent run's file.
    with tempfile.NamedTemporaryFile(mode="w", dir=manifest_path.parent, delete=False) as stream:
        temporary_path = Path(stream.name)
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    try:
        try:
            os.link(temporary_path, manifest_path)
        except FileExistsError:
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError(f"Conflicting cohort manifest: {manifest_path}")
    finally:
        temporary_path.unlink()
    return manifest, manifest_path


def _validate_manifest(manifest, order, n_pool, n_test):
    pool = manifest["train_val_patients"]
    train, val, test = (manifest[f"{group}_patients"] for group in ("train", "val", "test"))
    selected = pool + test
    if len(pool) != n_pool or len(test) != n_test or len(set(selected)) != len(selected):
        raise ValueError("Incorrect biometric cohort sizes or duplicate patients")
    if set(train) & set(val) or set(train) | set(val) != set(pool):
        raise ValueError("Invalid biometric train/validation partition")
    if len(train) + len(val) != n_pool:
        raise ValueError("Duplicate train/validation patients")
    selected_set = set(selected)
    if [pid for pid in order if pid in selected_set] != selected:
        raise ValueError("Biometric cohort does not follow downstream training order")
    if any(len(manifest["valid_sids"][pid]) < 2 for pid in selected):
        raise ValueError("Every biometric patient must have at least two valid sessions")
    provenance = manifest["provenance"]
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=provenance["rate_validation"], random_state=provenance["seed"]
    )
    ti, vi = next(splitter.split(pool, np.ones(len(pool), dtype=int)))
    if train != [pool[i] for i in ti] or val != [pool[i] for i in vi]:
        raise ValueError("Manifest train/validation split differs from excluded source splitter")
