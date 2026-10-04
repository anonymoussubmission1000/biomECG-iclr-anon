import os
import random
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ECGPairDataset(Dataset):
    def __init__(self, config, root, pids, labels_file):
        self.config = config
        self.seed = config["seed"]
        self.root = Path(root)
        self.ecg_path = self.root
        self.labels_file = Path(labels_file)
        self.global_leads = [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]
        self.selected_leads = config["architecture"]["leads"]
        self.lead_idx = [self.global_leads.index(l) for l in self.selected_leads]
        self._h5 = None
        self._h5_pid = None
        self.patient_strips = {}
        self.pairs = []
        self._load_paths(pids)
        self._generate_pairs()

    def _is_strip_valid(self, arr):
        return bool(np.isfinite(arr).all() and np.all(arr.std(axis=1) > 1e-06))

    def _load_paths(self, pids=None):
        labels = pd.read_csv(self.labels_file, dtype={"PID": str, "SID": str})
        if pids is not None:
            labels = labels[labels["PID"].isin({str(pid) for pid in pids})]
        order_column = (
            "ECG_Session_Index" if self.config["DB"] == "CODE15" else "ECG_Order"
        )
        labels = labels.sort_values(["PID", order_column, "SID"], kind="stable")
        with h5py.File(self.ecg_path, "r") as handle:
            ecgs = handle["ECG"]
            if ecgs.ndim != 3 or ecgs.shape[1] != 12 or ecgs.shape[2] < 4096:
                raise ValueError("MIMIC ECG must have shape (N, 12, samples >= 4096)")
            sids = handle["SID"][:]
            patient_ids = handle["PID"][:]
            if len(sids) != len(ecgs) or len(patient_ids) != len(ecgs):
                raise ValueError("MIMIC ECG/PID/SID lengths differ")
            if len(np.unique(sids)) != len(sids):
                raise ValueError("MIMIC HDF5 SID values must be unique")
            rows_by_sid = {str(sid): row for row, sid in enumerate(sids)}
            for pid, records in labels.groupby("PID", sort=False):
                if len(records) < 2:
                    continue
                sessions = []
                for sid in records["SID"]:
                    if sid not in rows_by_sid:
                        raise ValueError(f"SID {sid} is missing from MIMIC HDF5")
                    row = rows_by_sid[sid]
                    if str(patient_ids[row]) != pid:
                        raise ValueError(f"PID mismatch for MIMIC SID {sid}")
                    if self._is_strip_valid(ecgs[row]):
                        sessions.append([row])
                if len(sessions) >= 2:
                    self.patient_strips[pid] = sessions

    def _generate_pairs(self):
        rng = random.Random(self.seed)
        pairs = []
        if len(self.patient_strips) < 2:
            raise ValueError(
                "Negative pairs require at least two eligible MIMIC patients"
            )
        for pid, sessions in self.patient_strips.items():
            other_pids = [p for p in self.patient_strips if p != pid]
            for s_srcn in range(len(sessions)):
                other_sessions = [i for i in range(len(sessions)) if i != s_srcn]
                for _ in range(5):
                    s_dstn = rng.choice(other_sessions)
                    pairs.append([(pid, s_srcn, 0), (pid, s_dstn, 0), 1])
                    neg_pid = rng.choice(other_pids)
                    neg_sn = rng.randrange(len(self.patient_strips[neg_pid]))
                    pairs.append([(pid, s_srcn, 0), (neg_pid, neg_sn, 0), 0])
        rng.shuffle(pairs)
        self.pairs = pairs

    def _load_strip(self, pid, testn, sn):
        if self._h5 is None or self._h5_pid != os.getpid():
            self.close()
            self._h5 = h5py.File(self.ecg_path, "r")
            self._h5_pid = os.getpid()
        row = self.patient_strips[pid][testn][sn]
        arr = np.asarray(self._h5["ECG"][row], dtype=np.float32)
        center = arr.shape[1] // 2
        arr = arr[self.lead_idx, center - 2048 : center + 2048]
        arr = (arr - arr.mean(axis=1, keepdims=True)) / (
            arr.std(axis=1, keepdims=True) + 1e-06
        )
        return torch.tensor(arr, dtype=torch.float32)

    def close(self):
        if getattr(self, "_h5", None) is not None:
            self._h5.close()
            self._h5 = None
        self._h5_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_h5_pid"] = None
        return state

    def __del__(self):
        self.close()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (pid1, t1n, s1n), (pid2, t2n, s2n), label = self.pairs[idx]
        x1 = self._load_strip(pid1, t1n, s1n)
        x2 = self._load_strip(pid2, t2n, s2n)
        if self.config["phase"] == "test":
            return (
                x1,
                x2,
                torch.tensor(label, dtype=torch.float32),
                pid1,
                s1n,
                0,
                pid2,
                s2n,
                0,
                t1n,
                t2n,
            )
        else:
            return (x1, x2, torch.tensor(label, dtype=torch.float32))
