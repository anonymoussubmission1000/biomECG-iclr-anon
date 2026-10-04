"""Evaluate the supplied MIMIC Full, seed 12345 ECGFounder mortality model."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from biomecg.paths import ROOT, data_paths
from biomecg.training import mortality_founder as founder
from biomecg.training.mortality_resnet import (
    ECGDataset,
    calculate_metrics,
    load_cached_split,
    load_labels,
    rows_for_patients,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--labels", type=Path, default=data_paths("mimic")[0])
    parser.add_argument("--ecg", type=Path, default=data_paths("mimic")[1])
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT
        / "outputs/downstream/mimic/seed12345/splits/horizon365d/split.npz",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/founder_example")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(12345)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    labels = load_labels(args.labels, args.ecg, 365)
    _, _, patients = load_cached_split(args.split, labels)
    checkpoint_dir = ROOT / "checkpoints/examples/mimic_full_seed12345"
    protocol = json.loads((checkpoint_dir / "protocol.json").read_text())
    digest = hashlib.sha256("\n".join(sorted(patients)).encode()).hexdigest()
    if digest != protocol["test_patient_sha256"]:
        raise ValueError("These weights require their original test split.")
    rows = rows_for_patients(labels, patients)
    results = []
    architecture = "finetuned"
    name = "early_fusion"
    settings = SimpleNamespace(
        model=name, architecture=f"pretrained_{architecture}"
    )
    model, _ = founder.build_model(settings, device)
    weights = torch.load(
        checkpoint_dir / architecture / f"{name}.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(weights, strict=True)
    leads = founder.EARLY_LEADS
    dataset = ECGDataset(labels, args.ecg, rows, leads)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers
    )
    scores, target, order = founder.predict(
        model, loader, device, f"{architecture}/{name}"
    )
    if not np.array_equal(order, rows) or not np.isfinite(scores).all():
        raise ValueError("Invalid or misaligned predictions")
    results.append(
        dict(
            architecture=architecture,
            model=name,
            n=len(target),
            **calculate_metrics(target, scores, "test"),
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(args.output / "metrics.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
