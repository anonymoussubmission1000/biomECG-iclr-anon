"""Recompute the paper's patient-identity UMAP from local MIMIC ECGs."""

import argparse
from pathlib import Path
import h5py
import numpy as np
import torch
import umap
from biomecg.paths import ROOT, data_paths
from biomecg.models.network import ClassificationNetwork
from biomecg.training.mortality_resnet import load_labels
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/recomputed_artifacts"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    labels_path, ecg_path = data_paths("mimic")
    labels = load_labels(labels_path, ecg_path, 365)
    counts = labels.groupby("_pid").size()
    eligible = np.array(sorted(counts[counts >= 2].index))
    selected = np.random.default_rng(12345).choice(eligible, 50, replace=False)
    identity = (
        labels[labels._pid.isin(selected)]
        .sort_values(["_pid", "ECG_Order"])
        .reset_index(drop=True)
    )
    names = {pid: f"P{i + 1:02d}" for i, pid in enumerate(sorted(selected))}
    identity["patient_display"] = identity._pid.map(names)
    chosen = np.random.default_rng(12345).choice(
        np.array(sorted(identity.patient_display.unique())), 40, replace=False
    )
    identity = (
        identity[identity.patient_display.isin(chosen)]
        .sort_values(["patient_display", "ECG_Order", "_row"])
        .groupby("patient_display", sort=False)
        .head(5)
        .reset_index(drop=True)
    )
    checkpoint = ROOT / "checkpoints/biometric/MIMIC_0201_II_cohort1359_seed12345"
    config = json.loads((checkpoint / "run_config.json").read_text())
    config["phase"] = "test"
    config["training"]["device"] = args.device
    encoder = ClassificationNetwork(config).network.encoder.eval()
    with h5py.File(ecg_path, "r") as source:
        batch = np.stack(
            [
                np.asarray(source["ECG"][int(i), 1:2, :], dtype=np.float32)
                for i in identity._row
            ]
        )
    center = batch.shape[-1] // 2
    batch = batch[:, :, center - 2048 : center + 2048]
    std = batch.std(axis=-1, keepdims=True)
    if not np.isfinite(batch).all() or (std <= 1e-6).any():
        raise ValueError("Invalid ECGs in the selected identities")
    batch = (batch - batch.mean(axis=-1, keepdims=True)) / (std + 1e-6)
    with torch.inference_mode():
        features = encoder(torch.from_numpy(batch).to(args.device)).cpu().numpy()
    coordinates = [
        umap.UMAP(
            n_components=2,
            n_neighbors=k,
            min_dist=d,
            metric=m,
            random_state=12345,
            n_jobs=1,
        ).fit_transform(features)
        for k, d, m in [(15, 0.1, "euclidean"), (50, 0.5, "cosine")]
    ]
    identity[["patient_display"]].to_csv(
        args.output / "umap_display_labels.csv", index=False
    )
    np.savez_compressed(
        args.output / "umap_coordinates.npz",
        view_0=coordinates[0],
        view_1=coordinates[1],
    )
    print("Recomputed identity projections without exporting patient identifiers")


if __name__ == "__main__":
    main()
