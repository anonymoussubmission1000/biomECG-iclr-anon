"""Train mortality ECGFounder models on one first ECG per patient.

MIMIC and CODE-15 use the same label selection as the ResNet trainer. CODE-15
retains full HDF5 row alignment while only survival-labelled exams enter splits.
The CODE-15 launcher trains I, II, V1-V6 independently for eight-lead late fusion.
"""

# Workflow:
# 1. Load the labels and ECG file, then validate their structure.
# 2. Keep only the first ECG of each patient (minimum ECG_Order).
# 3. Split patients into stratified train, validation and test sets (70/15/15).
# 4. Build deterministic nested training cohorts from the training patients.
# 5. Train one single-lead or 12-lead model with frozen or fully fine-tuned
#    ECGFounder weights.
# 6. Select the best checkpoint on validation AUROC and evaluate it on test.
# 7. Save the checkpoint, predictions, metrics, split summary and metadata under
#    outputs/downstream/<site>/seed<seed>/.

import argparse
import gc
import json
import os
import random
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
REFACTORED_ROOT = ROOT / "outputs/downstream"
sys.path.append(str(ROOT))

FOUNDER_ROOT = ROOT / "biomecg/study/foundation_assets/founder"
CHECKPOINT_ROOT = ROOT / "biomecg/study/foundation_assets/checkpoints"
if not FOUNDER_ROOT.is_dir():
    raise FileNotFoundError(f"ECGFounder source directory not found: {FOUNDER_ROOT}")
sys.path.append(str(FOUNDER_ROOT))
from net1d import Net1D

LEADS = {
    "I": 0,
    "II": 1,
    "III": 2,
    "aVR": 3,
    "aVL": 4,
    "aVF": 5,
    "V1": 6,
    "V2": 7,
    "V3": 8,
    "V4": 9,
    "V5": 10,
    "V6": 11,
}
EARLY_LEADS = list(range(12))
REPRESENTATION_SIZE = 1024
REQUIRED_COLUMNS = {
    "PID",
    "Age",
    "Is_Male",
    "E_Mortality",
    "TTE_Mortality",
    "ECG_Order",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site",
        required=True,
        help="Site identifier used to isolate splits and results.",
    )
    parser.add_argument("--labels_path", type=Path, required=True)
    parser.add_argument("--ecg_path", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument(
        "--cohort_size",
        default="full",
        help="Number of training patients or 'full'.",
    )
    parser.add_argument(
        "--model",
        choices=[*LEADS, "early_fusion"],
        default="early_fusion",
    )
    parser.add_argument(
        "--architecture",
        choices=["pretrained_frozen", "pretrained_finetuned"],
        default="pretrained_frozen",
    )
    parser.add_argument("--lr_encoder", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cpu_threads", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Create and validate the patient split, then exit.",
    )
    parser.add_argument("--force_rebuild_split", action="store_true")
    return parser.parse_args()


def slug(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mortality_event(labels, horizon):
    died = labels["E_Mortality"].to_numpy() == 1
    within_horizon = labels["TTE_Mortality"].to_numpy() <= horizon
    return (died & within_horizon).astype(np.int8)


def validate_ecg_order(labels):
    numeric_order = pd.to_numeric(labels["ECG_Order"], errors="coerce")

    if numeric_order.isna().any():
        raise ValueError("ECG_Order must be numeric for every ECG.")

    if not np.isfinite(numeric_order.to_numpy()).all():
        raise ValueError("ECG_Order must be finite for every ECG.")

    if not np.equal(numeric_order, np.floor(numeric_order)).all():
        raise ValueError("ECG_Order must contain integers.")

    labels["ECG_Order"] = numeric_order.astype(np.int64)

    if labels.duplicated(["_pid", "ECG_Order"]).any():
        raise ValueError("ECG_Order must be unique within each patient.")


def load_labels(labels_path, ecg_path, horizon):
    from biomecg.study.train_resnet_unified_ecg1 import load_labels as shared_load
    return shared_load(labels_path, ecg_path, horizon)


def first_ecgs(labels):
    from biomecg.study.train_resnet_unified_ecg1 import first_ecgs as shared_first
    return shared_first(labels)


def stratified_order(patient_ids, patient_events, seed):
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(patient_events, return_counts=True)

    if len(classes) != 2:
        raise ValueError(
            "The training pool must contain patients from both outcome classes."
        )

    proportions = counts / counts.sum()
    pools = {
        class_value: rng.permutation(patient_ids[patient_events == class_value])
        for class_value in classes
    }
    positions = {class_value: 0 for class_value in classes}
    selected_counts = np.zeros(len(classes), dtype=int)
    order = []

    for position in range(len(patient_ids)):
        deficits = (position + 1) * proportions - selected_counts
        available = np.array(
            [
                positions[class_value] < len(pools[class_value])
                for class_value in classes
            ]
        )
        class_index = int(np.argmax(np.where(available, deficits, -np.inf)))
        class_value = classes[class_index]
        pool_position = positions[class_value]
        order.append(pools[class_value][pool_position])
        positions[class_value] += 1
        selected_counts[class_index] += 1

    return np.asarray(order, dtype=str)


def split_paths(args):
    folder = (
        REFACTORED_ROOT
        / slug(args.site)
        / f"seed{args.seed}"
        / "splits"
        / f"horizon{args.horizon}d"
    )
    return (
        folder / "split.npz",
        folder / "summary.csv",
        folder / "metadata.json",
    )


def summarize_split(
    labels,
    train_order,
    validation_patients,
    test_patients,
    requested_sizes,
):
    event_by_patient = first_ecgs(labels).set_index("_pid")["_event"]
    groups = {
        "train_full": train_order,
        "validation": validation_patients,
        "test": test_patients,
    }
    groups.update(
        {
            f"train_{size}": train_order[:size]
            for size in requested_sizes
            if size <= len(train_order)
        }
    )

    rows = []
    for group_name, patients in groups.items():
        patient_events = event_by_patient.loc[patients].to_numpy()
        event_count = int(patient_events.sum())
        event_rate = float(patient_events.mean())
        rows.append(
            {
                "group": group_name,
                "n_patients": len(patients),
                "n_ecgs": len(patients),
                "patient_events": event_count,
                "patient_event_rate": event_rate,
                "ecg_events": event_count,
                "ecg_event_rate": event_rate,
            }
        )

    return pd.DataFrame(rows)


def load_cached_split(split_path, labels):
    with np.load(split_path) as split:
        required_arrays = {
            "train_patient_order",
            "val_patients",
            "test_patients",
        }
        missing_arrays = required_arrays - set(split.files)
        if missing_arrays:
            raise ValueError(
                f"Cached split is missing arrays: {sorted(missing_arrays)}. "
                "Rebuild it with --force_rebuild_split."
            )

        train_order = split["train_patient_order"].astype(str)
        validation_patients = split["val_patients"].astype(str)
        test_patients = split["test_patients"].astype(str)

    for patients in (train_order, validation_patients, test_patients):
        if len(patients) != len(set(patients)):
            raise ValueError("Duplicate patients in saved split")
    saved_patients = set(
        np.concatenate([train_order, validation_patients, test_patients])
    )
    current_patients = set(first_ecgs(labels)["_pid"].unique())
    if saved_patients != current_patients:
        raise ValueError(
            f"Cached split does not match the current patients: {split_path}. "
            "Verify the input data and rebuild the split."
        )

    train_set = set(train_order)
    validation_set = set(validation_patients)
    test_set = set(test_patients)
    if train_set & validation_set:
        raise ValueError("Patient leakage detected between train and validation.")
    if train_set & test_set:
        raise ValueError("Patient leakage detected between train and test.")
    if validation_set & test_set:
        raise ValueError("Patient leakage detected between validation and test.")

    return train_order, validation_patients, test_patients


def create_or_load_split(labels, args):
    split_path, summary_path, metadata_path = split_paths(args)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    if split_path.exists() and not args.force_rebuild_split:
        split_arrays = load_cached_split(split_path, labels)
        return (*split_arrays, split_path)

    patient_table = first_ecgs(labels)[
        ["_pid", "_event", "ECG_Order", "_row"]
    ].reset_index(drop=True)
    patient_ids = patient_table["_pid"].to_numpy()
    patient_events = patient_table["_event"].to_numpy()

    classes, class_counts = np.unique(patient_events, return_counts=True)
    if len(classes) != 2:
        raise ValueError(
            f"A stratified split requires two outcome classes; found {classes.tolist()} "
            f"for site={args.site}, horizon={args.horizon}."
        )
    if class_counts.min() < 7:
        raise ValueError(
            "A stratified 70/15/15 split requires at least seven patients in "
            f"each class; found counts {class_counts.tolist()} for site={args.site}, "
            f"horizon={args.horizon}."
        )

    train_patients, temporary_patients = train_test_split(
        patient_ids,
        test_size=0.30,
        random_state=args.seed,
        stratify=patient_events,
    )
    event_by_patient = dict(zip(patient_ids, patient_events))
    temporary_events = np.asarray(
        [event_by_patient[patient] for patient in temporary_patients]
    )
    validation_patients, test_patients = train_test_split(
        temporary_patients,
        test_size=0.50,
        random_state=args.seed + 1,
        stratify=temporary_events,
    )

    train_events = np.asarray(
        [event_by_patient[patient] for patient in train_patients]
    )
    train_order = stratified_order(train_patients, train_events, args.seed)
    validation_patients = np.asarray(validation_patients, dtype=str)
    test_patients = np.asarray(test_patients, dtype=str)

    np.savez_compressed(
        split_path,
        train_patient_order=train_order,
        val_patients=validation_patients,
        test_patients=test_patients,
    )

    standard_sizes = [500, 1000, 2000, 4000, 8000, 16000, 50000, 100000]
    summary = summarize_split(
        labels,
        train_order,
        validation_patients,
        test_patients,
        standard_sizes,
    )
    summary.to_csv(summary_path, index=False)

    metadata = {
        "site": args.site,
        "horizon_days": args.horizon,
        "seed": args.seed,
        "labels_path": str(args.labels_path.resolve()),
        "ecg_path": str(args.ecg_path.resolve()),
        "split": "patient-level stratified 70/15/15",
        "ecg_selection": "one ECG per patient: minimum ECG_Order",
        "patient_stratification_label": (
            "mortality label of first ECG (minimum ECG_Order)"
        ),
        "nested_cohorts": "prefixes of train_patient_order",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return train_order, validation_patients, test_patients, split_path


def rows_for_patients(labels, patients):
    reference = first_ecgs(labels).set_index("_pid")
    patient_ids = np.asarray(patients, dtype=str)
    missing_patients = set(patient_ids) - set(reference.index)

    if missing_patients:
        examples = sorted(missing_patients)[:5]
        raise ValueError(f"Patients missing from labels: {examples}")

    rows = reference.loc[patient_ids, "_row"].to_numpy(dtype=int)
    if len(rows) != len(patient_ids):
        raise AssertionError(
            "The number of selected ECG rows differs from the patient count."
        )

    if len(np.unique(rows)) != len(rows):
        raise AssertionError("A first ECG row was assigned to multiple patients.")

    selected_order = labels.iloc[rows]["ECG_Order"].to_numpy()
    minimum_order = reference.loc[patient_ids, "ECG_Order"].to_numpy()
    if not np.array_equal(selected_order, minimum_order):
        raise AssertionError("A selected ECG is not the first patient ECG.")

    return rows


class ECGDataset(Dataset):
    def __init__(self, labels, ecg_path, rows, lead_indices):
        self.rows = np.asarray(rows)
        self.lead_indices = np.asarray(lead_indices)
        self.ecg_path = str(ecg_path)
        self.labels = labels["_event"].to_numpy(dtype=np.float32)
        self.age = labels["Age"].to_numpy(dtype=np.float32) / 100.0
        self.sex = labels["Is_Male"].to_numpy(dtype=np.float32)
        self.ecg = None

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["ecg"] = None
        return state

    def __del__(self):
        if self.ecg is not None:
            self.ecg.close()

    def __getitem__(self, item):
        row = int(self.rows[item])
        if self.ecg is None:
            self.ecg = h5py.File(self.ecg_path, "r")

        signal = self.ecg["ECG"][row, self.lead_indices].astype(
            np.float32,
            copy=False,
        )
        mean = signal.mean(axis=1, keepdims=True)
        standard_deviation = signal.std(axis=1, keepdims=True)
        signal = (signal - mean) / (standard_deviation + 1e-8)
        metadata = np.asarray(
            [self.age[row], self.sex[row]],
            dtype=np.float32,
        )

        return (
            torch.from_numpy(signal),
            torch.from_numpy(metadata),
            torch.tensor(self.labels[row]),
            torch.tensor(row, dtype=torch.long),
        )


class MortalityModel(nn.Module):
    def __init__(self, encoder, frozen):
        super().__init__()
        self.encoder = encoder
        self.frozen = frozen
        self.head = nn.Sequential(
            nn.Linear(REPRESENTATION_SIZE + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        if frozen:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, ecg, metadata):
        if self.frozen:
            with torch.no_grad():
                embedding = self.encoder(ecg)
        else:
            embedding = self.encoder(ecg)

        if embedding.ndim != 2 or embedding.shape[1] != REPRESENTATION_SIZE:
            raise ValueError(
                f"Expected encoder output with shape [batch, {REPRESENTATION_SIZE}]; "
                f"received {tuple(embedding.shape)}."
            )

        features = torch.cat([embedding, metadata], dim=1)
        return self.head(features).squeeze(1)


class ECGFounderEncoder(nn.Module):
    def __init__(self, number_of_leads, checkpoint_path):
        super().__init__()
        self.network = Net1D(in_channels=number_of_leads, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 1024, 1024], m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16, stride=2, groups_width=16, verbose=False, use_bn=False, use_do=False, n_classes=150, return_features=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" not in checkpoint:
            raise KeyError(f"state_dict is missing from ECGFounder checkpoint: {checkpoint_path}")
        self.network.load_state_dict(checkpoint["state_dict"], strict=True)
        self.network.dense = nn.Identity()

    def forward(self, ecg):
        _, embedding = self.network(ecg)
        return embedding


def build_encoder(args, device):
    """Shared encoder initialization for mortality and ECG-only outcomes."""
    number_of_leads = 12 if args.model == "early_fusion" else 1
    checkpoint_path = CHECKPOINT_ROOT / f"{number_of_leads}_lead_ECGFounder.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"ECGFounder checkpoint not found: {checkpoint_path}")
    encoder = ECGFounderEncoder(number_of_leads, checkpoint_path)
    frozen = args.architecture == "pretrained_frozen"
    return encoder, frozen, checkpoint_path.name


def build_model(args, device):
    encoder, frozen, checkpoint_name = build_encoder(args, device)
    return MortalityModel(encoder, frozen).to(device), checkpoint_name


def calculate_metrics(labels, scores, prefix):
    unique_labels = np.unique(labels)
    if len(unique_labels) != 2:
        raise ValueError(
            f"{prefix} contains {len(unique_labels)} class instead of two."
        )

    predicted_labels = scores >= 0.5
    matrix = confusion_matrix(labels, predicted_labels, labels=[0, 1])
    return {
        f"{prefix}_auc": roc_auc_score(labels, scores),
        f"{prefix}_pr_auc": average_precision_score(labels, scores),
        f"{prefix}_tn": int(matrix[0, 0]),
        f"{prefix}_fp": int(matrix[0, 1]),
        f"{prefix}_fn": int(matrix[1, 0]),
        f"{prefix}_tp": int(matrix[1, 1]),
    }


def predict(model, loader, device, description):
    model.eval()
    all_scores = []
    all_labels = []
    all_rows = []

    with torch.no_grad():
        for ecg, metadata, labels, rows in tqdm(
            loader,
            desc=description,
            leave=False,
        ):
            ecg = ecg.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(ecg, metadata)

            all_scores.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.numpy())
            all_rows.append(rows.numpy())

    if not all_scores:
        raise ValueError(f"The {description} DataLoader is empty.")

    return (
        np.concatenate(all_scores),
        np.concatenate(all_labels).astype(int),
        np.concatenate(all_rows),
    )


def output_dir(args):
    architecture = f"ecgfounder_{args.architecture.removeprefix('pretrained_')}"
    return (
        REFACTORED_ROOT
        / slug(args.site)
        / f"seed{args.seed}"
        / "runs"
        / f"horizon{args.horizon}d"
        / f"cohort{args.cohort_size}"
        / architecture
        / args.model
    )


def make_loader(labels, args, rows, lead_indices, shuffle, device):
    dataset = ECGDataset(labels, args.ecg_path, rows, lead_indices)
    return DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def train_model(
    model,
    train_loader,
    validation_loader,
    train_labels,
    args,
    device,
    checkpoint_path,
):
    positives = int(train_labels.sum())
    negatives = int(len(train_labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("The training cohort must contain both outcome classes.")

    positive_weight = torch.tensor([negatives / positives], device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    parameter_groups = [
        {"params": model.head.parameters(), "lr": args.lr_head}
    ]
    if not model.frozen:
        parameter_groups.append(
            {"params": model.encoder.parameters(), "lr": args.lr_encoder}
        )

    optimizer = torch.optim.AdamW(parameter_groups)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_auc = -np.inf
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        if model.frozen and model.encoder.training:
            raise AssertionError("The frozen encoder left evaluation mode.")

        train_losses = []
        for ecg, metadata, labels, _ in train_loader:
            ecg = ecg.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(ecg, metadata)
                loss = loss_function(logits, labels)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}."
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        if not train_losses:
            raise ValueError("The training DataLoader produced no batch.")

        validation_scores, validation_labels, _ = predict(
            model,
            validation_loader,
            device,
            f"Validation epoch {epoch}",
        )
        validation_metrics = calculate_metrics(
            validation_labels,
            validation_scores,
            "validation",
        )
        validation_auc = validation_metrics["validation_auc"]
        validation_pr_auc = validation_metrics["validation_pr_auc"]
        mean_train_loss = float(np.mean(train_losses))

        print(
            f"epoch={epoch:03d} "
            f"train_loss={mean_train_loss:.5f} "
            f"val_auc={validation_auc:.5f} "
            f"val_pr_auc={validation_pr_auc:.5f}"
        )

        if validation_auc > best_auc:
            best_auc = validation_auc
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_auc": validation_auc,
                    "validation_pr_auc": validation_pr_auc,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            break

    if not checkpoint_path.exists():
        raise RuntimeError(
            f"Training finished without creating a checkpoint: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def main():
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed)

    labels = load_labels(args.labels_path, args.ecg_path, args.horizon)
    (
        train_order,
        validation_patients,
        test_patients,
        split_path,
    ) = create_or_load_split(labels, args)

    if args.prepare_only:
        print(f"Prepared split: {split_path}")
        print(
            f"Patients: train={len(train_order)}, "
            f"validation={len(validation_patients)}, "
            f"test={len(test_patients)}"
        )
        return

    if str(args.cohort_size).lower() == "full":
        cohort_size = len(train_order)
    else:
        cohort_size = int(args.cohort_size)

    if cohort_size <= 0:
        raise ValueError("cohort_size must be positive.")

    if cohort_size > len(train_order):
        raise ValueError(
            f"Requested {cohort_size} training patients, but only "
            f"{len(train_order)} are available."
        )

    train_patients = train_order[:cohort_size]
    train_rows = rows_for_patients(labels, train_patients)
    validation_rows = rows_for_patients(labels, validation_patients)
    test_rows = rows_for_patients(labels, test_patients)

    if args.model == "early_fusion":
        lead_indices = EARLY_LEADS
    else:
        lead_indices = [LEADS[args.model]]

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Use --device cpu only "
            "for debugging."
        )

    model, config_id = build_model(args, device)
    experiment_dir = output_dir(args)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = experiment_dir / "best.pt"
    metrics_path = experiment_dir / "metrics.csv"

    train_loader = make_loader(
        labels,
        args,
        train_rows,
        lead_indices,
        True,
        device,
    )
    validation_loader = make_loader(
        labels,
        args,
        validation_rows,
        lead_indices,
        False,
        device,
    )
    test_loader = make_loader(
        labels,
        args,
        test_rows,
        lead_indices,
        False,
        device,
    )

    checkpoint = train_model(
        model,
        train_loader,
        validation_loader,
        labels.iloc[train_rows]["_event"].to_numpy(),
        args,
        device,
        checkpoint_path,
    )
    test_scores, test_labels, test_row_order = predict(
        model,
        test_loader,
        device,
        "Test",
    )

    predictions = pd.DataFrame(
        {
            "row_index": test_row_order,
            "pid": labels.iloc[test_row_order]["_pid"].to_numpy(),
            "ecg_order": labels.iloc[test_row_order]["ECG_Order"].to_numpy(),
            "label": test_labels,
            "prediction": test_scores,
        }
    ).sort_values("row_index")

    if predictions["pid"].duplicated().any():
        raise AssertionError("Test predictions contain multiple ECGs per patient.")

    predictions.to_csv(experiment_dir / "predictions.csv", index=False)
    patient_predictions = predictions.copy()
    patient_predictions["n_ecgs"] = 1
    patient_predictions.to_csv(
        experiment_dir / "predictions_patient.csv",
        index=False,
    )

    reference_events = first_ecgs(labels).set_index("_pid")["_event"]
    metrics = {
        "site": args.site,
        "seed": args.seed,
        "horizon_days": args.horizon,
        "cohort_label": args.cohort_size,
        "architecture": f"ecgfounder_{args.architecture.removeprefix('pretrained_')}",
        "model": args.model,
        "config_id": config_id,
        "lr_encoder": args.lr_encoder,
        "lr_head": args.lr_head,
        "batch_size": args.bs,
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_auc_ecg": float(checkpoint["validation_auc"]),
        "patient_reference_ecg": "first_min_ECG_Order",
        "n_train_patients": len(train_patients),
        "n_val_patients": len(validation_patients),
        "n_test_patients": len(test_patients),
        "n_train_ecgs": len(train_rows),
        "n_val_ecgs": len(validation_rows),
        "n_test_ecgs": len(test_rows),
        "train_event_rate_patient": float(
            reference_events.loc[train_patients].mean()
        ),
        "val_event_rate_patient": float(
            reference_events.loc[validation_patients].mean()
        ),
        "test_event_rate_patient": float(
            reference_events.loc[test_patients].mean()
        ),
        "train_event_rate_ecg": float(
            labels.iloc[train_rows]["_event"].mean()
        ),
        "split_path": str(split_path),
    }
    metrics.update(
        calculate_metrics(
            predictions["label"].to_numpy(),
            predictions["prediction"].to_numpy(),
            "test_ecg",
        )
    )
    metrics.update(
        calculate_metrics(
            patient_predictions["label"].to_numpy(),
            patient_predictions["prediction"].to_numpy(),
            "test_patient",
        )
    )
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(pd.Series(metrics).to_string())

    del model
    del train_loader
    del validation_loader
    del test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
