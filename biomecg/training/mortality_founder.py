from biomecg.training.run_state import begin_run, finish_training

"""Train mortality ECGFounder models on one first ECG per patient.

MIMIC and CODE-15 use the same label selection as the ResNet trainer. CODE-15
retains full HDF5 row alignment while only survival-labelled exams enter splits.
The CODE-15 launcher trains I, II, V1-V6 independently for eight-lead late fusion.
"""
import argparse
import gc
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "outputs/downstream"
FOUNDER_ROOT = ROOT / "third_party/ecgfounder"
CHECKPOINT_ROOT = ROOT / "checkpoints/ecgfounder"
from third_party.ecgfounder.net1d import Net1D
from biomecg.training.mortality_resnet import (
    calculate_metrics,
    create_or_load_split,
    first_ecgs,
    load_labels,
    make_loader,
    rows_for_patients,
    set_seed,
    slug,
)

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
        "--cohort_size", default="full", help="Number of training patients or 'full'."
    )
    parser.add_argument(
        "--model", choices=[*LEADS, "early_fusion"], default="early_fusion"
    )
    parser.add_argument(
        "--architecture",
        choices=["pretrained_frozen", "pretrained_finetuned"],
        default="pretrained_frozen",
    )
    parser.add_argument("--lr_encoder", type=float, default=0.0001)
    parser.add_argument("--lr_head", type=float, default=0.001)
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
                f"Expected encoder output with shape [batch, {REPRESENTATION_SIZE}]; received {tuple(embedding.shape)}."
            )
        features = torch.cat([embedding, metadata], dim=1)
        return self.head(features).squeeze(1)


class ECGFounderEncoder(nn.Module):
    def __init__(self, number_of_leads, checkpoint_path):
        super().__init__()
        self.network = Net1D(
            in_channels=number_of_leads,
            base_filters=64,
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            use_bn=False,
            use_do=False,
            n_classes=150,
            return_features=True,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "state_dict" not in checkpoint:
            raise KeyError(
                f"state_dict is missing from ECGFounder checkpoint: {checkpoint_path}"
            )
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
    return (encoder, frozen, checkpoint_path.name)


def build_model(args, device):
    encoder, frozen, checkpoint_name = build_encoder(args, device)
    return (MortalityModel(encoder, frozen).to(device), checkpoint_name)


def predict(model, loader, device, description):
    model.eval()
    all_scores = []
    all_labels = []
    all_rows = []
    with torch.no_grad():
        for ecg, metadata, labels, rows in tqdm(loader, desc=description, leave=False):
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
        RESULTS_ROOT
        / slug(args.site)
        / f"seed{args.seed}"
        / "runs"
        / f"horizon{args.horizon}d"
        / f"cohort{args.cohort_size}"
        / architecture
        / args.model
    )


def train_model(
    model, train_loader, validation_loader, train_labels, args, device, checkpoint_path
):
    positives = int(train_labels.sum())
    negatives = int(len(train_labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("The training cohort must contain both outcome classes.")
    positive_weight = torch.tensor([negatives / positives], device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    parameter_groups = [{"params": model.head.parameters(), "lr": args.lr_head}]
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
                    raise ValueError("Nonfinite training loss")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
        if not train_losses:
            raise ValueError("The training DataLoader produced no batch.")
        validation_scores, validation_labels, _ = predict(
            model, validation_loader, device, f"Validation epoch {epoch}"
        )
        validation_metrics = calculate_metrics(
            validation_labels, validation_scores, "validation"
        )
        validation_auc = validation_metrics["validation_auc"]
        validation_pr_auc = validation_metrics["validation_pr_auc"]
        mean_train_loss = float(np.mean(train_losses))
        print(
            f"epoch={epoch:03d} train_loss={mean_train_loss:.5f} val_auc={validation_auc:.5f} val_pr_auc={validation_pr_auc:.5f}"
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def main():
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)
    set_seed(args.seed)
    labels = load_labels(args.labels_path, args.ecg_path, args.horizon)
    train_order, validation_patients, test_patients, split_path = create_or_load_split(
        labels, args
    )
    if args.prepare_only:
        print(f"Prepared split: {split_path}")
        print(
            f"Patients: train={len(train_order)}, validation={len(validation_patients)}, test={len(test_patients)}"
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
            f"Requested {cohort_size} training patients, but only {len(train_order)} are available."
        )
    train_patients = train_order[:cohort_size]
    train_rows = rows_for_patients(labels, train_patients)
    validation_rows = rows_for_patients(labels, validation_patients)
    test_rows = rows_for_patients(labels, test_patients)
    if args.model == "early_fusion":
        lead_indices = EARLY_LEADS
    else:
        lead_indices = [LEADS[args.model]]
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError(
            "CUDA was requested but is unavailable. Use --device cpu only for debugging."
        )
    experiment_dir = output_dir(args)
    initialization = (
        CHECKPOINT_ROOT
        / f"{12 if args.model == 'early_fusion' else 1}_lead_ECGFounder.pth"
    )
    action = begin_run(experiment_dir, args, split_path, [initialization])
    if action == "skip":
        return
    model, config_id = build_model(args, device)
    experiment_dir = output_dir(args)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = experiment_dir / "best.pt"
    metrics_path = experiment_dir / "metrics.csv"
    train_loader = make_loader(labels, args, train_rows, lead_indices, True, device)
    validation_loader = make_loader(
        labels, args, validation_rows, lead_indices, False, device
    )
    test_loader = make_loader(labels, args, test_rows, lead_indices, False, device)
    if action == "train":
        checkpoint = train_model(
            model,
            train_loader,
            validation_loader,
            labels.iloc[train_rows]["_event"].to_numpy(),
            args,
            device,
            checkpoint_path,
        )
        finish_training(experiment_dir)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    test_scores, test_labels, test_row_order = predict(
        model, test_loader, device, "Test"
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
    patient_predictions.to_csv(experiment_dir / "predictions_patient.csv", index=False)
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
        "train_event_rate_patient": float(reference_events.loc[train_patients].mean()),
        "val_event_rate_patient": float(
            reference_events.loc[validation_patients].mean()
        ),
        "test_event_rate_patient": float(reference_events.loc[test_patients].mean()),
        "train_event_rate_ecg": float(labels.iloc[train_rows]["_event"].mean()),
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
