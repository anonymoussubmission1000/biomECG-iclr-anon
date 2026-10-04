import argparse
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from biomecg.data.cohort import prepare_cohort, configure_pretraining_cohort


def parse_args():
    """Read a config path and optional overrides without modifying its JSON file.

    ``--prepare-only`` prepares/checks the patient manifest and prints cohort
    counts; it does not construct a network or launch training/evaluation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--seed", type=int, help="Override config seed; also suffix the run name"
    )
    parser.add_argument(
        "--cohort-size",
        type=int,
        help="Train + validation patients (default: config, 1359)",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--phase", choices=["train", "test"])
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Save/validate patient manifest, without training",
    )
    return parser.parse_args()


def load_config(args):
    """Load the package config and resolve the seed-specific experiment name.

    CLI seed/phase values take precedence over the JSON values. Replacing an
    existing seed suffix prevents names such as ``..._seed42_seed12345``.
    Hyperparameter search and training resumption are not supported here.
    """
    config = json.loads(args.config.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    if args.phase is not None:
        config["phase"] = args.phase
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.workers is not None:
        config["training"]["num_workers"] = args.workers
    config = configure_pretraining_cohort(config, getattr(args, "cohort_size", None))
    if config["phase"] not in ("train", "test"):
        raise ValueError("MIMIC strip phase must be train or test")
    return config


def make_dataset(config, manifest, group):
    """Build the ``train``, ``val`` or ``test`` dataset from saved patient IDs.

    The dataset maps CSV sessions to HDF5 rows by SID and rechecks signal
    validity. Verify that patient membership and valid session counts still
    match the manifest.
    ``config["phase"]`` controls the output format: training returns signal
    pairs and labels; testing also returns IDs for the package evaluator.
    """
    from biomecg.data.pairs import ECGPairDataset

    source = {key: str(ROOT / value) for key, value in config["input"].items()}
    patients = manifest[f"{group}_patients"]
    dataset = ECGPairDataset(
        config=deepcopy(config),
        root=source["ecg"],
        labels_file=source["labels"],
        pids=patients,
    )
    if set(dataset.patient_strips) != set(patients):
        raise ValueError(
            f"Dataset eligibility changed for {group}; check the cohort manifest"
        )
    if any(
        (
            len(dataset.patient_strips[pid]) != len(manifest["valid_sids"][pid])
            for pid in patients
        )
    ):
        raise ValueError(f"Valid session counts changed for {group}")
    return dataset


def existing_run_action(config, root):
    """Return train/test/skip, requiring matching provenance before any reuse.

    Explicit --phase test reruns evaluation. Ordinary train invocations skip
    complete runs and evaluate existing checkpoints whose outputs are missing.
    A leftover config without weights is treated as interrupted work and fails
    rather than overwriting it. As in the downstream launcher, completion is
    determined by nonempty artifacts, not merely by the experiment directory.
    """
    base = Path(root)
    folder = base / "checkpoints/biometric" / config["name"]
    checkpoint, saved = (folder / "network.pt", folder / "run_config.json")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        if config["phase"] == "test":
            raise FileNotFoundError(f"Checkpoint not found or empty: {checkpoint}")
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(
                f"Incomplete run, inspect before restarting: {folder}"
            )
        return "train"
    if not saved.is_file():
        raise FileNotFoundError(f"Saved training config not found: {saved}")
    if not (folder / "training_complete.json").is_file():
        raise FileExistsError(f"Interrupted biometric training: {folder}")
    trained = json.loads(saved.read_text())
    for key in ("name", "seed", "DB", "network", "architecture", "biometrics", "input"):
        if trained[key] != config[key]:
            raise ValueError(f"Existing training config differs for {key}: {saved}")
    for key, value in config["training"].items():
        if (
            key not in ("device", "num_workers", "pin_memory", "resume")
            and trained["training"].get(key) != value
        ):
            raise ValueError(
                f"Existing training config differs for training.{key}: {saved}"
            )
    if config["phase"] == "test":
        return "test"
    inference = base / "outputs/biometric" / config["name"] / "test_multi"
    outputs = [
        inference / f"results_test_multi{suffix}"
        for suffix in (".csv", "_scores.csv", "_confusion.png", "_roc.png")
    ]
    return (
        "skip"
        if all((p.is_file() and p.stat().st_size > 0 for p in outputs))
        else "test"
    )


def main():
    """Prepare patients, optionally train, then evaluate the best checkpoint.

    Keep cohort preparation separate from model training so all nine configs
    can reuse one selection per seed. A test-only run checks its saved training
    config before loading weights; a training run refuses to overwrite an
    existing checkpoint. Completed matching runs are skipped.
    """
    args = parse_args()
    config = load_config(args)
    reference = ROOT / "checkpoints/biometric" / config["name"] / "run_config.json"
    if reference.is_file():
        saved = json.loads(reference.read_text())
        if "reference_cohort_sha256" in saved:
            config["reference_cohort_sha256"] = saved["reference_cohort_sha256"]
    manifest, manifest_path = prepare_cohort(config, ROOT)
    print(f"Run: {config['name']}")
    print(f"Cohort manifest: {manifest_path}")
    print(
        f"Downstream train prefix needed (including biometric test): {manifest['downstream_train_prefix_required']}"
    )
    for group in ("train", "val", "test"):
        patients = manifest[f"{group}_patients"]
        n_sessions = sum((len(manifest["valid_sids"][pid]) for pid in patients))
        print(
            f"{group}: {len(patients)} patients, {n_sessions} sessions, {10 * n_sessions} pairs"
        )
    if args.prepare_only:
        return
    config["biometrics"]["cohort_manifest"] = str(manifest_path.relative_to(ROOT))
    action = existing_run_action(config, ROOT)
    if action == "skip":
        print(f"SKIP: checkpoint and evaluation already exist for {config['name']}")
        return
    if action == "test":
        print("Existing checkpoint: evaluation only, no retraining.")
        config["phase"] = "test"
    import torch
    from torch.utils.data import DataLoader
    from biomecg.utils import set_seed, print_config_dict, print_dataloader_info
    from biomecg.models.network import ClassificationNetwork
    from biomecg.training.biometric_trainer import ClassificationTrainingSingleBeat
    from biomecg.evaluation.biometric_test import ClassificationTestSingleBeat
    from biomecg.evaluation.biometric_analysis import ClassificationAnalyser

    set_seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print_config_dict(config)
    checkpoint_dir = ROOT / "checkpoints/biometric" / config["name"]
    if action == "train":
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "run_config.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )
    if config["phase"] == "train":
        datasets = {
            group: make_dataset(config, manifest, group) for group in ("train", "val")
        }
        dataloaders = {
            group: DataLoader(
                dataset,
                batch_size=config["training"]["batch_size"],
                shuffle=group == "train",
                num_workers=config["training"]["num_workers"],
                pin_memory=config["training"]["pin_memory"],
            )
            for group, dataset in datasets.items()
        }
        for group, loader in dataloaders.items():
            print_dataloader_info(group, loader)
        model = ClassificationNetwork(config)
        trainer = ClassificationTrainingSingleBeat(
            config, dataloaders=dataloaders, model=model
        )
        try:
            trainer.train()
            (checkpoint_dir / "training_complete.json").write_text(
                '{"complete": true}\n'
            )
        finally:
            for dataset in datasets.values():
                dataset.close()
        del trainer, model, dataloaders, datasets
    config["phase"] = "test"
    model = ClassificationNetwork(config)
    loader_name = "test_multi"
    test_dataset = make_dataset(config, manifest, "test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
    )
    print_dataloader_info(loader_name, test_loader)
    try:
        inference = ClassificationTestSingleBeat(
            config, test_loader, model, dataloader_name=loader_name
        )
        inference.evaluate()
        analyser = ClassificationAnalyser(config, loader_name=loader_name)
        analyser.analyse_model()
    finally:
        test_dataset.close()


if __name__ == "__main__":
    main()
