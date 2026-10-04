"""Track training completion and input provenance."""

import hashlib
import json
from pathlib import Path


def begin_run(folder, args, split_path, initialization=()):
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["inputs"] = {
        str(p): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in (args.labels_path, args.ecg_path, split_path)
    }
    config["initialization_sha256"] = {}
    for source in initialization:
        source = Path(source)
        with source.open("rb") as stream:
            config["initialization_sha256"][str(source)] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    path = folder / "run_config.json"
    if folder.exists() and any(folder.iterdir()):
        previous = json.loads(path.read_text())
        ignored = {"device", "num_workers", "cpu_threads"}
        if {k: v for k, v in previous.items() if k not in ignored} != {
            k: v for k, v in config.items() if k not in ignored
        }:
            raise ValueError(f"Existing run has a different configuration: {folder}")
        if not (folder / "training_complete.json").is_file():
            raise FileExistsError(
                f"Interrupted run: inspect the existing artifacts before restarting {folder}"
            )
        if not (folder / "best.pt").is_file():
            raise FileNotFoundError(folder / "best.pt")
        if all(
            (folder / f).is_file() and (folder / f).stat().st_size
            for f in ("metrics.csv", "predictions.csv", "predictions_patient.csv")
        ):
            print(f"ALREADY COMPLETE: {folder}")
            return "skip"
        print(f"TRAINING COMPLETE: evaluating saved weights in {folder}")
        return "evaluate"
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")
    return "train"


def finish_training(folder):
    (folder / "training_complete.json").write_text('{"complete": true}\n')
