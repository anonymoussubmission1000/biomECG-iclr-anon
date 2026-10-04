"""Launch the explicit experiments used by the paper, one cohort/task/seed at a time."""

import argparse
import subprocess
import sys
from types import SimpleNamespace
import pandas as pd
from biomecg.paths import ROOT, data_paths


def plan(site, task, seed, study="main"):
    frame = pd.read_csv(ROOT / "configs/experiments.csv", dtype={"cohort_size": str})
    frame = frame[(frame.site == site) & (frame.task == task) & (frame.seed == seed)]
    if study == "main":
        frame = frame[frame.pretraining_size == 1359]
    else:
        frame = frame[
            (frame.task == "mortality")
            & frame.cohort_size.eq("full")
            & frame.architecture.isin(
                ["pretrained_mimic_frozen", "pretrained_mimic_finetuned"]
            )
        ]
    if frame.empty:
        raise ValueError(
            "No paper experiments match the requested site/task/seed/study"
        )
    return frame


def command(run, args):
    labels, ecg = data_paths(run.site)
    founder = run.architecture.startswith("ecgfounder_")
    module = (
        "demographics"
        if run.task != "mortality"
        else "mortality_founder"
        if founder
        else "mortality_resnet"
    )
    architecture = (
        run.architecture.replace("ecgfounder_", "pretrained_")
        if founder and run.task == "mortality"
        else run.architecture
    )
    cmd = [
        sys.executable,
        "-m",
        f"biomecg.training.{module}",
        "--site",
        run.site,
        "--seed",
        str(run.seed),
        "--labels_path",
        str(labels),
        "--ecg_path",
        str(ecg),
        "--cohort_size",
        run.cohort_size,
        "--architecture",
        architecture,
        "--model",
        run.model,
        "--lr_encoder",
        "0" if run.architecture.endswith("_frozen") else "0.0001",
        "--lr_head",
        "0.001",
        "--bs",
        str(args.batch_size),
        "--num_workers",
        str(args.workers),
        "--cpu_threads",
        str(args.cpu_threads),
        "--device",
        args.device,
    ]
    if run.task == "mortality":
        cmd += ["--horizon", "365"]
        if not founder:
            cmd += ["--mimic_pretraining_size", str(run.pretraining_size)]
    else:
        cmd += ["--task", run.task]
    return cmd


def validate(frame, args):
    from biomecg.training import mortality_resnet as resnet

    labels_path, ecg_path = data_paths(args.site)
    labels = resnet.load_labels(labels_path, ecg_path, 365)
    split = (
        ROOT
        / "outputs/downstream"
        / args.site
        / f"seed{args.seed}/splits/horizon365d/split.npz"
    )
    groups = resnet.load_cached_split(split, labels)
    for run in frame.itertuples():
        size = len(groups[0]) if run.cohort_size == "full" else int(run.cohort_size)
        if size > len(groups[0]):
            raise ValueError(f"Requested cohort {size} exceeds the training population")
    sources = frame.drop_duplicates(["architecture", "model", "pretraining_size"])
    for run in sources.itertuples():
        if run.architecture.startswith("ecgfounder_"):
            n = 12 if run.model == "early_fusion" else 1
            path = ROOT / "checkpoints/ecgfounder" / f"{n}_lead_ECGFounder.pth"
            if not path.is_file():
                raise FileNotFoundError(path)
        elif run.architecture in resnet.BIOMETRIC_ARCHITECTURES:
            settings = SimpleNamespace(
                **run._asdict(),
                horizon=365,
                labels_path=labels_path,
                ecg_path=ecg_path,
                mimic_pretraining_size=run.pretraining_size,
                force_rebuild_split=False,
            )
            resnet.resolve_biometric_pretraining(settings)
    if args.task in ("age", "sex"):
        from biomecg.training import demographics

        settings = SimpleNamespace(
            site=args.site,
            seed=args.seed,
            task=args.task,
            horizon=365,
            labels_path=labels_path,
            ecg_path=ecg_path,
        )
        demographics.load_data(settings)
    print(f"Validated {len(frame)} experiments")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=["mimic", "code15"], required=True)
    parser.add_argument(
        "--task", choices=["mortality", "age", "sex"], default="mortality"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--study", choices=["main", "scaling"], default="main")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--list",
        action="store_true",
        help="Display the exact run list without requiring data",
    )
    actions.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare mortality split before pretraining",
    )
    args = parser.parse_args()
    if args.prepare_only:
        labels, ecg = data_paths(args.site)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "biomecg.training.mortality_resnet",
                "--site",
                args.site,
                "--labels_path",
                str(labels),
                "--ecg_path",
                str(ecg),
                "--horizon",
                "365",
                "--seed",
                str(args.seed),
                "--prepare_only",
            ],
            cwd=ROOT,
            check=True,
        )
        return
    frame = plan(args.site, args.task, args.seed, args.study)
    if args.list:
        print(frame.drop(columns="run_path").to_string(index=False))
        print(f"{len(frame)} runs")
        return
    validate(frame, args)
    for run in frame.itertuples():
        print(f"RUN {run.run_path}", flush=True)
        subprocess.run(command(run, args), cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
