"""Recompute the paper's aggregate metrics from locally generated predictions.

The supplied snapshot defines which experiments belong to each figure, never
supplies their recomputed numerical values. Missing requested runs are errors.
"""

import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
)
from biomecg.paths import ROOT

LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
TWELVE = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# Each summary is rebuilt from its own per-seed numerical data.
SUMMARIES = {
    "mortality_same_dataset_late8_summary": (
        "mortality_same_dataset_late8_per_seed",
        "test_patient_auc",
    ),
    "mortality_mimic_early_late_summary": (
        "mortality_mimic_early_late_per_seed",
        "test_patient_auc",
    ),
    "mortality_full_pretraining_size_summary": (
        "mortality_full_pretraining_size_per_seed",
        "test_patient_auc",
    ),
    "mortality_full_pretraining_size_founder_summary": (
        "mortality_full_pretraining_size_founder_per_seed",
        "test_patient_auc",
    ),
    "mortality_mimic_code15_comparison_code15_summary": (
        "mortality_mimic_code15_comparison_code15_per_seed",
        "test_patient_auc",
    ),
    "demographics_complete_summary": ("demographics_complete_per_seed", "value"),
    "ecgfounder_late8_vs_late12_summary": (
        "ecgfounder_late8_vs_late12_per_seed",
        "auc",
    ),
    "complete_results_mean_sd": ("complete_results_per_seed", "value"),
    "biometric_eer_summary": ("biometric_eer_per_seed", "eer_percent"),
    "age_late_fusion_bias_summary": ("age_late_fusion_bias_per_seed", "bias"),
}
PER_SEED = sorted(
    {name for name, _ in SUMMARIES.values()}
    | {"frozen_source_comparison_mimic", "frozen_source_comparison_code15"}
)


def read_frame(path):
    return pd.read_csv(path, dtype={"cohort_label": str, "cohort": str})


class Evaluator:
    def __init__(self, results):
        self.results = Path(results)
        plan = pd.read_csv(ROOT / "configs/experiments.csv", dtype={"cohort_size": str})
        self.paths = {
            (
                r.site,
                r.task,
                r.seed,
                r.cohort_size,
                r.architecture,
                r.model,
                r.pretraining_size,
            ): r.run_path
            for r in plan.itertuples()
        }

    @lru_cache(maxsize=None)
    def measure(self, site, task, seed, cohort, architecture, fusion, size):
        models = (
            ["early_fusion"]
            if fusion == "Early"
            else TWELVE
            if fusion == "Late12"
            else LEADS
        )
        reference, predictions, counts = None, [], []
        for model in models:
            key = (site, task, seed, cohort, architecture, model, size)
            folder = self.results / self.paths[key]
            for name in (
                "best.pt",
                "metrics.csv",
                "predictions.csv",
                "predictions_patient.csv",
            ):
                if not (folder / name).is_file():
                    raise FileNotFoundError(folder / name)
            metrics = read_frame(folder / "metrics.csv")
            if len(metrics) != 1:
                raise ValueError(f"Expected exactly one metric record: {folder}")
            row = metrics.iloc[0]
            for column, value in [
                ("site", site),
                ("seed", seed),
                ("cohort_label", cohort),
                ("architecture", architecture),
                ("model", model),
            ]:
                if str(row[column]) != str(value):
                    raise ValueError(
                        f"{column} disagrees with the experiment directory: {folder}"
                    )
            counts.append(int(row.n_train_patients))
            pred = (
                pd.read_csv(folder / "predictions.csv", dtype={"pid": str})
                .sort_values("row_index")
                .reset_index(drop=True)
            )
            identity = pred[["row_index", "pid", "ecg_order", "label"]]
            if pred.pid.duplicated().any() or pred.row_index.duplicated().any():
                raise ValueError(f"Duplicate patients or ECGs: {folder}")
            if (
                len(pred) != int(row.n_test_patients)
                or not np.isfinite(pred[["label", "prediction"]].to_numpy()).all()
            ):
                raise ValueError(f"Invalid prediction count/values: {folder}")
            if reference is None:
                reference = identity
            elif not reference.equals(identity):
                raise ValueError(f"Late-fusion patient/ECG/target mismatch: {folder}")
            predictions.append(pred.prediction.to_numpy())
        if len(set(counts)) != 1:
            raise ValueError("Experts use different training cohort sizes")
        truth = reference.label.to_numpy()
        score = np.mean(predictions, axis=0)
        output = dict(
            n_train_patients=counts[0],
            n_test_ecgs=len(truth),
            n_test_patients=len(truth),
        )
        if task == "age":
            metric_truth = truth.astype(np.float32) if fusion == "Early" else truth
            metric_score = score.astype(np.float32) if fusion == "Early" else score
            output.update(
                value=mean_absolute_error(metric_truth, metric_score),
                bias=float((score - truth).mean()),
            )
        else:
            if (
                set(np.unique(truth)) != {0, 1}
                or not ((score >= 0) & (score <= 1)).all()
            ):
                raise ValueError(
                    "Binary evaluation requires two classes and probabilities in [0,1]"
                )
            auc = roc_auc_score(truth, score)
            ap = average_precision_score(truth, score)
            tn, fp, fn, tp = confusion_matrix(
                truth, score >= 0.5, labels=[0, 1]
            ).ravel()
            output.update(value=auc, auc=auc)
            for prefix in ("test_ecg", "test_patient"):
                output.update(
                    {
                        f"{prefix}_{k}": v
                        for k, v in dict(
                            auc=auc, pr_auc=ap, tn=tn, fp=fp, fn=fn, tp=tp
                        ).items()
                    }
                )
        return output

    def table(self, name, template):
        frame = template.copy()
        for index, row in frame.iterrows():
            task = (
                str(row["task"]).lower()
                if "task" in row
                else "age"
                if name.startswith("age_")
                else "mortality"
            )
            fusion = row["fusion"] if "fusion" in row else "Late"
            if (
                "model" in row
                and row["model"] == "mean_late_fusion"
                and row.architecture.startswith("ecgfounder_")
            ):
                fusion = "Late12"
            size = int(row["pretraining_size"]) if "pretraining_size" in row else 1359
            values = self.measure(
                row.site,
                task,
                int(row.seed),
                str(row.cohort_label),
                row.architecture,
                fusion,
                size,
            )
            for key, value in values.items():
                if key in frame.columns:
                    frame.loc[index, key] = value
        return frame


def recompute(results, output, biometric_results):
    source = ROOT / "artifacts/paper"
    if output.exists():
        raise FileExistsError(
            "Choose a new artifact directory; the supplied snapshot is never overwritten"
        )
    output.mkdir(parents=True)
    evaluator = Evaluator(results)
    tables = {}
    for name in PER_SEED:
        print(f"Recomputing {name}", flush=True)
        template = read_frame(source / f"{name}.csv")
        if name == "biometric_eer_per_seed":
            frame = template.copy()
            inventory = pd.read_csv(ROOT / "checkpoints/biometric_inventory.csv")
            for index, row in frame.iterrows():
                model = "early_fusion" if row.lead == "8l" else row.lead
                match = inventory[
                    (inventory.site == row.site.lower())
                    & (inventory.seed == row.seed)
                    & (inventory.pretraining_size == row.pretraining_size)
                    & (inventory.model == model)
                ]
                if len(match) != 1:
                    raise ValueError(
                        "Biometric checkpoint inventory does not match the EER recipe"
                    )
                run = Path(match.iloc[0].path).parent.name
                path = (
                    biometric_results / run / "test_multi/results_test_multi_scores.csv"
                )
                with path.open(newline="") as stream:
                    entries = [
                        r for r in csv.reader(stream) if r and r[0].strip() == "EER"
                    ]
                if len(entries) != 1:
                    raise ValueError(f"Expected one EER value: {path}")
                frame.loc[index, "eer_percent"] = 100 * float(entries[0][1])
        else:
            frame = evaluator.table(name, template)
        frame.to_csv(output / f"{name}.csv", index=False)
        tables[name] = frame
    for name, (per_seed, metric) in SUMMARIES.items():
        template = read_frame(source / f"{name}.csv")
        keys = [c for c in template if c not in ("mean", "sd", "n", "seeds")]
        groups = tables[per_seed].groupby(keys, dropna=False)
        stats = groups.agg(
            mean=(metric, "mean"), sd=(metric, "std"), n=("seed", "nunique")
        ).reset_index()
        if "seeds" in template:
            seeds = (
                groups.seed.apply(lambda x: ",".join(map(str, sorted(x))))
                .rename("seeds")
                .reset_index()
            )
            stats = stats.merge(seeds, on=keys, validate="one_to_one")
        stats.to_csv(output / f"{name}.csv", index=False)
    # Validate all experiments listed in the completed-seed table.
    completed = read_frame(source / "completed_seeds_by_experiment.csv")
    for row in completed.itertuples():
        evaluator.measure(
            row.site,
            row.task.lower(),
            row.seed,
            str(row.cohort),
            row.architecture,
            row.fusion,
            1359,
        )
    completed.to_csv(output / "completed_seeds_by_experiment.csv", index=False)
    completed.groupby(
        ["site", "task", "architecture", "fusion", "cohort"]
    ).seed.nunique().rename("n").reset_index().to_csv(
        output / "seed_counts_by_experiment.csv", index=False
    )
    full_sizes = {}
    for site in ("mimic", "code15"):
        subset = tables["mortality_same_dataset_late8_per_seed"]
        values = subset[
            (subset.site == site) & subset.cohort_label.eq("full")
        ].n_train_patients.unique()
        if len(values) != 1:
            raise ValueError("Inconsistent full training sizes")
        full_sizes[site] = int(values[0])
    (output / "cohort_sizes.json").write_text(json.dumps(full_sizes, indent=2) + "\n")
    print(
        "Metrics complete. Recompute UMAP into this directory before rendering all figures."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs/downstream")
    parser.add_argument(
        "--biometric-results-dir", type=Path, default=ROOT / "outputs/biometric"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/recomputed_artifacts"
    )
    args = parser.parse_args()
    recompute(args.results_dir, args.output, args.biometric_results_dir)


if __name__ == "__main__":
    main()
