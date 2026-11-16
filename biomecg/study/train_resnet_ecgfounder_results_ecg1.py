"""Plot completed ecg1 mortality experiments, including incremental training runs.

--site code15 selects only CODE-15: six variants, 1000/8000/full, matched eight-lead
late fusion. Uses the same metrics, paired comparisons and sparse-run semantics
as MIMIC. Default --site mimic preserves historical outputs and pretraining plots.

Examples (no training is launched):
    python biomecg/study/train_resnet_ecgfounder_results_ecg1.py --seed 23451 --metric all
    python biomecg/study/train_resnet_ecgfounder_results_ecg1.py --all_seeds --metric all

Keep the historical overview and CSV paths, and add focused comparisons under
aggregate/plots/comparisons/<metric> (PNG, PDF, summary CSV and per-seed CSV). Missing
experts defer only that late-fusion point; fusion_availability.csv lists native and matched-8 availability.
Without --seed, all available seeds are aggregated. The default metric remains
test_patient_auc. --metric all generates both
AUROC and average precision; --all_seeds includes available seeds without
requiring every training run to finish. Differences use matched seeds only.
Main comparisons include scratch, excluded source, MIMIC-1359 and ECGFounder. Other MIMIC
pretraining sizes are used exclusively in aggregate/plots/mortality_pretraining_size.
Focused comparisons live in aggregate/plots/comparisons; --seed scopes them to
aggregate/plots/seed<N>/comparisons without replacing the all-seed comparisons.
"""

# Workflow:
# 1. Load only the results from the 365-day horizon.
# 2. Validate completed metrics and prediction files; report unfinished groups.
# 3. Compute native mean late fusion (8 ResNet / 12 ECGFounder) plus matched 8-lead ECGFounder fusion.
# 4. Remove individual single-lead models from the displayed results.
# 5. Keep the original panels and add focused, paired transfer comparisons.
# 6. Save figures for each seed and mean ± SD figures across seeds.

import argparse
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

if __package__:
    from .plot_transfer_comparisons_ecg1 import make_comparison_plots, make_code15_comparison_plots, metric_label
else:
    from plot_transfer_comparisons_ecg1 import make_comparison_plots, make_code15_comparison_plots, metric_label

ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "outputs/downstream"
HORIZON_DAYS = 365

RESNET_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
ECGFOUNDER_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEADS_BY_ARCHITECTURE = {"scratch": RESNET_LEADS, "pretrained_frozen": RESNET_LEADS, "pretrained_finetuned": RESNET_LEADS, "ecgfounder_frozen": ECGFOUNDER_LEADS, "ecgfounder_finetuned": ECGFOUNDER_LEADS}
LEADS_BY_ARCHITECTURE.update({
    "pretrained_mimic_frozen": RESNET_LEADS,
    "pretrained_mimic_finetuned": RESNET_LEADS,
    "pretrained_code15_frozen": RESNET_LEADS,
    "pretrained_code15_finetuned": RESNET_LEADS,
})
PLOT_MODELS = ["early_fusion", "mean_late_fusion", "mean_late_fusion_8_leads"]

# Fair late-fusion comparison: ECGFounder is also evaluated on the exact eight
# leads available to the ResNet (I, II, V1-V6). ResNet already uses these eight
# leads, so we do not duplicate its identical late-fusion point.
ECGFOUNDER_MATCHED_8_LEADS = RESNET_LEADS
ARCH_ORDER = ["scratch", "pretrained_frozen", "pretrained_finetuned", "ecgfounder_frozen", "ecgfounder_finetuned"]
ARCH_ORDER += ["pretrained_mimic_frozen", "pretrained_mimic_finetuned", "pretrained_code15_frozen", "pretrained_code15_finetuned"]

ARCH_LABELS = {
    "pretrained_code15_frozen": "CODE-15 pretrained frozen",
    "pretrained_code15_finetuned": "CODE-15 pretrained fine-tuned",
    "scratch": "Scratch",
    "pretrained_mimic_frozen": "MIMIC pretrained frozen",
    "pretrained_mimic_finetuned": "MIMIC pretrained fine-tuned",
    "pretrained_frozen": "excluded source pretrained frozen",
    "pretrained_finetuned": "excluded source pretrained fine-tuned",
    "ecgfounder_frozen": "ECGFounder frozen",
    "ecgfounder_finetuned": "ECGFounder fine-tuned",
}
ARCH_COLORS = {
    "pretrained_code15_frozen": "#009E73",
    "pretrained_code15_finetuned": "#9467BD",
    "scratch": "#4C78A8",
    "pretrained_mimic_frozen": "#937860",
    "pretrained_mimic_finetuned": "#DA8BC3",
    "pretrained_frozen": "#F58518",
    "pretrained_finetuned": "#54A24B",
    "ecgfounder_frozen": "#B279A2",
    "ecgfounder_finetuned": "#E45756",
}
FUSION_LABELS = {
    "early_fusion": "Early fusion",
    "mean_late_fusion": "Mean late fusion (all available leads)",
    "mean_late_fusion_8_leads": "Mean late fusion (8 matched leads)",
}
FUSION_MARKERS = {
    "early_fusion": "o",
    "mean_late_fusion": "s",
    "mean_late_fusion_8_leads": "D",
}
FUSION_LINESTYLES = {
    "early_fusion": "-",
    "mean_late_fusion": "--",
    "mean_late_fusion_8_leads": ":",
}


# Native models added from https://github.com/AI4HealthUOL/ecg-fm-benchmarking.
# No artificial early/late variants for these pretrained encoders.
FOUNDATION_NAMES = {"ecgcpc":"ECG-CPC","ecgjepa":"ECG-JEPA","merl":"MERL","stmem":"ST-MEM"}
FOUNDATION_ARCHITECTURES = {f"{family}_{mode}" for family in FOUNDATION_NAMES for mode in ("frozen","finetuned")}
for i, (family, label) in enumerate(FOUNDATION_NAMES.items()):
    for mode in ("frozen","finetuned"):
        architecture = f"{family}_{mode}"
        ARCH_ORDER.append(architecture)
        ARCH_LABELS[architecture] = label + (" frozen" if mode == "frozen" else " fine-tuned")
        ARCH_COLORS[architecture] = ["#0072B2","#D55E00","#009E73","#CC79A7"][i]
        LEADS_BY_ARCHITECTURE[architecture] = []
CPC_LOCAL_ARCHITECTURES = {f'ecgcpc_local_{site}_{mode}' for site in ('mimic','code15') for mode in ('frozen','finetuned')}
FOUNDATION_ARCHITECTURES.update(CPC_LOCAL_ARCHITECTURES)
for architecture in sorted(CPC_LOCAL_ARCHITECTURES):
    ARCH_ORDER.append(architecture)
    ARCH_LABELS[architecture] = architecture.replace('ecgcpc_local_', 'ECG-CPC local ').replace('_', ' ')
    ARCH_COLORS[architecture] = '#56B4E9'
    LEADS_BY_ARCHITECTURE[architecture] = []
PCLR_ARCHITECTURES = [f'pclr_{site}_{mode}' for site in ('mimic','code15') for mode in ('frozen','finetuned')]
for architecture in PCLR_ARCHITECTURES:
    ARCH_ORDER.append(architecture)
    ARCH_LABELS[architecture] = architecture.replace('pclr_', 'PCLR ').replace('_', ' ')
    ARCH_COLORS[architecture] = '#009E73'
    LEADS_BY_ARCHITECTURE[architecture] = []
JEPA_RESNET_ARCHITECTURES = [f'{prefix}_{site}_frozen' for prefix in ('jepa_resnet', 'jepa_resnet_latebudget', 'jepa_resnet_early') for site in ('mimic','code15')]
for architecture in JEPA_RESNET_ARCHITECTURES:
    ARCH_ORDER.append(architecture)
    ARCH_LABELS[architecture] = architecture.replace('jepa_resnet_', 'JEPA-ResNet ').replace('_', ' ')
    ARCH_COLORS[architecture] = '#E69F00'
    LEADS_BY_ARCHITECTURE[architecture] = []
PLOT_MODELS.append("native")
FUSION_LABELS["native"] = "Native multilead"
FUSION_MARKERS["native"] = "^"
FUSION_LINESTYLES["native"] = "-"


EXPERIMENT_KEYS = [
    "site",
    "seed",
    "horizon_days",
    "cohort_label",
    "architecture",
    "model",
]
METRICS_COLUMNS = [*EXPERIMENT_KEYS, "n_test_ecgs", "n_test_patients"]
PREDICTION_COLUMNS = ["row_index", "pid", "ecg_order", "label", "prediction"]


def slug(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def calculate_metrics(labels, scores, prefix):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError(f"{prefix}: labels and scores must be one-dimensional")
    if len(labels) == 0 or len(labels) != len(scores):
        raise ValueError(f"{prefix}: labels and scores have incompatible lengths")
    if not np.isfinite(labels).all() or not np.isfinite(scores).all():
        raise ValueError(f"{prefix}: labels or scores contain non-finite values")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{prefix}: both binary classes are required")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError(f"{prefix}: predictions must be probabilities in [0, 1]")

    matrix = confusion_matrix(labels, scores >= 0.5, labels=[0, 1])
    return {
        f"{prefix}_auc": roc_auc_score(labels, scores),
        f"{prefix}_pr_auc": average_precision_score(labels, scores),
        f"{prefix}_tn": int(matrix[0, 0]),
        f"{prefix}_fp": int(matrix[0, 1]),
        f"{prefix}_fn": int(matrix[1, 0]),
        f"{prefix}_tp": int(matrix[1, 1]),
    }


def load_metrics(seed=None, pretraining_size=1359, site=None):
    pattern = (
        f"*/seed*/runs/horizon{HORIZON_DAYS}d/"
        + ("cohort*/*/*/metrics.csv" if pretraining_size == 1359
           else f"cohort*/pretraining{pretraining_size}/*/*/metrics.csv")
    )
    paths = sorted(RESULTS_ROOT.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No metrics.csv files found for horizon {HORIZON_DAYS} days "
            f"under {RESULTS_ROOT}"
        )

    frames = []
    for path in paths:
        if site is not None and path.relative_to(RESULTS_ROOT).parts[0] != site:
            continue
        if seed is not None and f"seed{seed}" not in path.parts:
            continue
        if path.parent.name == "native" and not (path.parent / "completed.json").is_file():
            continue
        if path.parent.parent.name in set(PCLR_ARCHITECTURES) | CPC_LOCAL_ARCHITECTURES and not (path.parent / "completed.json").is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            warnings.warn(f"Metrics still being written; skipped: {path}")
            continue
        if frame.empty:
            warnings.warn(f"Metrics still being written; skipped: {path}")
            continue
        if len(frame) != 1:
            raise ValueError(
                f"Expected exactly one metrics row in {path}, found {len(frame)}"
            )

        missing = [column for column in METRICS_COLUMNS if column not in frame.columns]
        if missing:
            raise KeyError(f"Missing columns in {path}: {missing}")

        frame["source_path"] = str(path)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No completed metrics found for seed={seed}")
    metrics = pd.concat(frames, ignore_index=True, sort=False)
    if seed is not None:
        metrics = metrics[metrics["seed"] == seed].copy()
        if metrics.empty:
            raise ValueError(f"No metrics.csv files found for seed{seed}")
    invalid_horizons = metrics["horizon_days"] != HORIZON_DAYS
    if invalid_horizons.any():
        invalid_paths = metrics.loc[invalid_horizons, "source_path"].tolist()
        raise ValueError(f"Files contain results from another horizon: {invalid_paths}")

    metrics["model"] = metrics["model"].replace(
        {"early_fusion_8_leads": "early_fusion"}
    )
    metrics["cohort_label"] = metrics["cohort_label"].astype(str)
    metrics["cohort_label"] = metrics["cohort_label"].replace(
        {"nan": "full", "None": "full"}
    )

    duplicates = metrics.duplicated(EXPERIMENT_KEYS, keep=False)
    if duplicates.any():
        columns = [*EXPERIMENT_KEYS, "source_path"]
        details = metrics.loc[duplicates, columns].to_string(index=False)
        raise ValueError(f"Duplicate experiments found:\n{details}")

    valid_models = set([*RESNET_LEADS, *ECGFOUNDER_LEADS, "early_fusion", "native"])
    invalid_models = sorted(set(metrics["model"]) - valid_models)
    if invalid_models:
        raise ValueError(f"Unknown models in metrics files: {invalid_models}")

    invalid_architectures = sorted(set(metrics["architecture"]) - set(ARCH_ORDER))
    if invalid_architectures:
        raise ValueError(
            f"Unknown architectures in metrics files: {invalid_architectures}"
        )
    return metrics


def load_predictions(metrics_row, lead):
    metrics_path = Path(metrics_row["source_path"])
    path = metrics_path.parent / "predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {lead} predictions: {path}")

    frame = pd.read_csv(path)
    missing = [column for column in PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")
    frame = frame[PREDICTION_COLUMNS].copy()

    if len(frame) != int(metrics_row["n_test_ecgs"]):
        raise ValueError(
            f"Prediction count differs from n_test_ecgs in {path}: "
            f"{len(frame)} != {int(metrics_row['n_test_ecgs'])}"
        )
    if frame["row_index"].duplicated().any():
        raise ValueError(f"Duplicate row_index values in {path}")
    if frame["pid"].duplicated().any():
        raise ValueError(f"More than one ECG per patient in {path}")
    if not np.isfinite(frame["prediction"]).all():
        raise ValueError(f"Non-finite predictions in {path}")
    if np.any((frame["prediction"] < 0) | (frame["prediction"] > 1)):
        raise ValueError(f"Predictions outside [0, 1] in {path}")

    return frame.rename(columns={"prediction": f"prediction_{lead}"})


def fusion_availability(metrics):
    """Report completed experts for full and matched 8-lead late fusion."""
    keys = ["site", "seed", "horizon_days", "cohort_label", "architecture"]
    rows = []
    for values, group in metrics.groupby(keys, dropna=False):
        architecture = values[-1]
        if architecture in FOUNDATION_ARCHITECTURES or architecture in PCLR_ARCHITECTURES:
            continue
        leads = (RESNET_LEADS if values[0] == "code15" else LEADS_BY_ARCHITECTURE[architecture])
        present = set(group["model"])
        missing = [lead for lead in leads if lead not in present]
        missing_files = [
            row["model"] for _, row in group.iterrows()
            if row["model"] in leads
            and not (Path(row["source_path"]).parent / "predictions.csv").is_file()
        ]

        matched_leads = ECGFOUNDER_MATCHED_8_LEADS if architecture.startswith("ecgfounder_") else leads
        matched_missing = [lead for lead in matched_leads if lead not in present]
        matched_missing_files = [
            row["model"] for _, row in group.iterrows()
            if row["model"] in matched_leads
            and not (Path(row["source_path"]).parent / "predictions.csv").is_file()
        ]

        rows.append({
            **dict(zip(keys, values)),
            "early_available": "early_fusion" in present,
            "available_leads": len(set(leads) & present),
            "required_leads": len(leads),
            "missing_leads": ",".join(missing),
            "missing_prediction_files": ",".join(missing_files),
            "late_available": not missing and not missing_files,
            "matched_8_required_leads": len(matched_leads),
            "matched_8_available_leads": len(set(matched_leads) & present),
            "matched_8_missing_leads": ",".join(matched_missing),
            "matched_8_missing_prediction_files": ",".join(matched_missing_files),
            "matched_8_late_available": not matched_missing and not matched_missing_files,
        })
    return pd.DataFrame(rows)


def _late_fusion_rows_for_spec(metrics, model_name, leads_for_architecture, strict=False):
    """Compute one late-fusion definition from completed single-lead predictions."""
    group_keys = ["site", "seed", "horizon_days", "cohort_label", "architecture"]
    single_metrics = metrics[metrics["model"].isin(set([*RESNET_LEADS, *ECGFOUNDER_LEADS]))]
    rows = []

    for values, group in single_metrics.groupby(group_keys, dropna=False):
        architecture = values[group_keys.index("architecture")]
        leads = leads_for_architecture(architecture)
        if leads is None:
            continue

        found_leads = set(group["model"])
        missing = sorted(set(leads) - found_leads)
        if missing:
            if strict:
                raise ValueError(f"Missing single leads {missing} for experiment {values} ({model_name})")
            continue

        relevant = group[group["model"].isin(leads)]
        missing_files = [
            row["model"] for _, row in relevant.iterrows()
            if not (Path(row["source_path"]).parent / "predictions.csv").is_file()
        ]
        if missing_files:
            if strict:
                raise FileNotFoundError(
                    f"Missing predictions for {missing_files}: {values} ({model_name})"
                )
            continue

        merged = None
        for lead in leads:
            lead_rows = group[group["model"] == lead]
            if len(lead_rows) != 1:
                raise ValueError(f"Expected one {lead} result for experiment {values}")

            predictions = load_predictions(lead_rows.iloc[0], lead)
            if merged is None:
                merged = predictions
            else:
                merged = merged.merge(
                    predictions,
                    on=["row_index", "pid", "ecg_order", "label"],
                    how="inner",
                    validate="one_to_one",
                )

        expected_counts = relevant["n_test_ecgs"].astype(int).unique()
        if len(expected_counts) != 1:
            raise ValueError(
                f"Inconsistent n_test_ecgs across experts for {values} ({model_name}): "
                f"{expected_counts.tolist()}"
            )
        expected_count = int(expected_counts[0])
        if len(merged) != expected_count:
            raise ValueError(
                f"Late-fusion prediction mismatch for {values} ({model_name}): "
                f"{len(merged)} matched rows instead of {expected_count}"
            )
        if merged["pid"].duplicated().any():
            raise ValueError(
                f"Late fusion contains multiple ECGs per patient for {values} ({model_name})"
            )

        prediction_columns = [f"prediction_{lead}" for lead in leads]
        merged["prediction"] = merged[prediction_columns].mean(axis=1)
        row = dict(zip(group_keys, values))
        row.update({
            "model": model_name,
            "patient_reference_ecg": "first_min_ECG_Order",
            "n_test_ecgs": len(merged),
            "n_test_patients": len(merged),
            "source_path": f"computed from {len(leads)} single-lead predictions",
            "n_fusion_leads": len(leads),
            "fusion_leads": ",".join(leads),
        })
        row.update(
            calculate_metrics(
                merged["label"].to_numpy(),
                merged["prediction"].to_numpy(),
                "test_ecg",
            )
        )
        row.update(
            calculate_metrics(
                merged["label"].to_numpy(),
                merged["prediction"].to_numpy(),
                "test_patient",
            )
        )
        rows.append(row)

    return pd.DataFrame(rows)


def late_fusion_rows(metrics, strict=False):
    """Compute native late fusion plus a fair 8-lead ECGFounder variant.

    Native late fusion uses all architecture-specific experts: 8 leads for ResNet
    and 12 for ECGFounder. The additional ECGFounder row uses exactly I, II,
    V1-V6, matching the ResNet late-fusion input set.
    """
    # CODE-15 Founder has only the matched eight experts, never a native 12-lead run.
    native_metrics = metrics[~(metrics.site.eq("code15") & metrics.architecture.str.startswith("ecgfounder_"))]
    native = _late_fusion_rows_for_spec(
        native_metrics,
        model_name="mean_late_fusion",
        leads_for_architecture=lambda architecture: LEADS_BY_ARCHITECTURE[architecture],
        strict=strict,
    )
    matched_8 = _late_fusion_rows_for_spec(
        metrics,
        model_name="mean_late_fusion_8_leads",
        leads_for_architecture=lambda architecture: (
            ECGFOUNDER_MATCHED_8_LEADS if architecture.startswith("ecgfounder_") else None
        ),
        strict=strict,
    )
    return pd.concat([native, matched_8], ignore_index=True, sort=False)


def cohort_key(value):
    text = str(value)
    if text == "full":
        return 1, np.inf
    return 0, int(float(text))


def plot_models_for_architecture(frame, architecture):
    """Return only fusion strategies that actually exist for this architecture."""
    present = set(frame.loc[frame["architecture"] == architecture, "model"])
    return [model for model in PLOT_MODELS if model in present]


def add_curves(axis, frame, cohorts, metric, aggregate, architectures):
    x_values = np.arange(len(cohorts))
    value_column = "mean" if aggregate else metric

    for architecture in architectures:
        for model in plot_models_for_architecture(frame, architecture):
            model_frame = frame[
                (frame["architecture"] == architecture)
                & (frame["model"] == model)
            ]
            model_frame = model_frame.set_index("cohort_label").reindex(cohorts)
            values = model_frame[value_column].to_numpy(dtype=float)

            axis.plot(
                x_values,
                values,
                color=ARCH_COLORS[architecture],
                marker=FUSION_MARKERS[model],
                linestyle=FUSION_LINESTYLES[model],
                linewidth=2.1,
                markersize=7,
            )

            if aggregate:
                deviations = model_frame["std"].fillna(0).to_numpy(dtype=float)
                axis.fill_between(
                    x_values,
                    values - deviations,
                    values + deviations,
                    color=ARCH_COLORS[architecture],
                    alpha=0.10,
                )

    axis.set_ylabel(metric_label(metric))
    axis.grid(alpha=0.25)


def add_legends(axis, frame, architectures=ARCH_ORDER):
    architecture_handles = [
        Line2D(
            [0],
            [0],
            color=ARCH_COLORS[architecture],
            linewidth=2.5,
            label=ARCH_LABELS[architecture],
        )
        for architecture in architectures
    ]
    present_models = set(frame["model"])
    fusion_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker=FUSION_MARKERS[model],
            linestyle=FUSION_LINESTYLES[model],
            linewidth=2,
            markersize=7,
            label=FUSION_LABELS[model],
        )
        for model in PLOT_MODELS
        if model in present_models
    ]

    architecture_legend = axis.legend(
        handles=architecture_handles,
        title="Training strategy",
        loc="lower right",
    )
    axis.add_artist(architecture_legend)
    axis.legend(
        handles=fusion_handles,
        title="Fusion strategy",
        loc="lower left",
    )


def plot_panels(frame, output_dir, metric, title, stem, aggregate=False):
    # Old result sets retain their original panels/legends until MIMIC runs exist.
    architectures = [a for a in ARCH_ORDER if a in set(frame["architecture"])]
    output_dir.mkdir(parents=True, exist_ok=True)
    cohorts = sorted(frame["cohort_label"].astype(str).unique(), key=cohort_key)
    x_values = np.arange(len(cohorts))

    figure, axes = plt.subplots(
        len(architectures) + 1,
        1,
        figsize=(15, 5 + 4 * len(architectures)),
        sharex=True,
        sharey=True,
        gridspec_kw={"height_ratios": [1.35, *([1] * len(architectures))]},
    )

    add_curves(
        axes[0],
        frame,
        cohorts,
        metric,
        aggregate,
        architectures,
    )
    axes[0].set_title(
        "Combined comparison of training and fusion strategies (including matched 8-lead ECGFounder)",
        loc="left",
    )
    add_legends(axes[0], frame, architectures)

    for axis, architecture in zip(axes[1:], architectures):
        add_curves(
            axis,
            frame,
            cohorts,
            metric,
            aggregate,
            [architecture],
        )
        axis.set_title(ARCH_LABELS[architecture], loc="left")

        fusion_handles = [
            Line2D(
                [0],
                [0],
                color=ARCH_COLORS[architecture],
                marker=FUSION_MARKERS[model],
                linestyle=FUSION_LINESTYLES[model],
                linewidth=2,
                markersize=7,
                label=FUSION_LABELS[model],
            )
            for model in plot_models_for_architecture(frame, architecture)
        ]
        axis.legend(handles=fusion_handles, loc="lower right")

    axes[-1].set_xticks(x_values)
    axes[-1].set_xticklabels(
        [f"N={cohort}" if cohort != "full" else "Full cohort" for cohort in cohorts]
    )
    axes[-1].set_xlabel("Training cohort size (patients)")

    figure.suptitle(title, fontsize=17)
    figure.tight_layout(rect=[0, 0, 1, 0.98])
    figure.savefig(
        output_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)
    frame.to_csv(output_dir / f"{stem}.csv", index=False)


def make_plots(results, metric, write_legacy=True, pretraining_size=1359, seed=None):
    code15 = results.loc[results.site.eq("code15")].copy()
    if not code15.empty:
        for (site, run_seed, horizon), frame in code15.groupby(["site", "seed", "horizon_days"]):
            title = f"CODE-15 — {int(horizon)}-day mortality (seed {int(run_seed)})"
            make_code15_comparison_plots(frame, metric, RESULTS_ROOT / site / f"seed{int(run_seed)}" / "plots", title)
        folder = RESULTS_ROOT / "code15" / "aggregate" / "plots"
        if seed is not None:
            folder /= f"seed{seed}"
        make_code15_comparison_plots(code15, metric, folder, "CODE-15 — 365-day mortality")
    results = results.loc[~results.site.eq("code15")].copy()
    if results.empty:
        return
    selected_seed = seed
    if pretraining_size != 1359:
        raise ValueError("Other MIMIC pretraining sizes belong only in figure 09.")
    # Hide the redundant 100,000-patient downstream point in all displays.
    # Keep it in the complete fusion_metrics.csv export; "full" stays visible.
    results = results.loc[
        pd.to_numeric(results["cohort_label"], errors="coerce") != 100000
    ].copy()
    if results.empty:
        return
    for values, frame in results.groupby(["site", "seed", "horizon_days"]):
        site, seed, horizon = values
        output_dir = RESULTS_ROOT / slug(site) / f"seed{int(seed)}" / "plots"
        if pretraining_size != 1359:
            output_dir /= f"pretraining{pretraining_size}"
        title = (
            f"{site.upper()} — {int(horizon)}-day mortality prediction "
            f"(seed {int(seed)})"
        )
        stem = f"resnet_ecgfounder_{slug(site)}_{int(horizon)}d_seed{int(seed)}_{metric}"
        plot_panels(
            frame,
            output_dir,
            metric,
            title,
            stem,
            aggregate=False,
        )


    aggregate = (
        results.groupby(
            ["site", "horizon_days", "cohort_label", "architecture", "model"],
            as_index=False,
        )[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregate_dir = RESULTS_ROOT / "aggregate" / "resnet_ecgfounder"
    if pretraining_size != 1359:
        aggregate_dir /= f"pretraining{pretraining_size}"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregate["metric"] = metric
    aggregate.to_csv(aggregate_dir / f"fusion_metrics_{metric}_mean_std.csv", index=False)
    if write_legacy:
        aggregate.to_csv(aggregate_dir / "fusion_metrics_mean_std.csv", index=False)

    for values, frame in aggregate.groupby(["site", "horizon_days"]):
        site, horizon = values
        output_dir = RESULTS_ROOT / slug(site) / "aggregate" / "plots" / "resnet_ecgfounder"
        if pretraining_size != 1359:
            output_dir /= f"pretraining{pretraining_size}"
        n_seeds = results.loc[(results["site"] == site) & (results["horizon_days"] == horizon), "seed"].nunique()
        title = (f"{site.upper()} — {int(horizon)}-day mortality prediction "
                 f"({n_seeds} available seed(s); per-point counts vary)")
        stem = f"resnet_ecgfounder_{slug(site)}_{int(horizon)}d_{metric}_mean_sd"
        plot_panels(
            frame,
            output_dir,
            metric,
            title,
            stem,
            aggregate=True,
        )
        raw = results[(results["site"] == site) & (results["horizon_days"] == horizon)]
        comparison_root = RESULTS_ROOT / slug(site) / "aggregate" / "plots"
        if selected_seed is not None:
            comparison_root /= f"seed{selected_seed}"
        make_comparison_plots(raw, metric, comparison_root, title)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=["mimic", "code15"], default="mimic")
    parser.add_argument(
        "--metric",
        choices=[
            "all",
            "test_patient_auc",
            "test_patient_pr_auc",
            "test_ecg_auc",
            "test_ecg_pr_auc",
        ],
        default="test_patient_auc",
    )
    parser.add_argument("--mimic_pretraining_size", type=int, choices=[1359], default=1359,
                        help="Main comparisons use MIMIC pretraining on 1359 patients; other sizes appear only in figure 09")
    parser.add_argument("--pretraining_size_plot", action="store_true",
                        help="Generate only figure 09: mortality full AUROC/AP versus pretraining size")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--seed", type=int, default=None,
                           help="Restrict plots to one seed; omitted = all available seeds.")
    selection.add_argument("--all_seeds", action="store_true",
                           help="Aggregate available runs across all seeds; counts may differ by point.")
    return parser.parse_args()


def plot_pretraining_size(frame, folder, founder=None):
    """Mortality full, eight-lead late fusion; average metrics across seeds.

    'Precision' means average precision (the existing test_patient_pr_auc),
    not precision at a chosen threshold. Missing sizes break curves; SD is
    shown only when multiple seeds are available, with n displayed per point.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = '09_mortality_full_pretraining_size'
    sizes = [100, 500, 1359, 8000, 16000, 50000]
    metrics = [('test_patient_auc', 'AUROC'),
               ('test_patient_pr_auc', 'Average precision (AP)')]
    keys = ['site', 'seed', 'pretraining_size', 'architecture']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate seed/size/strategy in pretraining-size figure')
    frame.to_csv(folder / f'{stem}_per_seed.csv', index=False)
    founder = pd.DataFrame() if founder is None else founder
    if not founder.empty and founder.duplicated(['site', 'seed', 'architecture']).any():
        raise ValueError('Duplicate ECGFounder reference seed/strategy')
    founder.to_csv(folder / f'{stem}_ecgfounder_per_seed.csv', index=False)
    reference_summaries = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    summaries = []
    for ax, (metric, label) in zip(axes, metrics):
        summary = frame.groupby(['pretraining_size', 'architecture'])[metric].agg(
            ['mean', 'std', 'count']).reset_index()
        summary['metric'] = metric
        summaries.append(summary)
        for architecture, legend, color in [
            ('pretrained_mimic_frozen', 'Frozen', '#0072B2'),
            ('pretrained_mimic_finetuned', 'Fine-tuned', '#D55E00'),
        ]:
            values = summary[summary.architecture == architecture].set_index('pretraining_size').reindex(sizes)
            ax.plot(sizes, values['mean'], 'o-', label=legend, color=color)
            ax.fill_between(sizes, values['mean'] - values['std'],
                            values['mean'] + values['std'], color=color, alpha=.15)
            for size, row in values.iterrows():
                if np.isfinite(row['mean']):
                    ax.annotate(f'n={int(row["count"])}', (size, row['mean']),
                                xytext=(0, 7 if legend == 'Frozen' else -13),
                                textcoords='offset points', ha='center', fontsize=8)
        if not founder.empty:
            reference = founder.groupby('architecture')[metric].agg(['mean', 'std', 'count']).reset_index()
            reference['metric'] = metric
            reference_summaries.append(reference)
            for architecture, legend, color, style in [
                ('ecgfounder_frozen', 'ECGFounder frozen', '#009E73', '--'),
                ('ecgfounder_finetuned', 'ECGFounder fine-tuned', '#CC79A7', ':'),
            ]:
                selected = reference[reference.architecture == architecture]
                if not selected.empty:
                    row = selected.iloc[0]
                    ax.axhline(row['mean'], color=color, linestyle=style, linewidth=1.8,
                               label=f'{legend} (n={int(row["count"])})')
        ax.set_xscale('log')
        ax.set_xticks(sizes, ['100', '500', '1 359', '8 000', '16 000', '50 000'])
        ax.minorticks_off()
        ax.set_xlabel('MIMIC pretraining patients (train + validation)')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('365-day mortality — downstream full — late fusion (8 matched leads)')
    fig.text(.5, .015, 'Available seeds per point; bands: ±1 SD (not CI). X axis: log scale.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    for ext in ('png', 'pdf'):
        fig.savefig(folder / f'{stem}.{ext}', dpi=180)
    plt.close(fig)
    pd.concat(summaries, ignore_index=True).to_csv(folder / f'{stem}_mean_sd.csv', index=False)
    references = (pd.concat(reference_summaries, ignore_index=True) if reference_summaries
                  else pd.DataFrame(columns=['architecture', 'mean', 'std', 'count', 'metric']))
    references.to_csv(folder / f'{stem}_ecgfounder_mean_sd.csv', index=False)


def mortality_pretraining_size_results(seed=None):
    """Read actual mortality predictions, fusing each source size separately.

    The 1359 results use historical paths; other sizes use pretraining<N> paths.
    No demographics outcomes enter this analysis and no existing plot is replaced.
    """
    frames, reports = [], []
    founder = pd.DataFrame()
    for size in (100, 500, 1359, 8000, 16000, 50000):
        try:
            raw = load_metrics(seed=seed, pretraining_size=size, site="mimic")
        except FileNotFoundError:
            continue
        if size == 1359:
            founder_raw = raw[(raw.cohort_label == 'full') &
                              raw.architecture.isin(('ecgfounder_frozen', 'ecgfounder_finetuned'))]
            if not founder_raw.empty:
                founder = late_fusion_rows(founder_raw)
                if not founder.empty:
                    founder = founder[founder.model == 'mean_late_fusion_8_leads']
        raw = raw[(raw.cohort_label == 'full') &
                  raw.architecture.isin(("pretrained_mimic_frozen", "pretrained_mimic_finetuned")) &
                  raw.model.isin(RESNET_LEADS)].copy()
        if raw.empty:
            continue
        report = fusion_availability(raw)
        report['pretraining_size'] = size
        reports.append(report)
        fused = late_fusion_rows(raw)
        if not fused.empty:
            fused['pretraining_size'] = size
            frames.append(fused)
    if not reports:
        print('No mortality full MIMIC runs available yet.')
        return
    availability = pd.concat(reports, ignore_index=True)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for site, report in availability.groupby('site'):
        base = RESULTS_ROOT / site
        folder = base / 'aggregate' / 'plots' / 'mortality_pretraining_size'
        if seed is not None:
            folder /= f'seed{seed}'
        selections = [(folder, report.seed.unique())]
        for folder, seeds in selections:
            folder.mkdir(parents=True, exist_ok=True)
            report[report.seed.isin(seeds)].to_csv(folder / 'fusion_availability.csv', index=False)
            if combined.empty:
                continue
            subset = combined[(combined.site == site) & combined.seed.isin(seeds)]
            if not subset.empty:
                references = (founder[(founder.site == site) & founder.seed.isin(subset.seed.unique())]
                              if not founder.empty else founder)
                plot_pretraining_size(subset, folder, references)
                print(folder)


def main():
    args = parse_arguments()
    if args.site == "mimic":
        mortality_pretraining_size_results(None if args.all_seeds else args.seed)
    elif args.pretraining_size_plot:
        raise ValueError("Pretraining-size figure is defined only for MIMIC")
    if args.pretraining_size_plot:
        return
    pretraining_size = args.mimic_pretraining_size
    metrics = load_metrics(None if args.all_seeds else args.seed, pretraining_size, site=args.site)
    aggregate_dir = (RESULTS_ROOT / "aggregate" / "resnet_ecgfounder" if args.site == "mimic"
                     else RESULTS_ROOT / args.site / "aggregate" / "resnet_ecgfounder")
    if args.site == "code15" and args.seed is not None:
        aggregate_dir /= f"seed{args.seed}"
    if pretraining_size != 1359:
        aggregate_dir /= f"pretraining{pretraining_size}"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    availability = fusion_availability(metrics)
    availability.to_csv(aggregate_dir / "fusion_availability.csv", index=False)
    pending = int((~availability["late_available"]).sum())
    ecgfounder_availability = availability[availability["architecture"].str.startswith("ecgfounder_")]
    matched_pending = int((~ecgfounder_availability["matched_8_late_available"]).sum())
    print(
        f"Deferred {pending} incomplete native late-fusion groups; "
        f"{matched_pending} incomplete matched-8 groups; see fusion_availability.csv"
    )
    late_fusion = late_fusion_rows(metrics)
    results = pd.concat([metrics, late_fusion], ignore_index=True, sort=False)
    results = results[results["model"].isin(PLOT_MODELS)].copy()

    if results.empty:
        print("No complete fusion results yet; saved availability report. Rerun after more experiments finish.")
        return

    selected_metrics = ["test_patient_auc", "test_patient_pr_auc"] if args.metric == "all" else [args.metric]
    for metric in selected_metrics:
        missing_metric = results[metric].isna()
        if missing_metric.any():
            details = results.loc[missing_metric, EXPERIMENT_KEYS].to_string(index=False)
            raise ValueError(f"Missing values for {metric}:\n{details}")
    results.to_csv(aggregate_dir / "fusion_metrics.csv", index=False)
    for i, metric in enumerate(selected_metrics):
        make_plots(results, metric, write_legacy=i == 0, pretraining_size=pretraining_size, seed=args.seed)
    print(
        f"Saved {len(results)} early/native-late/matched-8-late fusion rows under {RESULTS_ROOT}"
    )


if __name__ == "__main__":
    main()
