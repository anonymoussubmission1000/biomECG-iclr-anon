# biomECG

Biometric ECG pretraining and downstream mortality, age and sex prediction on
**MIMIC and CODE-15**, using ResNet and ECGFounder.

There are two ways to use this repository:

1. **Run the study:** download the datasets, preprocess the ECGs, train the models
   and compute their results. The full pipelines are included.
2. **Redraw the paper figures:** run the notebook on the supplied result CSVs,
   without datasets, training or inference.

The second route reproduces the **plots**, not the experiments. The datasets must
be obtained separately, and the trained downstream models containing ECGFounder
alone occupy more than 250 GB. We therefore provide the numerical
results as CSVs, together with the plotting notebook. The UMAP panel also uses
supplied projection coordinates.

## 1. Run the study

### Installation and hardware

From this directory, using Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-training.txt
python -m pip install --no-deps -e .
```

The development environment uses **PyTorch 2.11.0 with CUDA 13.0**
(`2.11.0+cu130`). Other dependency versions are recorded in the requirements files,
including NeuroKit2 0.2.12, WFDB 4.3.1 and h5py 3.16.0. CUDA requires a compatible
NVIDIA GPU and driver. CPU execution uses the same training code with
`--device cpu`.

A small, single-lead ResNet experiment can run on a conventional computer. Running
one model is much less work than reproducing the whole study.
CUDA is recommended for full cohorts and ECGFounder finetuning. The complete
study contains thousands of runs.

### Included checkpoints

`checkpoints/local_ssl/` contains tensor-only pretrained weights for the local biometric, PCLR, JEPA and CPC models. `inventory.json` lists each file and checksum. Private-cohort assets and external foundation-model weights are excluded. External model sources are linked in the corresponding scripts.

### Download and prepare the data

Dataset acquisition is inspired by the
[ECG survival benchmark setup instructions](https://github.com/cavalab/ecg-survival-benchmark#setting-up-datasets).
Download the raw data yourself under their respective access conditions.
**Use the local preprocessing commands below:** these apply the signal filtering
used by this project, rather than the benchmark's processed signal exports.

**MIMIC.** Obtain `patients.csv.gz` and `admissions.csv.gz` from the `hosp`
directory of [MIMIC-IV 2.2](https://physionet.org/content/mimiciv/2.2/).
Obtain `machine_measurements.csv`, `record_list.csv` and the waveform `files/`
tree from [MIMIC-IV-ECG 1.0](https://physionet.org/content/mimic-iv-ecg/1.0/).
PhysioNet credentialing and the applicable data agreement are required.
Place them as follows:

```text
data/mimic/raw/
  patients.csv.gz
  admissions.csv.gz
  machine_measurements.csv
  record_list.csv
  files/p1000/...
```

```bash
python -m biomecg.data.preprocess_mimic
```

**CODE-15.** Download `exams.csv` and **all 18** waveform shards,
`exams_part0.hdf5` through `exams_part17.hdf5`, from the
[CODE-15 release](https://zenodo.org/records/4916206).
Place all files directly in `data/code15/raw/`, then run:

```bash
python -m biomecg.data.preprocess_code15
```

Both commands write `data/<site>/processed/ecg.h5` and `labels.csv`, an exclusion
report and a preprocessing journal. Existing output directories are rejected.
An interrupted export must not be used as a completed dataset. To restart, move
its output directory aside and rerun preprocessing, or select a new directory
with `--output-dir`. MIMIC preserves source row and channel order
(`I, II, III, aVR, aVF, aVL, V1, V2, V3, V4, V5, V6`), matching the reference
export. ECGs with NaNs or a lead with raw range or standard deviation below
`1e-6` are excluded.

The common signal processing is 500-sample reflection padding per lead,
`neurokit2.ecg_clean(..., sampling_rate=500, method="neurokit")`, removal of the
reflection padding and a centered 4,096-sample window, stored as float32 in
`(N, 12, 4096)` order. CODE-15 additionally removes contiguous all-lead zero
padding and resamples **400 → 500 Hz** with `resample_poly(5, 4)` before cleaning.
Signals shorter than 4,096 samples after resampling are excluded and logged,
not stretched. No amplitude normalization is applied to these exports.

See [protocol and provenance](docs/protocol.md) for targets, normalization,
and patient split rules.

### Prepare patient splits and biometric cohorts

Examples use seed 12345. Repeat for the seeds listed in `configs/experiments.csv`.
MIMIC uses 12345, 23451, 34512, 45123 and 51234; CODE-15 uses 12345 and 23451.
Seed coverage is listed per architecture in `configs/experiments.csv`.

```bash
python -m biomecg.training.launch --site mimic --seed 12345 --prepare-only
python -m biomecg.training.launch --site code15 --seed 12345 --prepare-only
python -m biomecg.training.pretrain configs/mimic/200.json --seed 12345 --prepare-only
python -m biomecg.training.pretrain configs/code15/200.json --seed 12345 --prepare-only
```

The patient split must exist before biometric selection. For each source, seed
and pretraining size, all lead configurations reuse the same patient manifest.
The supplied weights carry a digest of the original patient selection. Reuse
requires matching patient selection as well as the same seed.

### Biometric pretraining or evaluation

The 151 biometric checkpoints occupy approximately 303 MiB. Their inventory
includes all available MIMIC/CODE-15 source, size and seed combinations, without
optimizer states. The MIMIC size experiment uses 100, 500, 1,359, 8,000, 16,000
and 50,000 pretraining patients (train + validation). CODE-15 uses 1,359.

To reevaluate a supplied checkpoint after preparing its cohort:

```bash
python -m biomecg.training.pretrain configs/mimic/200.json --seed 12345 --phase test
```

To **retrain**, first move the supplied initialization directory aside so that
its checkpoints cannot be overwritten:

```bash
mv checkpoints/biometric checkpoints/biometric_reference
for config in 200 201 202 203 204 205 206 207 208; do
  python -m biomecg.training.pretrain "configs/mimic/${config}.json" --seed 12345 || exit 1
done
for config in 200 201 202 203 204 205 206 207; do
  python -m biomecg.training.pretrain "configs/code15/${config}.json" --seed 12345 || exit 1
done
```

Repeat for the required seeds. Add `--cohort-size 100` (or another listed size)
for the MIMIC scaling experiment; it uses only configs 200–207. Hardware options
are `--device cpu|cuda` and `--workers N`. Completed runs are reported;
an interrupted run causes an error. Optimizer resumption is not implemented.

### Mortality, age and sex

```bash
python -m biomecg.training.launch --site mimic --task mortality --seed 12345 --list
python -m biomecg.training.launch --site mimic --task mortality --seed 12345
python -m biomecg.training.launch --site mimic --task age --seed 12345
python -m biomecg.training.launch --site mimic --task sex --seed 12345
python -m biomecg.training.launch --site code15 --task mortality --seed 12345
python -m biomecg.training.launch --site code15 --task age --seed 12345
python -m biomecg.training.launch --site code15 --task sex --seed 12345
python -m biomecg.training.launch --site mimic --task mortality --seed 12345 --study scaling
```

Repeat main experiments for the seeds present in the plan, and scaling for
12345 and 23451. Preparation manifests for every required pretraining size must
exist first. Cross-source experiments require both source cohorts and their
splits. Age/sex reuse the mortality split and use ECG inputs only.

All runs write to `outputs/downstream/`. Missing weights/data, changed
configurations, invalid labels and incomplete requested ensembles are errors.
Completed compatible runs are reported and reused. Evaluation can be completed
from a finished training checkpoint; interrupted optimization must be restarted
in a new directory.

### Train a single model

The launcher above runs a grid. To start with one small ResNet mortality model
on CPU after preparing MIMIC:

```bash
python -m biomecg.training.mortality_resnet \
  --site mimic \
  --labels_path data/mimic/processed/labels.csv \
  --ecg_path data/mimic/processed/ecg.h5 \
  --horizon 365 --seed 12345 --cohort_size 1000 \
  --model I --architecture scratch --device cpu --bs 32 --num_workers 0
```

Use `--device cuda` for GPU execution. The same command accepts `--epochs` and
`--patience`; changing these changes the training protocol. This example trains
one model and does not supply all results needed by the paper notebook.

### New SSL and sensitivity experiments

The local PCLR, JEPA and CPC implementations, downstream frozen evaluations, fusion controls, demographic ablations and censoring sensitivities are in `biomecg/study/`. Sources and protocol adaptations appear at the top of the model files. Commands are summarized in [docs/commands.md](docs/commands.md).

The migrated study pipeline is an integration draft: full cohort preparation and downstream reproduction still require end-to-end validation. Synthetic model checks and strict loading of released SSL weights have passed.

## 2. Redraw the paper figures

The notebook `notebooks/paper_figs_and_charts.ipynb` only renders figures from supplied aggregate/per-seed result tables. It does not prepare cohorts, train models or run inference. No patient-level data or GPU is required. Run its cells from top to bottom; figures are written to `outputs/paper/figures/`.

```bash
python -m pip install -r requirements-figures.txt
python -m pip install --no-deps -e .
python -m biomecg.reproduce
```

## Repository layout

```text
biomecg/data/       Dataset preparation and ECG filtering
biomecg/models/     Biometric ResNet
biomecg/training/   Biometric and downstream training
biomecg/study/      Local SSL, fusion controls and sensitivity scripts
biomecg/evaluation/ Evaluation and aggregation
configs/           Model and cohort settings
checkpoints/local_ssl/ Tensor-only local pretrained weights
artifacts/tables/  Numerical results for paper figures
notebooks/         Paper figures only
```

Clinical datasets, patient predictions and split manifests must be obtained or generated locally. They are not distributed.
