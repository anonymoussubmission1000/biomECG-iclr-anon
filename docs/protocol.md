# Protocol and provenance

## Signals and cohorts

The data acquisition and MIMIC follow-up construction are inspired by
[the ECG survival benchmark](https://github.com/cavalab/ecg-survival-benchmark).
The signal pipeline is local: NeuroKit cleaning at 500 Hz after one second of
reflection padding, followed by a central 4,096-sample crop. CODE-15 zero padding
is removed before 400-to-500 Hz resampling. The 50 Hz powerline default of the
NeuroKit call is retained; it is not replaced with a dataset-specific frequency.

The MIMIC label timestamp comes from `record_list.csv`. Death is the presence of
`dod`; its delay uses the ECG calendar date. Without death, follow-up ends at the
last ECG or, when available, the later of last ECG and last discharge plus 365
days. Age is ECG year minus anchor year plus anchor age. The retained first ECG
is the minimum `ECG_Order` after signal exclusions.

CODE-15 retains every valid examination for biometrics, including examinations
without survival labels. Only the survival-labelled first examination enters
mortality. Missing outcomes do not become negatives. Its follow-up is
`max(timey, 1/365) * 365` days; administrative session order is not chronology.

## Predictions and patient separation

Mortality is binary classification at 365 days:
`(E_Mortality == 1) & (TTE_Mortality <= 365)`. Patients without recorded death
within the horizon receive a negative label.
Shorter follow-up and negative recorded follow-up are retained; this is a
limitation of the binary endpoint, which does not model censoring.

Patient splits are stratified 70/15/15, using seed and seed+1. Nested training
cohorts are prefixes of one saved patient order. Validation/test stay fixed.
Biometric patients come only from the downstream training pool: first N eligible
patients for biometric train/validation, next 292 for biometric test. Eligibility
requires at least two sessions, all 12 leads finite, and lead-wise standard
deviation above 1e-6. Train/validation is split 80/20 with the experiment seed.
Each session generates five positive and five negative pairs. Biometric test
patients change with pretraining size, so their EERs do not share one fixed test
population across sizes.

Biometric signals use per-ECG/per-lead z-score with epsilon 1e-6; downstream
signals use epsilon 1e-8. ResNet encoder output has dimension 128. Mortality heads
also receive age/100 and sex; age/sex tasks receive ECG only. Frozen encoders
remain in evaluation mode during head training. Mortality selects validation
AUROC with class-weighted BCE; age selects validation MAE with train-only target
standardization; sex selects validation AUROC with unweighted BCE.

Late fusion averages probabilities for mortality/sex and predicted years for age.
Main comparisons use I, II, V1–V6 for both families. Early fusion uses eight leads
for ResNet and twelve for ECGFounder. The separate 8-versus-12 panel uses matched
seeds. Bands are descriptive sample SDs, not confidence intervals.

## Data exports

Signals are stored as `(N, 12, 4096)` arrays with aligned patient and ECG
identifiers. Preprocessing writes an exclusion report and a completion marker.
MIMIC requires space for the temporary and final HDF5 exports.

MIMIC follows the signal extraction in `representations/mimic/process_ecg.py`
and `tools.clean_cut` from the experiment code: source row order, native WFDB
channel order (`I, II, III, aVR, aVF, aVL, V1, V2, V3, V4, V5, V6`), and
exclusion of raw NaNs and leads with range or standard deviation below `1e-6`.
The metadata calculations follow `preproc_explore.ipynb`. The export retains
channel positions used for training the supplied models. In the historical
ECGFounder experiment labels, `aVL` denotes channel 4 and `aVF` channel 5;
these positions correspond to native MIMIC `aVF` and `aVL`, respectively.
The eight-lead comparisons do not use these two channels.

## Third-party network

The included ECGFounder network is the required subset of the
[official implementation](https://github.com/PKUDigitalHealth/ECGFounder), based on
revision `04edac702b61c91face519774ddcc0cd712fef23`. Its organizational MIT license
is retained in `third_party/ecgfounder/LICENSE`. The
[official model card](https://huggingface.co/PKUDigitalHealth/ECGFounder)
also labels the pretrained weights as MIT-licensed.

`checkpoints/ecgfounder_sources.json` records SHA-256 hashes of the original
upstream downloads; `checkpoints/ecgfounder_inventory.csv` records hashes of the
three supplied files, with the two initializations saved as network tensors only.
No upstream training examples or datasets are bundled.
