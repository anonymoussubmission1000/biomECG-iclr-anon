# Study entry points

Run from the repository root after preparing authorized input data and cohort manifests. Each launcher skips verified completed runs. Defaults use two CPU threads per process; avoid launching more workers than the machine can support.

```bash
python -m biomecg.study.run_pclr_ecg1 --seed 12345
python -m biomecg.study.run_jepa_resnet_ecg1 --seed 12345
python -m biomecg.study.run_cpc_local_ecg1 --seed 12345
python -m biomecg.study.run_ssl_ensembles_ecg1 --family all --seed 12345 --variant all
python -m biomecg.study.run_full_late_covariate_ablation_ecg1 --seed 12345
python -m biomecg.study.evaluate_followup_sensitivity_ecg1 --seeds 12345 --all-configs
python -m biomecg.study.run_followup_sensitivity_ecg1 --seed 12345
```

Cohort preparation and reuse of anonymous released weights are separate operations. Released weights have no patient manifest or resume metadata. `biomecg.study.released_weights.load_encoder` loads PCLR, JEPA and CPC tensors with strict architecture checks. Use the same normalization and sampling rates documented in the corresponding model modules.
