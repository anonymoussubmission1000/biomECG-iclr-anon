# Local SSL

Same 1,359-patient cohorts: 1,087 training, 272 validation. Sources are listed at the top of model scripts.

- PCLR: biomECG ResNet, 128→256→256 projection; eight patients/batch, two ECG draws with replacement, five passes/epoch. Pair exposure is approximately calibrated, not compute-matched. Adam, cosine learning rate, 50 epochs.
- JEPA: biomECG ResNet temporal features; waveform masking and visible-only normalization; convolutional predictor and EMA teacher. 200 epochs; clean training-only BatchNorm calibration.
- CPC: native twelve-lead CNN–S4; full 8.192 s resampled to 240 Hz; fixed validation negatives; 200 epochs maximum, patience 20. Cohort-matched, not backbone-matched.

Frozen downstream mortality: 1k/8k/Full. PCLR/JEPA support early, lead-wise late and early-ensemble variants. Covariate ablations retrain heads. Censoring sensitivity retains deaths within 365 days; test-only evaluation and filtered-partition retraining are separate.
