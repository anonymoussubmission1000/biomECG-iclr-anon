"""ECG-only age/sex downstream, sharing mortality encoders and saved splits.

CODE-15 adds frozen CODE-15 biometric experts on eight individual leads to the
MIMIC age/sex grid. It uses cohorts 1000/8000/full and its existing mortality
split. --site selects default data paths.
No age/sex covariates enter the network. Never creates or rebuilds a patient
split: use the existing mortality horizon365d split of the same seed. Missing
or invalid targets fail explicitly instead of silently changing cohort sizes.
"""
import argparse
from copy import copy
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from biomecg.study import train_resnet_unified_ecg1 as resnet

RESULTS_ROOT = resnet.REFACTORED_ROOT
COHORT_SIZES = [500, 1000, 2000, 8000, 50000, 'full']
ARCHITECTURES = ['scratch', 'pretrained_frozen', 'pretrained_finetuned',
                 *resnet.MIMIC_ARCHITECTURES, 'pretrained_code15_frozen', 'ecgfounder_frozen', 'ecgfounder_finetuned']
FOUNDER_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def cohort_sizes(site):
    return [1000, 8000, 'full'] if site == 'code15' else COHORT_SIZES


def site_paths(site):
    """Return the preprocessed inputs without creating or rebuilding anything."""
    if site == 'code15':
        folder = ROOT / 'data/code15/processed_all_ecgs_500hz'
        return folder / 'Labels_Code15_all_ecgs.csv', folder / 'Code15_ECG.hdf5'
    if site == 'mimic':
        folder = ROOT / 'data/MIMIC IV'
        return folder / 'Labels_MIMICIV_mort_062026_pd_8020.csv', folder / 'MIMIC_ECG.hdf5'
    raise ValueError(f'Unknown site: {site}')


def output_dir(args):
    return (RESULTS_ROOT / resnet.slug(args.site) / f'seed{args.seed}' / 'runs' /
            args.task / f'cohort{args.cohort_size}' / args.architecture / args.model)


def completed(folder):
    return all((folder / name).is_file() and (folder / name).stat().st_size > 0
               for name in ('best.pt', 'metrics.csv', 'predictions.csv',
                            'predictions_patient.csv', 'run_config.json', 'training_complete.json'))


def load_data(args):
    """Reuse split membership AND nested order; select first ECG before targets."""
    labels = resnet.load_labels(args.labels_path, args.ecg_path, 365)
    path = resnet.split_paths(args)[0]
    if not path.is_file():
        raise FileNotFoundError(f'Existing mortality split required: {path}')
    groups = resnet.load_cached_split(path, labels)
    for patients in groups:
        if len(patients) != len(set(patients)):
            raise ValueError('Duplicate patients in saved split')
    column = 'Age' if args.task == 'age' else 'Is_Male'
    targets = pd.to_numeric(labels[column], errors='coerce').to_numpy(dtype=np.float32)
    rows = resnet.rows_for_patients(labels, np.concatenate(groups))
    values = targets[rows]
    if not np.isfinite(values).all():
        raise ValueError(f'Missing/nonfinite {column} on first ECG; resolve before changing cohorts')
    if args.task == 'sex' and not np.isin(values, [0, 1]).all():
        raise ValueError('Is_Male must be encoded as 0/1')
    if args.task == 'age' and (values < 0).any():
        raise ValueError('Age must be nonnegative')
    return labels, targets, groups, path


class ECGOnlyDataset(Dataset):
    """Return signal, target and row ID only; no demographic input tensor."""
    def __init__(self, ecg_path, targets, rows, leads):
        self.path, self.targets = str(ecg_path), targets
        self.rows, self.leads = np.asarray(rows), np.asarray(leads)
        self.handle = None

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        return dict(self.__dict__, handle=None)

    def __del__(self):
        if getattr(self, 'handle', None) is not None:
            self.handle.close()

    def __getitem__(self, index):
        if self.handle is None:
            self.handle = h5py.File(self.path, 'r')
        row = int(self.rows[index])
        signal = self.handle['ECG'][row, self.leads].astype(np.float32)
        if not np.isfinite(signal).all():
            raise ValueError(f'Nonfinite ECG at row {row}')
        signal = (signal - signal.mean(axis=1, keepdims=True)) / (signal.std(axis=1, keepdims=True) + 1e-8)
        return torch.from_numpy(signal), torch.tensor(self.targets[row]), row


class ECGOnlyModel(nn.Module):
    def __init__(self, encoder, frozen, dimension):
        super().__init__()
        self.encoder, self.frozen, self.dimension = encoder, frozen, dimension
        self.head = nn.Sequential(nn.Linear(dimension, 256), nn.ReLU(),
                                  nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        if frozen:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, signal):
        if self.frozen:
            with torch.no_grad():
                features = self.encoder(signal)
        else:
            features = self.encoder(signal)
        if features.ndim != 2 or features.shape[1] != self.dimension:
            raise ValueError('Unexpected encoder feature shape')
        return self.head(features).squeeze(1)


def build_model(args, device, source=None):
    if args.architecture.startswith('ecgfounder_'):
        from biomecg.study import train_ecgfounder_unified_ecg1 as founder
        mapped = copy(args)
        mapped.architecture = args.architecture.replace('ecgfounder_', 'pretrained_')
        encoder, frozen, config_id = founder.build_encoder(mapped, device)
        dimension = founder.REPRESENTATION_SIZE
        leads = founder.EARLY_LEADS if args.model == 'early_fusion' else [founder.LEADS[args.model]]
    else:
        encoder, frozen, config_id = resnet.build_encoder(args, device, source)
        dimension = 128
        leads = resnet.EARLY_LEADS if args.model == 'early_fusion' else [resnet.LEADS[args.model][0]]
    return ECGOnlyModel(encoder, frozen, dimension).to(device), leads, config_id


def calculate_metrics(task, labels, scores):
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite predictions')
    if task == 'age':
        return dict(test_patient_mae=mean_absolute_error(labels, scores),
                    test_patient_rmse=float(np.sqrt(mean_squared_error(labels, scores))),
                    test_patient_r2=r2_score(labels, scores),
                    test_patient_bias=float(np.mean(scores - labels)))
    metrics = resnet.calculate_metrics(labels, scores, 'test_patient')
    tn, fp, fn, tp = (metrics[f'test_patient_{key}'] for key in ('tn', 'fp', 'fn', 'tp'))
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    metrics.update(test_patient_sensitivity=sensitivity, test_patient_specificity=specificity,
                   test_patient_balanced_accuracy=(sensitivity + specificity) / 2)
    return metrics


def age_scaling(targets):
    mean, scale = float(np.mean(targets)), float(np.std(targets))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Training ages must have positive variance')
    return mean, scale


def predict(model, loader, device, task, mean, scale):
    model.eval()
    scores, labels, rows = [], [], []
    with torch.no_grad():
        for signal, target, row in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                output = model(signal.to(device))
            output = output.float()
            output = output * scale + mean if task == 'age' else output.sigmoid()
            scores.append(output.cpu().numpy())
            labels.append(target.numpy())
            rows.append(row.numpy())
    return tuple(np.concatenate(items) for items in (scores, labels, rows))


def train(model, train_loader, val_loader, args, device, folder, mean, scale):
    """Same optimizer groups as mortality; select only on validation MAE/AUROC."""
    groups = [{'params': model.head.parameters(), 'lr': args.lr_head}]
    if not model.frozen:
        groups.append({'params': model.encoder.parameters(), 'lr': args.lr_encoder})
    optimizer = torch.optim.AdamW(groups)
    criterion = nn.MSELoss() if args.task == 'age' else nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    best, stale, history = -np.inf, 0, []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for signal, target, _ in train_loader:
            target = target.to(device)
            if args.task == 'age':
                target = (target - mean) / scale
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                loss = criterion(model(signal.to(device)), target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scores, labels, _ = predict(model, val_loader, device, args.task, mean, scale)
        metrics = calculate_metrics(args.task, labels, scores)
        value = -metrics['test_patient_mae'] if args.task == 'age' else metrics['test_patient_auc']
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), validation_selection=value))
        pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
        print(history[-1], flush=True)
        if value > best:
            best, stale = value, 0
            torch.save(dict(model_state_dict=model.state_dict(), epoch=epoch,
                            validation_selection=value, age_mean=mean, age_scale=scale,
                            task=args.task), folder / 'best.pt')
        else:
            stale += 1
        if stale >= args.patience:
            break
    if not (folder / 'best.pt').is_file():
        raise RuntimeError('Training produced no valid checkpoint')
    (folder / 'training_complete.json').write_text(json.dumps({'best_validation_selection': best}))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=['age', 'sex'])
    parser.add_argument('--site', choices=['mimic', 'code15'], default='mimic')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--labels_path', type=Path)
    parser.add_argument('--ecg_path', type=Path)
    parser.add_argument('--cohort_size', default='full')
    parser.add_argument('--architecture', choices=ARCHITECTURES, default='scratch')
    parser.add_argument('--model', choices=[*FOUNDER_LEADS, 'early_fusion'], default='early_fusion')
    parser.add_argument('--lr_encoder', type=float, default=1e-4)
    parser.add_argument('--lr_head', type=float, default=1e-3)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--cpu_threads', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--prepare_only', action='store_true')
    args = parser.parse_args()
    labels_path, ecg_path = site_paths(args.site)
    args.labels_path = args.labels_path or labels_path
    args.ecg_path = args.ecg_path or ecg_path
    args.horizon, args.force_rebuild_split, args.mimic_pretraining_size = 365, False, 1359
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    resnet.set_seed(args.seed)
    if not args.architecture.startswith('ecgfounder_') and args.model not in [*resnet.LEADS, 'early_fusion']:
        raise ValueError('ResNet supports I, II, V1-V6 or early_fusion')
    if args.epochs < 1 or args.patience < 1 or args.bs < 1:
        raise ValueError('epochs, patience and batch size must be positive')
    labels, targets, (order, val, test), split_path = load_data(args)
    size = len(order) if args.cohort_size == 'full' else int(args.cohort_size)
    if not 1 < size <= len(order):
        raise ValueError(f'Invalid training cohort size: {size}')
    rows = [resnet.rows_for_patients(labels, patients) for patients in (order[:size], val, test)]
    if args.task == 'sex' and any(len(np.unique(targets[r])) != 2 for r in rows):
        raise ValueError('Each split must contain both sex classes')
    mean, scale = age_scaling(targets[rows[0]]) if args.task == 'age' else (0., 1.)
    source = resnet.resolve_biometric_pretraining(args) if args.architecture in resnet.BIOMETRIC_ARCHITECTURES else None
    if args.prepare_only:
        print(f'{args.task}: train={size}, val={len(val)}, test={len(test)}; split={split_path}')
        return
    folder = output_dir(args)
    provenance = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    provenance.update(age_mean=mean, age_scale=scale, covariates=[],
                      split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest())
    if source:
        provenance.update(source[1])
    elif args.architecture.startswith('ecgfounder_'):
        n_leads = 12 if args.model == 'early_fusion' else 1
        provenance['pretraining_checkpoint'] = str(ROOT / 'biomecg/study/foundation_assets/checkpoints' / f'{n_leads}_lead_ECGFounder.pth')
    elif args.architecture != 'scratch':
        cid = 208 if args.model == 'early_fusion' else resnet.LEADS[args.model][1]
        config_path = ROOT / f'configs/excluded_source/{cid}.json'
        config = json.loads(config_path.read_text())
        provenance['pretraining_config'] = str(config_path)
        provenance['pretraining_checkpoint'] = str(ROOT / 'results' / config['network'] / 'checkpoints' / config['name'] / 'network.pt')
    inputs = [args.labels_path, args.ecg_path]
    for key in ('pretraining_checkpoint', 'pretraining_config'):
        if key in provenance:
            inputs.append(Path(provenance[key]))
    provenance['input_fingerprints'] = {str(p.resolve()): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in inputs}
    saved = folder / 'run_config.json'
    if saved.exists():
        previous = json.loads(saved.read_text())
        ignored = {'device', 'num_workers', 'cpu_threads'}
        if {k:v for k,v in previous.items() if k not in ignored} != {k:v for k,v in provenance.items() if k not in ignored}:
            raise ValueError(f'Existing experiment configuration differs: {saved}')
    elif folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f'Artifacts without provenance: {folder}')
    if completed(folder):
        print(f'ALREADY COMPLETE | {folder}')
        return
    if (folder / 'best.pt').exists() and not (folder / 'training_complete.json').exists():
        raise FileExistsError(f'Interrupted training: inspect before restarting {folder}')
    folder.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(provenance, indent=2) + '\n')
    device = torch.device(args.device)
    model, leads, config_id = build_model(args, device, source)
    loaders = [DataLoader(ECGOnlyDataset(args.ecg_path, targets, r, leads), batch_size=args.bs,
                          shuffle=i == 0, num_workers=args.num_workers,
                          pin_memory=device.type == 'cuda', persistent_workers=args.num_workers > 0)
               for i, r in enumerate(rows)]
    if not (folder / 'training_complete.json').exists():
        train(model, loaders[0], loaders[1], args, device, folder, mean, scale)
    checkpoint = torch.load(folder / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    scores, truth, indices = predict(model, loaders[2], device, args.task, mean, scale)
    predictions = pd.DataFrame(dict(row_index=indices, pid=labels.iloc[indices]['_pid'].to_numpy(),
                                   ecg_order=labels.iloc[indices]['ECG_Order'].to_numpy(),
                                   label=truth, prediction=scores)).sort_values('row_index')
    predictions.to_csv(folder / 'predictions.csv', index=False)
    predictions.assign(n_ecgs=1).to_csv(folder / 'predictions_patient.csv', index=False)
    metrics = dict(site=args.site, seed=args.seed, task=args.task, cohort_label=args.cohort_size,
                   architecture=args.architecture, model=args.model, config_id=config_id,
                   n_train_patients=size, n_val_patients=len(val), n_test_patients=len(test),
                   best_epoch=checkpoint['epoch'], validation_selection=checkpoint['validation_selection'],
                   split_path=str(split_path), age_mean=mean, age_scale=scale, covariates='none')
    if source:
        metrics.update(source[1])
    metrics.update(calculate_metrics(args.task, truth, scores))
    # Publish metrics last: aggregators never consume a half-written run.
    temporary = folder / 'metrics.tmp.csv'
    pd.DataFrame([metrics]).to_csv(temporary, index=False)
    temporary.replace(folder / 'metrics.csv')


if __name__ == '__main__':
    main()
