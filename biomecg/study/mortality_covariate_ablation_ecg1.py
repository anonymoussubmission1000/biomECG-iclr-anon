"""Age/sex and ECG-only ablations of the existing frozen biomECG mortality study.

References: train_resnet_unified_ecg1.py (MLP, loss, optimization, splits,
normalization and metrics); train_resnet_ecgfounder_results_ecg1.py (late fusion).
No biometric pretraining, fine-tuning, or random-encoder study is launched here.
"""
import contextlib
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from biomecg.study import train_resnet_unified_ecg1 as shared
from biomecg.study import pclr_ecg1 as io

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'mortality_covariate_ablation_ecg1_v1'
SEEDS = (12345, 23451, 34512, 45123, 51234)
COHORTS = ('1000', '8000', 'full')
LEADS = tuple(shared.LEADS)
ARTIFACTS = ('best.pt', 'metrics.csv', 'predictions.csv', 'predictions_patient.csv')


def file_signature(path):
    path = Path(path).resolve()
    return dict(path=str(path), size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)


def args_for(site, seed, size, lead='I', device='cuda', workers=4, threads=2):
    config = io.source_config(site, seed)
    paths = config['dir_input'][config['run_from']][config['DB']]
    return SimpleNamespace(site=site, seed=seed, horizon=365, cohort_size=str(size),
        labels_path=Path(paths['labels']), ecg_path=Path(paths['ecg']),
        architecture=f'pretrained_{site}_frozen', model=lead,
        mimic_pretraining_size=1359, pretraining_training_seed=None,
        force_rebuild_split=False, prepare_only=False, device=device,
        lr_encoder=0., lr_head=.001, bs=256, epochs=200, patience=20,
        num_workers=workers, cpu_threads=threads)


def prepare_context(site, seed, device='cuda', workers=4, threads=2):
    args = args_for(site, seed, 'full', device=device, workers=workers, threads=threads)
    split = shared.split_paths(args)[0]
    if not split.is_file():
        raise FileNotFoundError(f'Existing mortality split required: {split}')
    labels = shared.load_labels(args.labels_path, args.ecg_path, 365)
    train, val, test = shared.load_cached_split(split, labels)
    if len(train) < 8000:
        raise ValueError('The saved training split must contain at least 8000 patients')
    groups = {'train': train, 'val': val, 'test': test}
    rows = {name: shared.rows_for_patients(labels, pids) for name, pids in groups.items()}
    all_rows = np.sort(np.concatenate(list(rows.values())))
    if len(np.unique(all_rows)) != len(all_rows):
        raise ValueError('Overlapping ECG rows across splits')
    sources = {}
    for lead in LEADS:
        args.model = lead
        sources[lead] = shared.resolve_biometric_pretraining(args)
    fingerprint = dict(protocol=PROTOCOL, site=site, seed=seed, horizon=365,
        inputs={k: file_signature(getattr(args, k)) for k in ('ecg_path', 'labels_path')},
        split_sha256=io.sha256(split), first_ecg_rows_sha256=hashlib.sha256(all_rows.tobytes()).hexdigest())
    print(f'CHECK {site} seed{seed}: train={len(train)}, val={len(val)}, test={len(test)}; 8 frozen encoders OK', flush=True)
    return dict(args=args, labels=labels, groups=groups, rows=rows, all_rows=all_rows,
                sources=sources, fingerprint=fingerprint)


def ablation_dir(args):
    return (shared.REFACTORED_ROOT / args.site / f'seed{args.seed}' / 'runs' /
            'horizon365d' / f'cohort{args.cohort_size}' / 'ablations')


def run_dir(args, variant):
    if variant == 'age_sex_only':
        return ablation_dir(args) / variant / 'mlp'
    return ablation_dir(args) / variant / args.architecture / args.model


def cache_signature(context, lead, device):
    _, provenance = context['sources'][lead]
    return dict(**context['fingerprint'], lead=lead,
        checkpoint_sha256=io.sha256(provenance['pretraining_checkpoint']),
        pretraining_config_sha256=io.sha256(provenance['pretraining_config']),
        manifest_sha256=io.sha256(provenance['pretraining_cohort_manifest']),
        representation_dim=128, normalization='historical ECGDataset zscore epsilon1e-8',
        encoder_eval=True, extraction_precision='float16' if device.type == 'cuda' else 'float32')


def feature_cache(context, lead, device, signature):
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    folder = ROOT / 'results/MortalityCovariateAblation/cache' / f"{context['args'].site}_seed{context['args'].seed}_{lead}_{key}"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = folder / 'completed.json'
        if marker.exists():
            if json.loads(marker.read_text()) != signature:
                raise ValueError('Frozen feature cache signature mismatch')
            features = np.load(folder / 'features.npy', mmap_mode='r')
            rows = np.load(folder / 'rows.npy')
            if features.shape != (len(context['all_rows']), 128) or not np.array_equal(rows, context['all_rows']):
                raise ValueError('Frozen cache shape/row mismatch')
            return features, rows
        args = context['args']; args.model = lead
        shared.set_seed(args.seed)
        encoder, frozen, _ = shared.build_encoder(args, device, context['sources'][lead])
        if not frozen:
            raise AssertionError('Only frozen encoders may be cached')
        encoder.to(device).eval().requires_grad_(False)
        dataset = shared.ECGDataset(context['labels'], args.ecg_path, context['all_rows'], [shared.LEADS[lead][0]])
        loader = DataLoader(dataset, batch_size=args.bs, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == 'cuda', persistent_workers=False)
        path = folder / 'features.tmp.npy'
        array = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=(len(dataset), 128))
        offset = 0
        with torch.no_grad():
            for ecg, _, _, batch_rows in tqdm(loader, desc=f'Cache {args.site} seed{args.seed} {lead}'):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    features = encoder(ecg.to(device, non_blocking=True))
                values = features.float().cpu().numpy()
                if values.shape != (len(ecg),128) or not np.isfinite(values).all():
                    raise ValueError('Invalid frozen ECG representations')
                if not np.array_equal(batch_rows.numpy(), context['all_rows'][offset:offset+len(ecg)]):
                    raise ValueError('Cache row order changed')
                array[offset:offset+len(ecg)] = values; offset += len(ecg)
        if offset != len(dataset):
            raise ValueError('Incomplete frozen feature extraction')
        array.flush(); del array, encoder, loader, dataset
        path.replace(folder / 'features.npy')
        with (folder / 'rows.tmp.npy').open('wb') as handle:
            np.save(handle, context['all_rows'])
        (folder / 'rows.tmp.npy').replace(folder / 'rows.npy')
        io.atomic_json(marker, signature)
        return np.load(folder / 'features.npy', mmap_mode='r'), context['all_rows']


class AblationDataset(Dataset):
    """Same first-ECG rows and targets; excluded inputs are never exposed."""
    def __init__(self, labels, rows, variant, cache=None):
        if variant not in ('age_sex_only', 'ecg_only'):
            raise ValueError('Unknown ablation variant')
        self.rows = np.asarray(rows, dtype=np.int64)
        self.targets = labels.iloc[self.rows]['_event'].to_numpy(dtype=np.float32)
        self.variant = variant
        if variant == 'age_sex_only':
            self.metadata = labels.iloc[self.rows][['Age', 'Is_Male']].to_numpy(dtype=np.float32)
            self.metadata[:,0] /= 100.
            self.features = None
        else:
            self.metadata = np.zeros((len(rows),2), dtype=np.float32)
            features, cached_rows = cache
            positions = np.searchsorted(cached_rows, self.rows)
            if (positions >= len(cached_rows)).any() or not np.array_equal(cached_rows[positions], self.rows):
                raise ValueError('Patient ECG missing from cache')
            self.features = features[positions].copy()
        if not np.isfinite(self.metadata).all():
            raise ValueError('Nonfinite covariates')

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        features = np.empty(0, dtype=np.float32) if self.features is None else self.features[index]
        return (torch.from_numpy(features), torch.from_numpy(self.metadata[index]),
                torch.tensor(self.targets[index]), torch.tensor(self.rows[index], dtype=torch.long))


def completed(folder, expected):
    config = folder / 'run_config.json'; marker = folder / 'completed.json'
    if config.exists() and json.loads(config.read_text()) != expected:
        raise ValueError(f'Ablation run configuration differs: {folder}')
    if not marker.exists(): return False
    saved = json.loads(marker.read_text())
    if saved['signature'] != expected:
        raise ValueError(f'Ablation completion configuration differs: {folder}')
    for name in ARTIFACTS:
        path = folder / name
        if not path.is_file() or io.sha256(path) != saved['artifacts'][name]:
            raise ValueError(f'Incomplete or changed ablation artifact: {path}')
    return True


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, value):
        for stream in self.streams: stream.write(value); stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()


def train_one(context, args, variant, device):
    expected = dict(**context['fingerprint'], variant=variant, cohort_size=args.cohort_size,
        architecture=args.architecture if variant=='ecg_only' else 'covariates_only',
        model=args.model if variant=='ecg_only' else 'mlp',
        head=[128 if variant=='ecg_only' else 2,256,128,1],
        lr_head=args.lr_head, batch_size=args.bs, epochs=args.epochs, patience=args.patience,
        optimizer='historical AdamW defaults', loss='BCE pos_weight negatives/positives on train cohort',
        selection='maximum validation AUROC', precision='float16' if device.type=='cuda' else 'float32')
    if variant == 'ecg_only': expected['features'] = cache_signature(context, args.model, device)
    folder = run_dir(args, variant); folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if completed(folder, expected):
            print('SKIP',folder,flush=True); return
        io.atomic_json(folder / 'run_config.json', expected)
        cache = feature_cache(context,args.model,device,expected['features']) if variant=='ecg_only' else None
        labels = context['labels']; groups = context['groups']
        n = len(groups['train']) if args.cohort_size=='full' else int(args.cohort_size)
        if not 0 < n <= len(groups['train']): raise ValueError('Invalid nested training cohort')
        rows = dict(context['rows']); rows['train'] = shared.rows_for_patients(labels, groups['train'][:n])
        datasets = {g:AblationDataset(labels,r,variant,cache) for g,r in rows.items()}
        # Preserve the historical initialization order for ECG heads, then replace
        # the frozen encoder by identity because inputs are its exact cached outputs.
        shared.set_seed(args.seed)
        encoder = (shared.build_encoder(args,device,context['sources'][args.model])[0]
                   if variant=='ecg_only' else nn.Identity())
        model = shared.MortalityModel(encoder,True,input_mode=variant)
        model.encoder = nn.Identity().eval()
        model.to(device); model.amp_dtype = torch.float16
        loaders = {g:DataLoader(ds,batch_size=args.bs,shuffle=g=='train',num_workers=0,
            pin_memory=device.type=='cuda') for g,ds in datasets.items()}
        with (folder / 'training.log').open('a') as log, contextlib.redirect_stdout(Tee(sys.stdout,log)):
            print('RUN',args.site,args.seed,args.cohort_size,variant,args.model,flush=True)
            checkpoint = shared.train_model(model,loaders['train'],loaders['val'],
                labels.iloc[rows['train']]['_event'].to_numpy(),args,device,folder/'best.pt')
            scores, targets, indices = shared.predict(model,loaders['test'],device,'Test ablation')
        predictions = pd.DataFrame(dict(row_index=indices,
            pid=labels.iloc[indices]['_pid'].to_numpy(),
            ecg_order=labels.iloc[indices]['ECG_Order'].to_numpy(),label=targets,prediction=scores)).sort_values('row_index')
        if predictions.pid.duplicated().any(): raise ValueError('Duplicate test patient')
        predictions.to_csv(folder/'predictions.csv',index=False)
        predictions.assign(n_ecgs=1).to_csv(folder/'predictions_patient.csv',index=False)
        metrics = dict(site=args.site,seed=args.seed,horizon_days=365,cohort_label=args.cohort_size,
            variant=variant,architecture=expected['architecture'],model=expected['model'],
            n_train_patients=n,n_val_patients=len(rows['val']),n_test_patients=len(rows['test']),
            best_epoch=int(checkpoint['epoch']),best_val_auc_ecg=float(checkpoint['validation_auc']),
            input_dim=expected['head'][0],head_parameters=sum(p.numel() for p in model.head.parameters()),
            split_path=str(shared.split_paths(args)[0]),split_sha256=expected['split_sha256'])
        if variant=='ecg_only': metrics.update(context['sources'][args.model][1])
        metrics.update(shared.calculate_metrics(targets,scores,'test_patient'))
        pd.DataFrame([metrics]).to_csv(folder/'metrics.csv',index=False)
        io.atomic_json(folder/'completed.json',dict(signature=expected,
            artifacts={name:io.sha256(folder/name) for name in ARTIFACTS}))
        print('COMPLETE',folder,flush=True)


def mean_predictions(paths):
    """Exactly the historical late fusion: mean probabilities on matched patients."""
    reference = None; scores = []
    for path in paths:
        frame = pd.read_csv(path,dtype={'pid':str}).sort_values('row_index').reset_index(drop=True)
        if frame.pid.duplicated().any() or not np.isfinite(frame.prediction).all():
            raise ValueError(f'Invalid predictions: {path}')
        identity = frame[['row_index','pid','ecg_order','label']]
        if reference is None: reference = identity
        elif not reference.equals(identity): raise ValueError(f'Mismatched test patients/labels: {path}')
        scores.append(frame.prediction.to_numpy())
    if not scores: raise ValueError('No predictions to aggregate')
    return reference.assign(prediction=np.mean(scores,axis=0))


def summarize(context):
    rows = []
    for size in COHORTS:
        args = SimpleNamespace(**vars(context['args'])); args.cohort_size=size
        age_path = run_dir(args,'age_sex_only')/'predictions.csv'
        ecg_paths=[]; combined_paths=[]
        for lead in LEADS:
            args.model=lead
            ecg_paths.append(run_dir(args,'ecg_only')/'predictions.csv')
            combined_paths.append(shared.output_dir(args)/'predictions.csv')
        variants = {'age_sex_only':[age_path], 'ecg_only_late':ecg_paths,
                    'ecg_age_sex_late_existing':combined_paths}
        reference = None; scores_by_variant={}
        for name, paths in variants.items():
            if not all(p.is_file() for p in paths):
                print('Missing comparison reference:',name,size,flush=True); continue
            predictions = mean_predictions(paths)
            identity = predictions[['row_index','pid','ecg_order','label']]
            if reference is None: reference=identity
            elif not reference.equals(identity): raise ValueError('Ablation and combined test sets differ')
            scores_by_variant[name] = shared.calculate_metrics(predictions.label,predictions.prediction,'test_patient')
            if name=='ecg_only_late':
                folder=ablation_dir(args)/'ecg_only'/'late';folder.mkdir(exist_ok=True,parents=True)
                predictions.to_csv(folder/'predictions.csv',index=False)
                predictions.assign(n_ecgs=1).to_csv(folder/'predictions_patient.csv',index=False)
                pd.DataFrame([scores_by_variant[name]]).to_csv(folder/'metrics.csv',index=False)
            rows.append(dict(site=args.site,seed=args.seed,cohort_size=size,variant=name,
                n_experts=len(paths),**scores_by_variant[name]))
        if 'age_sex_only' in scores_by_variant:
            baseline=scores_by_variant['age_sex_only']
            for row in rows:
                if row['cohort_size']==size:
                    for metric in ('auc','pr_auc'):
                        row[f'delta_{metric}_vs_age_sex']=row[f'test_patient_{metric}']-baseline[f'test_patient_{metric}']
    path=shared.REFACTORED_ROOT/context['args'].site/f"seed{context['args'].seed}"/'mortality_covariate_ablation_summary.csv'
    pd.DataFrame(rows).to_csv(path,index=False)
    print('SUMMARY',path,flush=True)
    return pd.DataFrame(rows)


def run_context(context, device):
    args=context['args']
    for size in COHORTS:
        args.cohort_size=size
        train_one(context,args,'age_sex_only',device)
    for lead in LEADS:
        args.model=lead
        for size in COHORTS:
            args.cohort_size=size
            train_one(context,args,'ecg_only',device)
    summarize(context)
