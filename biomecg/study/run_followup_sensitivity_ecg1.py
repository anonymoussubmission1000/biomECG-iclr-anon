"""Refit frozen downstream models after excluding unascertained 365-day outcomes.
Original splits and nested training subsets are retained; exclusions are not replaced.
Each worker runs in its own process and patches only row selection and output paths.
"""
import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FAMILIES = ('biomecg', 'pclr', 'jepa', 'ecgfounder', 'ecgcpc', 'ecgjepa', 'cpc_local')
LEADS = ('I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
OUTPUT = ROOT / 'outputs/sensitivity'


def eligible_mask(duration, event):
    import numpy as np
    duration, event = np.asarray(duration, dtype=float), np.asarray(event, dtype=float)
    if not np.isfinite(duration).all() or not np.isin(event, [0, 1]).all():
        raise ValueError('Invalid follow-up/event values')
    return (duration >= 0) & ((duration >= 365) | ((event == 1) & (duration <= 365)))


def folder(a, model=None):
    return OUTPUT / a.site / f'seed{a.seed}' / f'cohort{a.cohort}' / a.family / (model or a.model)


def worker(a):
    import numpy as np
    import pandas as pd
    from biomecg.study import pclr_ecg1 as io
    from biomecg.study import train_resnet_unified_ecg1 as shared
    stem = {'biomecg': 'resnet', 'pclr': 'pclr', 'jepa': 'jepa_resnet', 'ecgfounder': 'ecgfounder'}.get(a.family, 'foundation')
    trainer = importlib.import_module(f'biomecg.study.train_{stem}_unified_ecg1')
    engine = shared if a.family in ('biomecg', 'pclr', 'jepa') else trainer
    cfg = io.source_config(a.site, a.seed)
    paths = cfg['dir_input'][cfg['run_from']][cfg['DB']]
    arch = {'biomecg': f'pretrained_{a.site}_frozen', 'pclr': f'pclr_{a.site}_frozen', 'jepa': f'jepa_resnet_early_{a.site}_frozen'}.get(a.family, 'pretrained_frozen')
    cli = ['--site', a.site, '--seed', str(a.seed), '--labels_path', str(paths['labels']), '--ecg_path', str(paths['ecg']), '--horizon', '365', '--cohort_size', a.cohort, '--architecture', arch, '--num_workers', str(a.num_workers), '--cpu_threads', '2']
    if stem == 'foundation':
        cli += ['--family', 'ecgcpc' if a.family == 'cpc_local' else a.family]
        if a.family == 'cpc_local':
            cli += ['--cpc-local']
    else:
        cli += ['--model', a.model, '--bs', '256', '--lr_encoder', '0', '--lr_head', '0.001', '--epochs', '200', '--patience', '20']
    sys.argv = [__file__, *cli]
    args = engine.parse_args()
    split_engine = shared if stem == 'foundation' else engine
    split = split_engine.split_paths(args)[0]
    if not split.is_file():
        raise FileNotFoundError(split)
    labels = split_engine.load_labels(args.labels_path, args.ecg_path, 365)
    groups = split_engine.load_cached_split(split, labels)
    n = len(groups[0]) if a.cohort == 'full' else int(a.cohort)
    if n > len(groups[0]):
        raise ValueError('Training budget exceeds saved partition')
    groups = [groups[0][:n], groups[1], groups[2]]
    original_rows = split_engine.rows_for_patients
    counts, selected = [], []
    for name, patients in zip(('train', 'validation', 'test'), groups):
        rows = original_rows(labels, patients)
        truth = labels.iloc[rows]
        keep = eligible_mask(truth.TTE_Mortality, truth.E_Mortality)
        retained = rows[keep]
        if len(np.unique(labels.iloc[retained]._event)) != 2:
            raise ValueError(f'Both outcome classes required: {name}')
        counts.append(dict(partition=name, n_original=len(rows), n_retained=len(retained), n_excluded=int((~keep).sum()), n_negative_duration=int((truth.TTE_Mortality < 0).sum()), n_events=int(labels.iloc[retained]._event.sum())))
        selected.extend(dict(partition=name, pid=str(labels.iloc[r]._pid), row_index=int(r)) for r in retained)
    signature = dict(protocol='followup_refit_v1', family=a.family, site=a.site, seed=a.seed, cohort=a.cohort, model=a.model, arguments=cli, split_sha256=io.sha256(split), labels_sha256=io.sha256(args.labels_path), counts=counts)
    # Worker count affects throughput only.
    signature['arguments'] = [*cli]
    signature['arguments'][cli.index('--num_workers') + 1] = '16'
    target = folder(a)
    if a.check_only:
        if a.family in ('pclr', 'jepa'):
            trainer.signature(args)
        elif a.family == 'biomecg':
            shared.resolve_biometric_pretraining(args)
        elif stem == 'foundation':
            trainer.signature(args)
        print('READY', target, json.dumps(counts), flush=True)
        return
    target.mkdir(parents=True, exist_ok=True)
    with (target / '.sensitivity.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, done = target / 'sensitivity_config.json', target / 'sensitivity_completed.json'
        if config.exists() and json.loads(config.read_text()) != signature:
            raise ValueError(f'Configuration mismatch: {target}')
        required = ('best.pt', 'metrics.csv', 'predictions.csv', 'predictions_patient.csv')
        if done.exists() and json.loads(done.read_text()) == signature and all((target / f).is_file() and (target / f).stat().st_size for f in required):
            print('SKIP', target, flush=True)
            return
        io.atomic_json(config, signature)
        pd.DataFrame(counts).to_csv(target / 'cohort_counts.csv', index=False)
        pd.DataFrame(selected).to_csv(target / 'retained_patients.csv', index=False)
        def filtered_rows(table, patients):
            rows = original_rows(table, patients)
            truth = table.iloc[rows]
            return rows[eligible_mask(truth.TTE_Mortality, truth.E_Mortality)]
        split_engine.rows_for_patients = filtered_rows
        # Require existing splits and avoid any split creation or metadata changes.
        if stem != 'foundation':
            split_engine.create_or_load_split = lambda table, unused: (*split_engine.load_cached_split(split, table), split)
        engine.output_dir = lambda unused: target
        engine.parse_args = lambda *unused: args
        trainer.main()
        if not all((target / f).is_file() for f in required):
            raise RuntimeError(f'Missing outputs: {target}')
        pred = pd.read_csv(target / 'predictions.csv', dtype={'pid': str})
        expected = {r['pid'] for r in selected if r['partition'] == 'test'}
        if set(pred.pid) != expected or pred.pid.duplicated().any():
            raise ValueError('Test predictions do not match retained patients')
        metrics = pd.read_csv(target / 'metrics.csv')
        for count in counts:
            prefix = {'train': 'train', 'validation': 'val', 'test': 'test'}[count['partition']]
            metrics[f'n_{prefix}_patients'] = count['n_retained']
            metrics[f'n_{prefix}_patients_before_exclusion'] = count['n_original']
            for column in (f'{prefix}_event_rate', f'{prefix}_prevalence', f'{prefix}_event_rate_patient', f'{prefix}_event_rate_ecg'):
                if column in metrics:
                    metrics[column] = count['n_events'] / count['n_retained']
        metrics.to_csv(target / 'metrics.csv', index=False)
        io.atomic_json(done, signature)


def aggregate(a):
    import pandas as pd
    from scripts.run_ssl_ensembles_ecg1 import mean_predictions
    from biomecg.study import train_resnet_unified_ecg1 as shared
    if not all((folder(a, lead) / 'sensitivity_completed.json').is_file() for lead in LEADS):
        return
    frames = [pd.read_csv(folder(a, lead) / 'predictions.csv', dtype={'pid': str}) for lead in LEADS]
    result = mean_predictions(frames)
    target = folder(a, 'late')
    target.mkdir(parents=True, exist_ok=True)
    result.to_csv(target / 'predictions.csv', index=False)
    metrics = dict(site=a.site, seed=a.seed, cohort_label=a.cohort, family=a.family, model='late', n_test_patients=len(result))
    metrics.update(shared.calculate_metrics(result.label.to_numpy(), result.prediction.to_numpy(), 'test_patient'))
    pd.DataFrame([metrics]).to_csv(target / 'metrics.csv', index=False)
    (target / 'sensitivity_completed.json').write_text(json.dumps({'members': [str(folder(a, lead)) for lead in LEADS]}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, choices=[12345, 23451], default=12345)
    p.add_argument('--site', choices=['all', 'mimic', 'code15'], default='all')
    p.add_argument('--family', choices=['all', *FAMILIES], default='all')
    p.add_argument('--cohort', choices=['8000', 'full'])
    p.add_argument('--model', default='early_fusion')
    p.add_argument('--num-workers', type=int, default=16)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    if a.worker:
        worker(a)
        return
    for site in ['mimic', 'code15'] if a.site == 'all' else [a.site]:
        for family in FAMILIES if a.family == 'all' else [a.family]:
            for cohort in [a.cohort] if a.cohort else ['8000', 'full']:
                models = ['early_fusion', *LEADS] if family in ('biomecg', 'pclr', 'jepa') else list(LEADS) if family == 'ecgfounder' else ['native']
                for model in models:
                    cmd = [sys.executable, '-u', __file__, '--worker', '--site', site, '--family', family, '--seed', str(a.seed), '--cohort', cohort, '--model', model, '--num-workers', str(a.num_workers)]
                    if a.check_only:
                        cmd.append('--check-only')
                    subprocess.run(cmd, cwd=ROOT, check=True)
                if len(models) > 1 and not a.check_only:
                    aggregate(argparse.Namespace(**{**vars(a), 'site': site, 'family': family, 'cohort': cohort}))

if __name__ == '__main__':
    main()
