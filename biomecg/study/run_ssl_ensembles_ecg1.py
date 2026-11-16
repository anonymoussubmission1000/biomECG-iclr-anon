"""PCLR / JEPA: eight single leads (late) and eight early seeds (ensemble control).

Sources and unchanged objective recipes:
  representations/mimic/{pclr_ecg1,jepa_resnet_ecg1}.py
  https://github.com/mit-ccrg/PCLR
  https://github.com/AI4HealthUOL/ecg-pretraining-strategies
Protocol: same saved cohort1359 and mortality split; frozen 1k/8k/Full,
identical 128D ECG + age/sex MLP. Mean test probabilities, never mean weights.
The original early member is reused, including its historical output path.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from biomecg.study import pclr_ecg1 as p
from biomecg.study import train_resnet_unified_ecg1 as shared

COHORTS=('1000','8000','full')


def members(seed,variant):
    if variant=='late': return [(lead,seed) for lead in p.LEADS]
    if variant=='early8': return [('early_fusion',s) for s in [seed]+[int(f'{seed}{i}') for i in range(7)]]
    raise ValueError(variant)


def architecture(family,site):
    return f'pclr_{site}_frozen' if family=='pclr' else f'jepa_resnet_early_{site}_frozen'


def downstream_args(family,site,seed,model,training_seed,cohort):
    return SimpleNamespace(site=site,seed=seed,horizon=365,cohort_size=cohort,
        architecture=architecture(family,site),model=model,mimic_pretraining_size=1359,
        pretraining_training_seed=None if training_seed==seed else training_seed)


def commands(family,site,seed,variant,workers):
    stem='pclr' if family=='pclr' else 'jepa_resnet'
    config=p.source_config(site,seed)
    paths=config['dir_input'][config['run_from']][config['DB']]
    for model,training_seed in members(seed,variant):
        extra=[] if training_seed==seed else ['--training-seed',str(training_seed)]
        yield [sys.executable,'-u',str(ROOT/f'biomecg/study/train_{stem}_ecg1.py'),
            '--site',site,'--seed',str(seed),'--model',model,'--num-workers',str(workers),*extra]
        extra=[] if training_seed==seed else ['--pretraining_training_seed',str(training_seed)]
        for cohort in COHORTS:
            yield [sys.executable,'-u',str(ROOT/f'biomecg/study/train_{stem}_unified_ecg1.py'),
                '--site',site,'--seed',str(seed),'--labels_path',paths['labels'],'--ecg_path',paths['ecg'],
                '--horizon','365','--cohort_size',cohort,'--architecture',architecture(family,site),
                '--model',model,'--lr_encoder','0','--lr_head','0.001','--bs','256',
                '--epochs','200','--patience','20','--num_workers',str(workers),'--cpu_threads','2',*extra]


def mean_predictions(frames):
    import numpy as np
    import pandas as pd
    if len(frames)!=8: raise ValueError('Exactly eight completed members required')
    aligned=[]
    identity=['pid','row_index','ecg_order','label']
    for frame in frames:
        frame=frame.sort_values('pid').reset_index(drop=True)
        if frame.empty or frame.pid.duplicated().any(): raise ValueError('Empty/duplicate test patients')
        if not np.isfinite(frame.prediction).all() or not frame.prediction.between(0,1).all():
            raise ValueError('Invalid probabilities')
        if aligned: pd.testing.assert_frame_equal(frame[identity],aligned[0][identity])
        aligned.append(frame)
    result=aligned[0].copy()
    result['prediction']=np.mean([f.prediction.to_numpy() for f in aligned],axis=0)
    return result


def aggregate(family,site,seed,variant,cohort):
    import pandas as pd
    frames=[];sources=[]
    for model,training_seed in members(seed,variant):
        args=downstream_args(family,site,seed,model,training_seed,cohort)
        folder=shared.output_dir(args)
        for name in ('completed.json','metrics.csv','predictions.csv','predictions_patient.csv','best.pt'):
            if not (folder/name).is_file() or not (folder/name).stat().st_size:
                raise FileNotFoundError(folder/name)
        marker=json.loads((folder/'completed.json').read_text())
        for key in ('site','seed','model','architecture','cohort_size','pretraining_training_seed'):
            if marker.get(key)!=getattr(args,key): raise ValueError(f'Member provenance mismatch: {folder}/{key}')
        frames.append(pd.read_csv(folder/'predictions.csv',dtype={'pid':str}))
        sources.append(dict(path=str(folder),predictions_sha256=p.sha256(folder/'predictions.csv'),
            completion_sha256=p.sha256(folder/'completed.json')))
    result=mean_predictions(frames)
    folder=shared.REFACTORED_ROOT/site/f'seed{seed}'/'ensembles'/'horizon365d'/f'cohort{cohort}'/architecture(family,site)/variant
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        marker=folder/'completed.json'
        expected=dict(protocol='ssl_frozen_ensemble_v1',members=sources,aggregation='mean probabilities')
        if marker.exists() and json.loads(marker.read_text())==expected and all((folder/n).is_file() for n in ('predictions.csv','predictions_patient.csv','metrics.csv')):
            print('SKIP ENSEMBLE',folder,flush=True);return
        marker.unlink(missing_ok=True)
        result.to_csv(folder/'predictions.csv',index=False)
        result.assign(n_ecgs=1).to_csv(folder/'predictions_patient.csv',index=False)
        metrics=dict(site=site,seed=seed,horizon_days=365,cohort_label=cohort,
            architecture=architecture(family,site),model=variant,n_members=8,n_test_patients=len(result))
        for prefix in ('test_ecg','test_patient'):
            metrics.update(shared.calculate_metrics(result.label.to_numpy(),result.prediction.to_numpy(),prefix))
        pd.DataFrame([metrics]).to_csv(folder/'metrics.csv',index=False)
        p.atomic_json(folder/'members.json',sources)
        p.atomic_json(marker,expected)
        print('ENSEMBLE',folder,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=['pclr','jepa','all'],default='all')
    parser.add_argument('--seed',type=int,choices=p.SEEDS,required=True)
    parser.add_argument('--site',choices=['mimic','code15','all'],default='all')
    parser.add_argument('--variant',choices=['late','early8','all'],default='all')
    parser.add_argument('--num-workers',type=int,default=2)
    parser.add_argument('--check-only',action='store_true',help='Validate existing manifests/splits and print the grid; no training')
    parser.add_argument('--aggregate-only',action='store_true')
    args=parser.parse_args()
    if args.num_workers<0: parser.error('num-workers must be nonnegative')
    families=['pclr','jepa'] if args.family=='all' else [args.family]
    sites=['mimic','code15'] if args.site=='all' else [args.site]
    variants=['late','early8'] if args.variant=='all' else [args.variant]
    # Validate both sites before launching anything; never construct new splits.
    for site in sites:
        _,_,_,source=p.context(site,args.seed)
        labels=shared.load_labels(Path(source['labels']),Path(source['ecg']),365)
        order,_,_=shared.load_cached_split(shared.split_paths(SimpleNamespace(site=site,seed=args.seed,horizon=365))[0],labels)
        if len(order)<8000: raise ValueError('Insufficient mortality patients')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for family in families:
        for site in sites:
            for variant in variants:
                if not args.aggregate_only:
                    for command in commands(family,site,args.seed,variant,args.num_workers):
                        print(('CHECK ' if args.check_only else 'RUN ')+ ' '.join(command),flush=True)
                        if not args.check_only: subprocess.run(command,cwd=ROOT,env=env,check=True)
                if not args.check_only:
                    for cohort in COHORTS: aggregate(family,site,args.seed,variant,cohort)

if __name__=='__main__':main()
