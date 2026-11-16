"""One tmux batch per split seed: JEPA-ResNet pretraining + same-source mortality grid.
Sources: https://arxiv.org/html/2605.12241v1 and
https://github.com/AI4HealthUOL/ecg-pretraining-strategies
Adaptations: representations/mimic/README_jepa_resnet_ecg1.md.
Eight-lead early fusion only. MIMIC and CODE-15 each reuse their own biometric
cohort1359 and mortality splits. Historical biomECG/ECGFounder never launched.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from biomecg.study import pclr_ecg1 as io
from biomecg.study import jepa_resnet_ecg1 as p


def commands(seed,sites,workers,check_only=False):
    for site in sites:
        yield [sys.executable,'-u',str(ROOT/'biomecg/study/train_jepa_resnet_ecg1.py'),
            '--site',site,'--seed',str(seed),'--num-workers',str(workers)]+(['--check-only'] if check_only else [])
        if check_only:continue
        source=io.source_config(site,seed)
        paths=source['dir_input'][source['run_from']][source['DB']]
        for cohort in ['1000','8000','full']:
            yield [sys.executable,'-u',str(ROOT/'biomecg/study/train_jepa_resnet_unified_ecg1.py'),
                '--site',site,'--seed',str(seed),'--labels_path',paths['labels'],'--ecg_path',paths['ecg'],
                '--horizon','365','--cohort_size',cohort,'--architecture',f'jepa_resnet_early_{site}_frozen',
                '--model','early_fusion','--lr_encoder','0',
                '--lr_head','0.001','--bs','256','--epochs','200','--patience','20',
                '--num_workers',str(workers),'--cpu_threads','2']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True,choices=p.SEEDS)
    parser.add_argument('--site',choices=['all','mimic','code15'],default='all')
    parser.add_argument('--num-workers',type=int,default=2)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.num_workers<0:parser.error('num-workers must be nonnegative')
    sites=['mimic','code15'] if args.site=='all' else [args.site]
    # Validate every site's manifest/split before any expensive work.
    from biomecg.study import train_resnet_unified_ecg1 as shared
    for site in sites:
        p.pretrain_signature(site,args.seed)
        config=io.source_config(site,args.seed);source=config['dir_input'][config['run_from']][config['DB']]
        labels=shared.load_labels(Path(source['labels']),Path(source['ecg']),365)
        from types import SimpleNamespace
        order,_,_=shared.load_cached_split(shared.split_paths(SimpleNamespace(site=site,seed=args.seed,horizon=365))[0],labels)
        if len(order)<8000:raise ValueError('Insufficient downstream patients')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for command in commands(args.seed,sites,args.num_workers,args.check_only):
        print('RUN: '+' '.join(command),flush=True)
        subprocess.run(command,cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()
