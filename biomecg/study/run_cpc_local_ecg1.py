"""One tmux batch per seed: local CPC pretraining + frozen-only mortality, MIMIC then CODE15.
Sources: biomecg/study/cpc_local_ecg1.py and AUDIT_CPC_PRETRAINING.md.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from biomecg.study import pclr_ecg1 as cohort
from biomecg.study import cpc_local_ecg1 as c


def commands(seed,sites,workers,check_only=False):
    for site in sites:
        yield [sys.executable,'-u',str(ROOT/'biomecg/study/train_cpc_local_ecg1.py'),
               '--site',site,'--seed',str(seed),'--num-workers',str(workers)]+(['--check-only'] if check_only else [])
        if check_only:continue
        config=cohort.source_config(site,seed)
        paths=config['dir_input'][config['run_from']][config['DB']]
        for size in ['1000','8000','full']:
            for mode in ['pretrained_frozen']:
                yield [sys.executable,'-u',str(ROOT/'biomecg/study/train_foundation_unified_ecg1.py'),
                    '--family','ecgcpc','--cpc-local','--site',site,'--seed',str(seed),
                    '--labels_path',paths['labels'],'--ecg_path',paths['ecg'],
                    '--horizon','365','--cohort_size',size,'--architecture',mode,
                    '--num_workers',str(workers),'--cpu_threads','2']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=cohort.SEEDS,required=True)
    parser.add_argument('--site',choices=['all','mimic','code15'],default='all')
    parser.add_argument('--num-workers',type=int,default=2)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.num_workers<0:parser.error('num-workers must be nonnegative')
    sites=['mimic','code15'] if args.site=='all' else [args.site]
    # Validate both cohorts and mortality splits before beginning GPU work.
    from biomecg.study import train_resnet_unified_ecg1 as shared
    from types import SimpleNamespace
    for site in sites:
        c.signature(site,args.seed)
        config=cohort.source_config(site,args.seed);paths=config['dir_input'][config['run_from']][config['DB']]
        labels=shared.load_labels(Path(paths['labels']),Path(paths['ecg']),365)
        order,_,_=shared.load_cached_split(shared.split_paths(SimpleNamespace(site=site,seed=args.seed,horizon=365))[0],labels)
        if len(order)<8000:raise ValueError('Insufficient mortality train patients')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for command in commands(args.seed,sites,args.num_workers,args.check_only):
        print('RUN: '+' '.join(command),flush=True)
        subprocess.run(command,cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()
