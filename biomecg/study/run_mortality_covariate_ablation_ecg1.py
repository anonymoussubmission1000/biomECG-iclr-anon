"""One tmux batch per seed: age/sex-only and frozen ECG-only, MIMIC + CODE-15.
Uses the historical biomECG cohort1359 and mortality MLP/training protocol.
See representations/mimic/README_mortality_covariate_ablation_ecg1.md.
"""
import argparse
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from biomecg.study import mortality_covariate_ablation_ecg1 as a


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=a.SEEDS,required=True)
    parser.add_argument('--site',choices=['all','mimic','code15'],default='all')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--num-workers',type=int,default=4,help='Workers for one-time ECG extraction; cached MLP uses zero')
    parser.add_argument('--cpu-threads',type=int,default=2)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.num_workers<0 or args.cpu_threads<1:parser.error('Invalid CPU settings')
    os.environ['OMP_NUM_THREADS']=str(args.cpu_threads);os.environ['MKL_NUM_THREADS']=str(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)
    device=torch.device(args.device)
    if not args.check_only and device.type=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    sites=['mimic','code15'] if args.site=='all' else [args.site]
    contexts=[a.prepare_context(site,args.seed,args.device,args.num_workers,args.cpu_threads) for site in sites]
    if args.check_only:
        print('CHECK OK: no runs launched. Per site: 3 covariate MLPs + 24 ECG-only frozen MLPs.');return
    for context in contexts:a.run_context(context,device)

if __name__=='__main__':main()
