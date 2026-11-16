"""Pretrain PCLR on the historical MIMIC/CODE-15 cohort1359 manifests.
Sources/protocol: biomecg/study/pclr_ecg1.py and README_pclr_ecg1.md.
"""
import argparse
import fcntl
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from biomecg.study import pclr_ecg1 as p
from biomecg.study.train_resnet_unified_ecg1 import set_seed


def epoch_loss(model,loader,device,optimizer=None):
    model.train(optimizer is not None);total=0.;count=0
    with torch.set_grad_enabled(optimizer is not None):
        for first,second,indices in loader:
            if len(first)<2:raise ValueError('A batch needs at least two distinct patients')
            if len(set(indices.tolist()))!=len(indices):raise ValueError('Repeated patient in batch')
            x=torch.cat([first,second],0).to(device,non_blocking=True)
            if optimizer is not None:optimizer.zero_grad(set_to_none=True)
            loss=p.contrastive_loss(model(x))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PCLR loss')
            if optimizer is not None:loss.backward();optimizer.step()
            total+=loss.item()*len(first);count+=len(first)
    return total/count,count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site',choices=['mimic','code15'],required=True)
    parser.add_argument('--seed',type=int,choices=p.SEEDS,required=True)
    parser.add_argument('--num-workers',type=int,default=4)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--model',choices=['early_fusion',*p.LEADS],default='early_fusion')
    parser.add_argument('--training-seed',type=int)
    args=parser.parse_args()
    training_seed=p.variant_seed(args.seed,args.training_seed)
    if args.num_workers<0:parser.error('num-workers must be nonnegative')
    torch.set_num_threads(2);set_seed(training_seed)
    expected=p.pretrain_signature(args.site,args.seed,args.model,args.training_seed)
    config,manifest,path,source=p.context(args.site,args.seed)
    print(f'PCLR {args.site}/seed{args.seed}: '+', '.join(f'{g}={len(manifest[g+"_patients"])}' for g in ['train','val']),flush=True)
    folder=p.pretrain_dir(args.site,args.seed,args.model,args.training_seed)
    if args.check_only:
        if (folder/'completed.json').exists() and json.loads((folder/'completed.json').read_text())!=expected:
            raise ValueError('Existing PCLR config differs')
        print('CHECK OK',flush=True);return
    device=torch.device(args.device)
    if args.smoke_test:
        ds=p.PatientViews(manifest,source,'train',training_seed,args.model)
        batch=next(iter(DataLoader(ds,batch_size=8,num_workers=0)))
        model=p.PCLRModel(8 if args.model=='early_fusion' else 1).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=.1,eps=1e-7)
        result=epoch_loss(model,[batch],device,optimizer)
        result_val=epoch_loss(model,[batch],device)
        print(f'SMOKE OK train={result} eval={result_val}',flush=True);return
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        marker=folder/'completed.json';saved=folder/'run_config.json'
        if saved.exists() and json.loads(saved.read_text())!=expected:raise ValueError('Existing PCLR config differs')
        if marker.exists():
            if json.loads(marker.read_text())!=expected or not (folder/'network.pt').is_file():
                raise ValueError('Invalid PCLR completion marker')
            print(f'SKIP {folder}',flush=True);return
        p.atomic_json(saved,expected)
        datasets={g:p.PatientViews(manifest,source,g,training_seed+(g=='val'),args.model) for g in ['train','val']}
        if any(len(ds.patients)%8==1 for ds in datasets.values()):
            raise ValueError('Singleton final batch: define a policy before changing cohort size')
        model=p.PCLRModel(8 if args.model=='early_fusion' else 1).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=.1,betas=(.9,.999),eps=1e-7,weight_decay=0.)
        start=0;best=float('inf');history=[]
        latest=folder/'latest.pt'
        if latest.exists():
            state=torch.load(latest,map_location=device,weights_only=False)
            if state['signature']!=expected:raise ValueError('Resume signature mismatch')
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            start=state['epoch']+1;best=state['best'];history=state['history']
            torch.set_rng_state(state['rng'].cpu())
            if device.type=='cuda':torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
            print(f'RESUME epoch {start+1}',flush=True)
        for epoch in range(start,50):
            lr=.1*(1+math.cos(math.pi*epoch/50))/2
            for group in optimizer.param_groups:group['lr']=lr
            datasets['train'].epoch=epoch;datasets['val'].epoch=0
            loaders={g:DataLoader(ds,batch_sampler=p.PatientPassBatchSampler(ds,g=='train'),
                num_workers=args.num_workers,pin_memory=device.type=='cuda',persistent_workers=False,
                generator=torch.Generator().manual_seed(training_seed+epoch if g=='train' else training_seed))
                for g,ds in datasets.items()}
            train,n=epoch_loss(model,loaders['train'],device,optimizer)
            val,nval=epoch_loss(model,loaders['val'],device)
            if n!=len(datasets['train']) or nval!=len(datasets['val']):raise AssertionError('Patient omitted')
            history.append(dict(epoch=epoch+1,lr=lr,train_loss=train,val_loss=val,
                train_patients=len(datasets['train'].patients),patient_visits=n,train_steps=len(loaders['train']),positive_pairs=n))
            if val<best:
                best=val;p.atomic_save(folder/'network.pt',dict(model=model.state_dict(),signature=expected,epoch=epoch+1,val_loss=val))
            p.atomic_save(latest,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                best=best,history=history,signature=expected,rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []))
            p.atomic_json(folder/'history.json',history)
            print(f'epoch={epoch+1:02d}/50 steps={len(loaders["train"])} patients={n} lr={lr:.6g} train={train:.5f} val={val:.5f} best={best:.5f}',flush=True)
        if not (folder/'network.pt').is_file():raise RuntimeError('Missing best checkpoint')
        p.atomic_json(marker,expected)
        print(f'COMPLETE {folder}',flush=True)

if __name__=='__main__':main()
