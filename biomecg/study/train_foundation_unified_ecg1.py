"""Shared ECG1 mortality engine for ECG-CPC, ECG-JEPA, MERL and ST-MEM.

Sources: https://github.com/AI4HealthUOL/ecg-fm-benchmarking (fm_ecg.py, run.sh)
https://arxiv.org/abs/2509.25095
Original models: https://github.com/sehunfromdaegu/ECG_JEPA
https://github.com/cheliu-computation/MERL-ICML2024 ; https://github.com/vuno/ST-MEM
CPC is the checkpoint introduced by the benchmark authors.

Protocol: existing ECG1 patient splits / nested cohorts / first ECG / labels,
MLP [features+age+sex,256,128,1], weighted BCE and validation-AUROC selection.
New models only: benchmark-native windows, raw mV (no extra normalization),
resampy resampling, three depth LR groups 0.01/0.1/1 * lr_head. Random training
window, four evenly spaced overlapping windows for evaluation, mean probabilities per
patient. Short input is symmetrically zero-padded (JEPA needs 10 seconds).
"""
import argparse
from contextlib import nullcontext
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import pandas as pd
import resampy
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from biomecg.study import train_resnet_unified_ecg1 as shared
from biomecg.study.foundation_models_ecg1 import (
    FoundationEncoder, FoundationMortalityModel, SPECS, provenance)

PROTOCOL = 'foundation_ecg1_v2_full_coverage'
REQUIRED_OUTPUTS = ('best.pt','metrics.csv','predictions.csv','predictions_patient.csv','completed.json')


def parse_args(family=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=list(SPECS), required=family is None, default=family)
    p.add_argument('--cpc-local', action='store_true', help='CPC pretrained locally on the same-source cohort1359')
    p.add_argument('--cpc-model',choices=['native','early_fusion',*shared.LEADS],default='native')
    p.add_argument('--cpc-training-seed',type=int)
    p.add_argument('--site', choices=['mimic','code15'], required=True)
    p.add_argument('--labels_path', type=Path, required=True)
    p.add_argument('--ecg_path', type=Path, required=True)
    p.add_argument('--horizon', type=int, default=365)
    p.add_argument('--cohort_size', default='full')
    p.add_argument('--architecture', choices=['pretrained_frozen','pretrained_finetuned'], default='pretrained_frozen')
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--lr_head', type=float, default=1e-3)
    p.add_argument('--bs', type=int, default=64, help='Effective training batch size')
    p.add_argument('--micro_batch', type=int, default=8, help='Physical batch; accumulate gradients to --bs')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--cpu_threads', type=int, default=2)
    p.add_argument('--device', default='cuda')
    p.add_argument('--input_fs', type=int, default=500)
    p.add_argument('--check_only', action='store_true')
    p.add_argument('--smoke_test', action='store_true', help='Real train/val/test rows, one step, no experiment outputs')
    a = p.parse_args()
    if family is not None and a.family != family:
        p.error('This entry point fixes the model family')
    if a.bs < a.micro_batch or a.bs % a.micro_batch or min(a.micro_batch,a.epochs,a.patience,a.input_fs,a.cpu_threads)<=0 or a.num_workers<0:
        p.error('Positive sizes required; bs must be a multiple of micro_batch')
    if a.cpc_local and a.family != 'ecgcpc':
        p.error('--cpc-local requires ecgcpc')
    if a.cpc_local and a.architecture != 'pretrained_frozen':
        p.error('Local CPC study is frozen-only; fine-tuning is disabled')
    if not a.cpc_local and (a.cpc_model != 'native' or a.cpc_training_seed is not None):
        p.error('CPC variants require --cpc-local')
    return a


def architecture_name(args):
    family = f'ecgcpc_local_{args.site}' if getattr(args,'cpc_local',False) else args.family
    return f"{family}_{args.architecture.removeprefix('pretrained_')}"


def make_encoder(args):
    if getattr(args,'cpc_local',False):
        from biomecg.study.cpc_local_ecg1 import load_encoder
        return load_encoder(args.site,args.seed,getattr(args,'cpc_model','native'),getattr(args,'cpc_training_seed',None))
    return FoundationEncoder(args.family)


def output_dir(args):
    if getattr(args,'cpc_local',False) and (getattr(args,'cpc_model','native') != 'native' or getattr(args,'cpc_training_seed',None) not in (None,args.seed)):
        training_seed=getattr(args,'cpc_training_seed',None) or args.seed
        return (shared.REFACTORED_ROOT/args.site/f'seed{args.seed}'/'runs'/f'horizon{args.horizon}d'/f'cohort{args.cohort_size}'/f'trainseed{training_seed}'/architecture_name(args)/args.cpc_model)
    return (shared.REFACTORED_ROOT / args.site / f'seed{args.seed}' / 'runs' /
            f'horizon{args.horizon}d' / f'cohort{args.cohort_size}' /
            architecture_name(args) / 'native')


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temp.replace(path)


def signature(args):
    result = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
              if k not in ('check_only','smoke_test','num_workers','cpu_threads','device')}
    if result.get('cpc_model') == 'native':result.pop('cpc_model')
    if result.get('cpc_training_seed') is None:result.pop('cpc_training_seed',None)
    # Absent/false flag must preserve signatures of historical external runs.
    if not result.get('cpc_local',False):result.pop('cpc_local',None)
    result['protocol'] = PROTOCOL
    if getattr(args,'cpc_local',False):
        from biomecg.study.cpc_local_ecg1 import provenance as local_provenance
        result.update(local_provenance(args.site,args.seed,getattr(args,'cpc_model','native'),getattr(args,'cpc_training_seed',None)))
    else:
        result.update(provenance(args.family))
    split = shared.split_paths(args)[0]
    if not split.is_file():
        raise FileNotFoundError(f'Existing ECG1 split required: {split}')
    result['split_sha256'] = hashlib.sha256(split.read_bytes()).hexdigest()
    for key in ('labels_path','ecg_path'):
        path = getattr(args,key).resolve()
        result[key] = str(path)
        result[key+'_stat'] = [path.stat().st_size, path.stat().st_mtime_ns]
    return result


def is_complete(args, expected=None):
    folder = output_dir(args)
    if not all((folder/n).is_file() and (folder/n).stat().st_size for n in REQUIRED_OUTPUTS):
        return False
    expected = signature(args) if expected is None else expected
    if json.loads((folder/'completed.json').read_text()) != expected:
        raise ValueError(f'Completed run has a different configuration: {folder}')
    return True


class FoundationDataset(Dataset):
    def __init__(self, labels, path, rows, family, training, input_fs=500):
        self.path, self.rows = str(path), np.asarray(rows,dtype=int)
        self.events = labels['_event'].to_numpy(dtype=np.float32)
        self.metadata = labels[['Age','Is_Male']].to_numpy(dtype=np.float32)
        self.metadata[:,0] /= 100.
        self.family, self.training, self.input_fs = family, training, input_fs
        self.file = None
        self.raw_signals = None
        with h5py.File(path,'r') as f:
            if f['ECG'].shape[1] != 12:
                raise ValueError('Native foundation models require the 12 stored leads')
            if int(f.attrs.get('sampling_rate',input_fs)) != input_fs:
                raise ValueError('Input sampling rate mismatch')

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy(); state['file'] = None
        return state

    def close(self):
        if self.file is not None:
            self.file.close(); self.file=None

    def __del__(self):
        self.close()

    def __getitem__(self, item):
        row = int(self.rows[item])
        if self.raw_signals is None:
            if self.file is None:
                self.file = h5py.File(self.path,'r')
            x = self.file['ECG'][row].astype(np.float32)
        else:
            x = self.raw_signals[item].copy()
        if hasattr(self,'lead_indices'):x=x[self.lead_indices]
        if not np.isfinite(x).all() or not np.isfinite(self.metadata[row]).all():
            raise ValueError(f'Non-finite input at row {row}')
        spec = SPECS[self.family]
        # Crop at source rate before antialiased resampling, as in benchmark.
        source_length = round(spec['samples'] * self.input_fs / spec['fs'])
        if x.shape[-1] < source_length:
            missing = source_length-x.shape[-1]
            x = np.pad(x, ((0,0),(missing//2,missing-missing//2)))
        if self.training:
            starts = [np.random.randint(x.shape[-1]-source_length+1)]
        else:
            starts = ([0] if x.shape[-1] == source_length else
                      np.rint(np.linspace(0, x.shape[-1]-source_length, 4)).astype(int).tolist())
        windows = []
        for start in starts:
            part = x[:,start:start+source_length]
            if spec['fs'] != self.input_fs:
                part = resampy.resample(part, self.input_fs, spec['fs'], axis=-1)
            if part.shape[-1] != spec['samples']:
                raise ValueError(f'Unexpected resampled shape {part.shape}')
            windows.append(np.ascontiguousarray(part,dtype=np.float32))
        signal = windows[0] if self.training else np.stack(windows)
        return torch.from_numpy(signal), torch.from_numpy(self.metadata[row]), torch.tensor(self.events[row]), torch.tensor(row)


def cache_training_signals(dataset):
    """Immutable raw ECGs in CPU RAM; preserve random crop-before-resampling."""
    with h5py.File(dataset.path,'r') as handle:
        shape=(len(dataset),*handle['ECG'].shape[1:])
        needed=int(np.prod(shape))*4
        available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith('MemAvailable:'))
        # Two local-CPC jobs fit within 128GB, retaining >=20GB for other work.
        if needed>24*2**30 or available-needed<20*2**30:
            print(f'RAM train cache skipped: need {needed/2**30:.1f} GiB; available {available/2**30:.1f} GiB',flush=True)
            return False
        values=np.empty(shape,dtype=np.float32)
        for start in tqdm(range(0,len(dataset),128),desc='Raw train ECGs to RAM'):
            rows=dataset.rows[start:start+128]
            order=np.argsort(rows)
            values[start:start+len(rows)] = handle['ECG'][rows[order]][np.argsort(order)]
        values.flags.writeable=False
        dataset.raw_signals=values
    print(f'RAM train cache: {len(dataset)} ECGs, {needed/2**30:.2f} GiB; random crops unchanged',flush=True)
    return True


def loader(labels,args,rows,training):
    dataset=FoundationDataset(labels,args.ecg_path,rows,args.family,training,args.input_fs)
    if getattr(args,'cpc_local',False):
        from biomecg.study.cpc_local_ecg1 import lead_indices
        dataset.lead_indices=lead_indices(getattr(args,'cpc_model','native'))
    ram=bool(training and getattr(args,'cpc_local',False) and args.architecture=='pretrained_frozen')
    if ram:ram=cache_training_signals(dataset)
    # Linux fork shares the immutable cache, rather than copying 20GB per worker.
    options=dict(multiprocessing_context='fork') if ram and args.num_workers else {}
    return DataLoader(dataset,batch_size=args.micro_batch,shuffle=training,num_workers=args.num_workers,
        persistent_workers=bool(ram and args.num_workers),pin_memory=str(args.device).startswith('cuda'),**options)


def autocast(args, device):
    # S4 complex arithmetic stays float32, as specified in benchmark run.sh.
    return torch.autocast(device.type,dtype=torch.bfloat16) if device.type=='cuda' and args.family!='ecgcpc' else nullcontext()


@torch.no_grad()
def predict(model, dataloader, args, device, description):
    model.eval()
    # CPU cache scoped to this loader/run: no stale checkpoint or split reuse.
    # Keep individual windows: pooling remains mean(sigmoid(head(window))).
    cached = getattr(dataloader, '_frozen_features', None) if model.frozen else None
    if cached is not None and dataloader._cache_encoder is not model.encoder:
        raise ValueError('Evaluation cache belongs to another encoder')
    building = model.frozen and cached is None
    feature_batches, metadata_batches = [], []
    iterable = (DataLoader(cached, batch_size=dataloader.batch_size, shuffle=False,
                           num_workers=0) if cached is not None else dataloader)
    probabilities, targets, indices = [], [], []
    for windows, metadata, labels, rows in tqdm(iterable,desc=description,leave=False):
        if building:
            metadata_batches.append(metadata.cpu())
        metadata=metadata.to(device)
        window_scores=[]
        embeddings=[]
        for i in range(windows.shape[1]):
            with autocast(args,device):
                if model.frozen:
                    features = (windows[:,i].to(device) if cached is not None else
                                model.encoder(windows[:,i].to(device)))
                    logits = model.head(torch.cat([features,metadata],dim=1)).squeeze(1)
                else:
                    logits=model(windows[:,i].to(device),metadata)
            if building:
                embeddings.append(features.cpu())
            window_scores.append(logits.float().sigmoid())
        scores=torch.stack(window_scores).mean(0)
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Non-finite predictions')
        if building:
            feature_batches.append(torch.stack(embeddings,dim=1))
        probabilities.append(scores.cpu().numpy());targets.append(labels.numpy());indices.append(rows.numpy())
    if building:
        features = torch.cat(feature_batches)
        dataloader._frozen_features = torch.utils.data.TensorDataset(
            features, torch.cat(metadata_batches), torch.from_numpy(np.concatenate(targets)),
            torch.from_numpy(np.concatenate(indices)))
        dataloader._cache_encoder = model.encoder
        print(f'Frozen evaluation cache: {len(features)} ECGs, '
              f'{features.numel()*features.element_size()/2**20:.1f} MiB CPU',flush=True)
    return np.concatenate(probabilities),np.concatenate(targets).astype(int),np.concatenate(indices)


def save_weights(path, payload):
    temp=path.with_suffix('.tmp')
    torch.save(payload,temp);temp.replace(path)


def train_model(model, train_loader, val_loader, events, args, device, folder):
    positives=int(events.sum()); negatives=len(events)-positives
    if not positives or not negatives:
        raise ValueError('Both classes required in the training cohort')
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives/positives],device=device),reduction='sum')
    groups=model.parameter_groups(args.lr_head)
    print('Learning rates:',[(g['name'],g['lr']) for g in groups],flush=True)
    optimizer=torch.optim.AdamW(groups,weight_decay=0.01) # historical ECG1 default
    best=-np.inf;wait=0
    accumulation=args.bs//args.micro_batch
    history=[]
    for epoch in range(args.epochs):
        model.train();optimizer.zero_grad(set_to_none=True)
        total_loss=0.; group_count=0;seen=0
        for i,(x,metadata,y,_) in enumerate(train_loader):
            x=x.to(device);metadata=metadata.to(device);y=y.to(device)
            with autocast(args,device):
                loss=criterion(model(x,metadata),y)
            if not torch.isfinite(loss):raise FloatingPointError(f'Non-finite loss at epoch {epoch}')
            loss.backward();group_count+=len(y);seen+=len(y);total_loss+=loss.item()
            if (i+1)%accumulation==0 or i+1==len(train_loader):
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(group_count)
                        if not torch.isfinite(p.grad).all():raise FloatingPointError('Non-finite gradient')
                optimizer.step();optimizer.zero_grad(set_to_none=True);group_count=0
        scores,labels,_=predict(model,val_loader,args,device,f'Validation epoch {epoch}')
        metrics=shared.calculate_metrics(labels,scores,'validation')
        auc=float(metrics['validation_auc'])
        record=dict(epoch=epoch,train_loss=total_loss/seen,**metrics)
        history.append(record);pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        print(f'epoch={epoch:03d} train_loss={total_loss/seen:.5f} val_auc={auc:.5f}',flush=True)
        if auc>best:
            best=auc;wait=0
            save_weights(folder/'best.pt',dict(model_state_dict=model.state_dict(),epoch=epoch,
                validation_auc=auc,validation_pr_auc=metrics['validation_pr_auc']))
        else:wait+=1
        if wait>=args.patience:break
    checkpoint=torch.load(folder/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    return checkpoint


def main(family=None):
    args=parse_args(family)
    os.environ['OMP_NUM_THREADS']=str(args.cpu_threads);os.environ['MKL_NUM_THREADS']=str(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads);shared.set_seed(args.seed)
    expected=signature(args)
    if is_complete(args,expected):
        print(f'ALREADY COMPLETE | {output_dir(args)}');return
    labels=shared.load_labels(args.labels_path,args.ecg_path,args.horizon)
    split=shared.split_paths(args)[0]
    train_order,val_patients,test_patients=shared.load_cached_split(split,labels)
    n=len(train_order) if str(args.cohort_size).lower()=='full' else int(args.cohort_size)
    if not 0<n<=len(train_order):raise ValueError('Invalid cohort size')
    patient_groups=[train_order[:n],val_patients,test_patients]
    rows=[shared.rows_for_patients(labels,p) for p in patient_groups]
    if args.check_only:
        print(f'READY | {args.family} {args.architecture} seed{args.seed} cohort{n}');return
    device=torch.device(args.device)
    folder=output_dir(args)
    # Smoke tests never create production output folders.
    if args.smoke_test:
        smoke(args,labels,rows,device);return
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'run.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError(f'Already running: {folder}')
        if is_complete(args,expected):return
        config=folder/'run_config.json'
        if config.is_file() and json.loads(config.read_text())!=expected:
            raise ValueError(f'Incomplete run configuration differs: {folder}')
        atomic_json(config,expected)
        model=FoundationMortalityModel(make_encoder(args),args.architecture.endswith('frozen')).to(device)
        loaders=[loader(labels,args,r,i==0) for i,r in enumerate(rows)]
        checkpoint=train_model(model,loaders[0],loaders[1],labels.iloc[rows[0]]['_event'].to_numpy(),args,device,folder)
        scores,targets,order=predict(model,loaders[2],args,device,'Test')
        predictions=pd.DataFrame(dict(row_index=order,pid=labels.iloc[order]['_pid'].to_numpy(),
            ecg_order=labels.iloc[order]['ECG_Order'].to_numpy(),label=targets,prediction=scores)).sort_values('row_index')
        if predictions.pid.duplicated().any():raise ValueError('Duplicate test patients')
        predictions.to_csv(folder/'predictions.csv',index=False)
        predictions.assign(n_ecgs=1).to_csv(folder/'predictions_patient.csv',index=False)
        metrics=dict(site=args.site,seed=args.seed,horizon_days=args.horizon,cohort_label=args.cohort_size,
            architecture=architecture_name(args),model=getattr(args,'cpc_model','native'),
            config_id=args.family,lr_head=args.lr_head,lr_encoder_early=0 if model.frozen else args.lr_head*.01,
            lr_encoder=0 if model.frozen else args.lr_head*.1,batch_size=args.bs,micro_batch=args.micro_batch,
            best_epoch=checkpoint['epoch'],best_val_auc_ecg=checkpoint['validation_auc'],
            n_train_patients=n,n_val_patients=len(val_patients),n_test_patients=len(test_patients),
            n_train_ecgs=n,n_val_ecgs=len(val_patients),n_test_ecgs=len(test_patients),
            patient_reference_ecg='first_min_ECG_Order',split_path=str(split),
            protocol=PROTOCOL,input_fs=args.input_fs,model_fs=SPECS[args.family]['fs'],
            window_samples=SPECS[args.family]['samples'],normalization='none_raw_mV',
            eval_pooling='mean_window_probabilities',**{k:expected[k] for k in
              ('pretraining_checkpoint','checkpoint_sha256','benchmark_commit')})
        for prefix in ('test_ecg','test_patient'):
            metrics.update(shared.calculate_metrics(targets,scores,prefix))
        pd.DataFrame([metrics]).to_csv(folder/'metrics.csv',index=False)
        atomic_json(folder/'completed.json',expected)
        for dl in loaders:dl.dataset.close()
        print(pd.Series(metrics).to_string())


def smoke(args,labels,rows,device):
    model=FoundationMortalityModel(make_encoder(args),args.architecture.endswith('frozen')).to(device)
    model.train()
    before={name:p.detach().clone() for name,p in model.encoder.named_parameters()} if model.frozen else None
    ds=FoundationDataset(labels,args.ecg_path,rows[0][:args.micro_batch],args.family,True,args.input_fs)
    if getattr(args,'cpc_local',False):
        from biomecg.study.cpc_local_ecg1 import lead_indices
        ds.lead_indices=lead_indices(getattr(args,'cpc_model','native'))
    x,meta,y,_=next(iter(DataLoader(ds,batch_size=args.micro_batch)))
    groups=model.parameter_groups(args.lr_head);optimizer=torch.optim.AdamW(groups)
    with autocast(args,device):
        out=model(x.to(device),meta.to(device));loss=nn.functional.binary_cross_entropy_with_logits(out,y.to(device))
    if not torch.isfinite(loss):raise FloatingPointError('Smoke loss is non-finite')
    loss.backward()
    if not model.frozen:
        for group in groups:
            if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in group['params']):
                raise AssertionError(f'No finite gradient in {group["name"]}')
    optimizer.step()
    if model.frozen:
        for name,p in model.encoder.named_parameters():
            if p.grad is not None or not torch.equal(before[name],p):raise AssertionError('Frozen weights changed')
    ds.close()
    for name,r in zip(('Validation smoke','Test smoke'),rows[1:]):
        dl=loader(labels,args,r[:2],False)
        scores,_,_=predict(model,dl,args,device,name)
        if len(scores)!=2:raise AssertionError('Incorrect prediction count')
        dl.dataset.close()
    if device.type == 'cuda':
        print(f'GPU peak allocated: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB',flush=True)
    print(f'SMOKE OK {args.family} {args.architecture} loss={loss.item():.6f}',flush=True)


if __name__=='__main__':main()
