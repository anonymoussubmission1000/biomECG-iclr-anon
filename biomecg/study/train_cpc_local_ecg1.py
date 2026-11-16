"""Local ECG-CPC pretraining; source links and recipe in cpc_local_ecg1.py.
Reuses biomECG cohort1359, not its pair labels or mortality labels.
"""
import argparse
import fcntl
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from torch.utils.data import DataLoader
from biomecg.study import cpc_local_ecg1 as c
from biomecg.study import pclr_ecg1 as io
from biomecg.study.train_resnet_unified_ecg1 import set_seed


def epoch_loss(model,loader,device,optimizer=None,validation_seed=0):
    model.train(optimizer is not None);total=0.;seen=0
    generator=None if optimizer is not None else torch.Generator(device=device).manual_seed(validation_seed)
    with torch.set_grad_enabled(optimizer is not None):
        for x in loader:
            x=x.to(device,non_blocking=True)
            if optimizer is not None:optimizer.zero_grad(set_to_none=True)
            pred,encoded=model(x);loss=c.cpc_loss(pred,encoded,generator=generator)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite CPC loss')
            if optimizer is not None:
                loss.backward()
                if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError('Non-finite CPC gradient')
                optimizer.step()
            total+=loss.item()*len(x);seen+=len(x)
    return total/seen,seen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site',choices=['mimic','code15'],required=True)
    parser.add_argument('--seed',type=int,choices=io.SEEDS,required=True)
    parser.add_argument('--num-workers',type=int,default=2)
    parser.add_argument('--model',choices=['native','early_fusion',*io.LEADS],default='native')
    parser.add_argument('--training-seed',type=int)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args()
    if args.num_workers<0:parser.error('num-workers must be nonnegative')
    torch.set_num_threads(2)
    training_seed=args.seed if args.training_seed is None else args.training_seed
    set_seed(training_seed)
    expected=c.signature(args.site,args.seed,args.model,args.training_seed);folder=c.pretrain_dir(args.site,args.seed,args.model,args.training_seed)
    _,manifest,_,source=io.context(args.site,args.seed)
    print('CPC local',args.site,args.seed,{g:sum(len(manifest['valid_sids'][pid]) for pid in manifest[g+'_patients']) for g in ['train','val']},flush=True)
    marker=folder/'completed.json'
    if marker.exists():
        if json.loads(marker.read_text())!=expected or not (folder/'network.pt').is_file():raise ValueError('Invalid completion')
        print(f'SKIP {folder}',flush=True);return
    if args.check_only:print('CHECK OK',flush=True);return
    device=torch.device(args.device)
    if args.smoke_test:
        ds=c.SourceECGs(manifest,source,'train')
        x=next(iter(DataLoader(ds,batch_size=2,num_workers=0)))[:,c.lead_indices(args.model)];ds.close()
        model=(c.CPCModel() if args.model == 'native' else c.CPCModel(len(c.lead_indices(args.model)))).to(device);optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=.001)
        train=epoch_loss(model,[x],device,optimizer)
        val=epoch_loss(model,[x],device,validation_seed=args.seed)
        # Save/reload to temporary storage only, including S4 length buffers.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';torch.save(model.state_dict(),path)
            loaded=(c.CPCModel() if args.model == 'native' else c.CPCModel(len(c.lead_indices(args.model)))).to(device);loaded.load_state_dict(torch.load(path,map_location=device,weights_only=True))
            loaded.eval();model.eval()
            with torch.no_grad():
                a=model.encoder(x.to(device));b=loaded.encoder(x.to(device))
                torch.testing.assert_close(a,b)
                short=loaded.encoder(x[:,:,:600].to(device))
                if short.shape!=(2,512) or not short.isfinite().all():raise AssertionError('Invalid downstream features')
        print(f'SMOKE OK train={train} val={val}',flush=True);return
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if marker.exists():
            if json.loads(marker.read_text())!=expected:raise ValueError('Completion changed')
            print(f'SKIP {folder}',flush=True);return
        saved=folder/'run_config.json'
        if saved.exists() and json.loads(saved.read_text())!=expected:raise ValueError('Pretraining config mismatch')
        io.atomic_json(saved,expected)
        datasets={g:c.cached_signals(args.site,args.seed,g,c.signature(args.site,args.seed),args.num_workers) for g in ['train','val']}
        for dataset in datasets.values(): dataset.indices=c.lead_indices(args.model)
        # Cache preparation must not alter independent initialization/sampling seeds.
        if args.model != 'native' or args.training_seed not in (None,args.seed):
            set_seed(training_seed)
        # Cached arrays need no resampling workers. Keep optimizer batch64.
        model=(c.CPCModel() if args.model == 'native' else c.CPCModel(len(c.lead_indices(args.model)))).to(device);optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=.001)
        start=0;best=float('inf');wait=0;history=[]
        latest=folder/'latest.pt'
        if latest.exists():
            state=torch.load(latest,map_location=device,weights_only=False)
            if state['signature']!=expected:raise ValueError('Resume signature mismatch')
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            start=state['epoch']+1;best=state['best'];wait=state['wait'];history=state['history']
            torch.set_rng_state(state['rng'].cpu())
            if device.type=='cuda':torch.cuda.set_rng_state_all([t.cpu() for t in state['cuda_rng']])
            print(f'RESUME epoch {start+1}',flush=True)
        for epoch in range(start,expected['epochs']):
            if wait>=expected['patience']:break
            loaders={g:DataLoader(ds,batch_size=64,shuffle=g=='train',num_workers=0,
                pin_memory=device.type=='cuda',generator=torch.Generator().manual_seed(training_seed+epoch if g=='train' else args.seed))
                for g,ds in datasets.items()}
            train,n=epoch_loss(model,loaders['train'],device,optimizer)
            val,nval=epoch_loss(model,loaders['val'],device,validation_seed=args.seed)
            if n!=len(datasets['train']) or nval!=len(datasets['val']):raise AssertionError('ECG omitted')
            history.append(dict(epoch=epoch+1,train_loss=train,val_loss=val,train_ecgs=n,
                val_ecgs=nval,train_steps=len(loaders['train'])))
            if val<best:
                best=val;wait=0
                io.atomic_save(folder/'network.pt',dict(model=model.state_dict(),signature=expected,epoch=epoch+1,val_loss=val))
            else:wait+=1
            io.atomic_save(latest,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                epoch=epoch,best=best,wait=wait,history=history,signature=expected,
                rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []))
            io.atomic_json(folder/'history.json',history)
            print(f'epoch={epoch+1:03d} train={train:.4f} val={val:.4f} best={best:.4f} wait={wait}',flush=True)
        if not (folder/'network.pt').is_file():raise RuntimeError('Missing best checkpoint')
        io.atomic_json(marker,expected);print(f'COMPLETE {folder}',flush=True)

if __name__=='__main__':main()
