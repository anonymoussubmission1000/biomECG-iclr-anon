"""Local JEPA-ResNet pretraining on saved biomECG cohort1359.
Sources: https://arxiv.org/html/2605.12241v1 and
https://github.com/AI4HealthUOL/ecg-pretraining-strategies (commit in module).
Adaptations and commands: representations/mimic/README_jepa_resnet_ecg1.md.
"""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader
from biomecg.study import jepa_resnet_ecg1 as j
from biomecg.study import pclr_ecg1 as io
from biomecg.study.train_resnet_unified_ecg1 import set_seed


def epoch_loss(model, loader, device, seed, optimizer=None):
    model.train(optimizer is not None)
    generator = torch.Generator().manual_seed(seed)
    loss_sum = 0.; seen = 0; steps = 0
    sums = [torch.zeros(j.DIM, device=device) for _ in range(4)]
    with torch.set_grad_enabled(optimizer is not None):
        for x in loader:
            x = x.to(device, non_blocking=True)
            context, targets = j.sample_masks(len(x), generator)
            context = context.to(device); targets = targets.to(device)
            if optimizer is not None: optimizer.zero_grad(set_to_none=True)
            loss, features, teacher = model(x, context, targets)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite JEPA loss')
            if optimizer is not None:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], float('inf'), error_if_nonfinite=True)
                optimizer.step(); model.update_teacher()
            for offset, values in [(0, features), (2, teacher)]:
                sums[offset] += values.sum(0); sums[offset+1] += values.square().sum(0)
            loss_sum += float(loss.detach())*len(x); seen += len(x); steps += 1
    if not seen: raise ValueError('Empty JEPA epoch')
    # Monitor variation across recordings; a falling loss alone is not evidence of useful features.
    std = lambda a,b: float((b/seen-(a/seen).square()).clamp_min(0).sqrt().mean())
    return dict(loss=loss_sum/seen, ecgs=seen, steps=steps,
                online_record_std=std(sums[0], sums[1]), teacher_record_std=std(sums[2], sums[3]))


@torch.no_grad()
def calibrate_batchnorm(encoder, loader, device):
    """Re-estimate BN on clean TRAIN only for the unchanged downstream encoder."""
    bns = [m for m in encoder.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    momenta = [m.momentum for m in bns]
    for bn in bns: bn.reset_running_stats(); bn.momentum = None
    encoder.train()
    for x in loader: encoder(j.normalize_signal(x.to(device, non_blocking=True)))
    for bn, momentum in zip(bns, momenta): bn.momentum = momentum
    encoder.eval()


def initialize_training(seed, device, input_channels=8):
    set_seed(seed)
    model = j.JEPAResNet(input_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameter_groups(), weight_decay=.001)
    return model, optimizer


def run(args):
    training_seed=io.variant_seed(args.seed,getattr(args,'training_seed',None))
    input_channels=8 if args.model=='early_fusion' else 1
    torch.set_num_threads(2); set_seed(training_seed)
    expected = j.pretrain_signature(args.site, args.seed, args.epochs,args.model,args.training_seed)
    folder = j.pretrain_dir(args.site, args.seed,args.model,args.training_seed)
    _, manifest, _, source = io.context(args.site, args.seed)
    counts = {g: sum(len(manifest['valid_sids'][p]) for p in manifest[g+'_patients']) for g in ('train','val')}
    print('JEPA-ResNet', args.site, args.seed, counts, flush=True)
    if args.check_only:
        print('CHECK OK: existing cohort/splits; epochs',args.epochs,flush=True); return
    device = torch.device(args.device)
    if args.smoke_test:
        ds = j.ECGRecords(manifest, source, 'train')
        x = torch.stack([ds[i] for i in range(4)]); ds.close()
        x=x[:,[io.LEADS.index(lead) for lead in io.variant_leads(args.model)]]
        model = j.JEPAResNet(input_channels).to(device)
        optimizer = torch.optim.AdamW(model.parameter_groups(), weight_decay=.001)
        before = {k:v.clone() for k,v in model.teacher.state_dict().items()}
        train = epoch_loss(model, [x], device, args.seed, optimizer)
        val = epoch_loss(model, [x], device, args.seed)
        calibrate_batchnorm(model.encoder, [x], device)
        with torch.no_grad():
            y = model.encoder(j.normalize_signal(x.to(device)))
        if y.shape != (4,j.REPRESENTATION_DIM) or not y.isfinite().all(): raise AssertionError('Bad downstream features')
        if all(torch.equal(before[k],v) for k,v in model.teacher.state_dict().items()): raise AssertionError('Teacher not updated')
        print('SMOKE OK',train,val,'encoder params',sum(p.numel() for p in model.encoder.parameters()),flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = folder/'completed.json'; config = folder/'run_config.json'
        if config.exists() and json.loads(config.read_text()) != expected:
            raise ValueError(f'JEPA run config changed: {folder}; use the same --epochs')
        if marker.exists():
            state = torch.load(folder/'network.pt',map_location='cpu',weights_only=True)
            if json.loads(marker.read_text()) != expected or state['signature'] != expected or not state.get('bn_calibrated'):
                raise ValueError('Invalid JEPA completion')
            print('SKIP',folder,flush=True); return
        io.atomic_json(config, expected)
        cache_signature=j.pretrain_signature(args.site,args.seed,args.epochs)
        datasets = {g:j.SelectedLeads(j.cached_records(args.site,args.seed,g,cache_signature,args.num_workers),args.model) for g in ('train','val')}
        # Cache construction can consume torch RNG via DataLoader iterator seeds.
        # Identical seeds must initialize identically with a fresh or existing cache.
        model, optimizer = initialize_training(training_seed, device,input_channels)
        print('Parameters: encoder',sum(p.numel() for p in model.encoder.parameters()),
              'SSL predictor',sum(p.numel() for p in model.predictor.parameters()),
              'exact biomECG early encoder',flush=True)
        latest = folder/'latest.pt'; best_path=folder/'best_training.pt'
        start=0; best=float('inf'); history=[]; elapsed=0.; wait=0
        if latest.exists():
            state=torch.load(latest,map_location=device,weights_only=False)
            if state['signature'] != expected: raise ValueError('Resume config mismatch')
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            start=state['epoch']; best=state['best']; history=state['history']; elapsed=state['elapsed_seconds']; wait=state['wait']
            torch.set_rng_state(state['rng'].cpu())
            if device.type=='cuda': torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
            print('RESUME epoch',start+1,flush=True)
        for epoch in range(start,args.epochs):
            tick=time.monotonic()
            loaders={g:DataLoader(ds,batch_size=64,shuffle=g=='train',num_workers=0,
                pin_memory=device.type=='cuda',generator=torch.Generator().manual_seed(training_seed+epoch if g=='train' else training_seed))
                for g,ds in datasets.items()}
            train=epoch_loss(model,loaders['train'],device,training_seed+epoch,optimizer)
            val=epoch_loss(model,loaders['val'],device,training_seed)
            if train['ecgs']!=len(datasets['train']) or val['ecgs']!=len(datasets['val']):raise AssertionError('ECGs omitted')
            if device.type=='cuda':torch.cuda.synchronize(device)
            seconds=time.monotonic()-tick;elapsed+=seconds
            history.append(dict(epoch=epoch+1,train_loss=train['loss'],val_loss=val['loss'],
                train_ecgs=train['ecgs'],val_ecgs=val['ecgs'],train_steps=train['steps'],
                online_record_std=val['online_record_std'],teacher_record_std=val['teacher_record_std'],
                epoch_seconds=seconds,elapsed_seconds=elapsed))
            if val['loss']<best:
                best=val['loss']; wait=0
                io.atomic_save(best_path,dict(model=model.state_dict(),signature=expected,epoch=epoch+1,val_loss=best))
            else:
                wait += 1
            io.atomic_save(latest,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                signature=expected,epoch=epoch+1,best=best,wait=wait,history=history,elapsed_seconds=elapsed,
                rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []))
            io.atomic_json(folder/'history.json',history)
            print(f"epoch={epoch+1:03d}/{args.epochs} train={train['loss']:.5f} val={val['loss']:.5f} best={best:.5f} epochs_since_best={wait} "
                  f"online_std={val['online_record_std']:.5f} teacher_std={val['teacher_record_std']:.5f} seconds={seconds:.1f}",flush=True)
        selected=torch.load(best_path,map_location=device,weights_only=True)
        model.load_state_dict(selected['model'])
        clean_loader=DataLoader(datasets['train'],batch_size=64,shuffle=False,num_workers=0,pin_memory=device.type=='cuda')
        calibrate_batchnorm(model.encoder,clean_loader,device)
        io.atomic_save(folder/'network.pt',dict(encoder=model.encoder.state_dict(),signature=expected,
            epoch=selected['epoch'],val_loss=selected['val_loss'],bn_calibrated=True))
        io.atomic_json(folder/'history.json',history)
        io.atomic_json(marker,expected)
        print('COMPLETE',folder,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site',choices=['mimic','code15'],required=True)
    parser.add_argument('--seed',type=int,choices=j.SEEDS,required=True)
    parser.add_argument('--epochs',type=int,default=j.DEFAULT_EPOCHS)
    parser.add_argument('--num-workers',type=int,default=2)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--model',choices=['early_fusion',*io.LEADS],default='early_fusion')
    parser.add_argument('--training-seed',type=int)
    args=parser.parse_args()
    if args.epochs<1 or args.num_workers<0:parser.error('Invalid epochs/workers')
    run(args)

if __name__=='__main__':main()
