"""Model-only GPU inference benchmark; no disk IO, preprocessing or feature cache.
Same device, batch and FP32 precision for every model. Sequential experts/windows;
all expert parameters resident on GPU. Synthetic model-ready inputs, native window
counts and mean probabilities. Encoder weights are loaded for foundation/CPC;
ResNet topology uses random weights (weight values do not change operation count).
"""
import argparse
import copy
import csv
import gc
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
CHOICES=['resnet_early','resnet_late','resnet_early8','cpc_local','ecgfounder_late','ecgcpc','ecgjepa']


def build(name,site,seed):
    from biomecg.study import train_resnet_unified_ecg1 as shared
    count=8 if name in ['resnet_late','resnet_early8','ecgfounder_late'] else 1
    windows=1;channels=1 if name in ['resnet_late','ecgfounder_late'] else 8;length=4096
    if name.startswith('resnet'):
        from biomecg.study.jepa_resnet_ecg1 import new_encoder
        model=shared.MortalityModel(new_encoder(channels),True)
    elif name=='ecgfounder_late':
        from biomecg.study import train_ecgfounder_unified_ecg1 as founder
        model=founder.build_model(SimpleNamespace(model='I',architecture='pretrained_frozen'),torch.device('cpu'))[0]
    else:
        from biomecg.study.foundation_models_ecg1 import FoundationEncoder,FoundationMortalityModel,SPECS
        family='ecgcpc' if name=='cpc_local' else name
        if name=='cpc_local':
            from biomecg.study.cpc_local_ecg1 import load_encoder
            encoder=load_encoder(site,seed)
        else:encoder=FoundationEncoder(family)
        model=FoundationMortalityModel(encoder,True)
        channels=12;length=SPECS[family]['samples'];windows=1 if family=='ecgjepa' else 4
    return nn.ModuleList([model]+[copy.deepcopy(model) for _ in range(count-1)]),channels,length,windows


def measure(name,a):
    gc.collect();torch.cuda.empty_cache()
    models,channels,length,windows=build(name,a.site,a.seed)
    models=models.cuda().eval()
    # Independent windows/leads have the same shape and computational cost.
    inputs=torch.randn(a.batch,channels,length,device='cuda')
    metadata=torch.zeros(a.batch,2,device='cuda')
    def forward():
        scores=[]
        for model in models:
            for _ in range(windows):scores.append(model(inputs,metadata).float().sigmoid())
        return torch.stack(scores).mean(0)
    with torch.inference_mode():
        for _ in range(a.warmup):forward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings=[]
        for _ in range(a.repeats):
            start=time.perf_counter();output=forward();torch.cuda.synchronize()
            timings.append(time.perf_counter()-start)
        if not torch.isfinite(output).all():raise ValueError('Nonfinite output')
    row=dict(model=name,batch=a.batch,precision='float32',experts=len(models),windows_per_expert=windows,
        encoder_parameters=sum(p.numel() for m in models for p in m.encoder.parameters()),
        total_parameters=sum(p.numel() for p in models.parameters()),
        weight_MiB=sum(p.numel()*p.element_size() for p in models.parameters())/2**20,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
        median_batch_ms=float(np.median(timings)*1000),ecg_per_second=float(a.batch/np.median(timings)))
    del models,inputs,metadata,output
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=CHOICES,default=CHOICES)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--warmup',type=int,default=10);p.add_argument('--repeats',type=int,default=30)
    p.add_argument('--site',choices=['mimic','code15'],default='mimic');p.add_argument('--seed',type=int,default=12345)
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/paper/tables/inference_benchmark.csv')
    a=p.parse_args()
    if min(a.batch,a.warmup,a.repeats)<1:p.error('Positive sizes required')
    if not torch.cuda.is_available():p.error('CUDA required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    a.output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in a.models:
        row=measure(name,a);rows.append(row);print(json.dumps(row),flush=True)
        with a.output.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
    a.output.with_suffix('.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},scope=__doc__,resnet_rows='Shared inference topology for biomECG, PCLR and local JEPA; respective early/late/early8 configurations. Run on an idle GPU for publishable timings.'),indent=2))
    latex=[r'\begin{tabular}{lrrr}',r'\hline',r'Model & Encoder (M) & VRAM (GiB) & ECG/s \\',r'\hline']
    for r in rows:latex.append(r['model'].replace('_',r'\_')+f" & {r['encoder_parameters']/1e6:.2f} & {r['peak_allocated_GiB']:.2f} & {r['ecg_per_second']:.1f}"+r' \\')
    latex += [r'\hline',r'\end{tabular}']
    a.output.with_suffix('.tex').write_text('\n'.join(latex)+'\n')

if __name__=='__main__':main()
