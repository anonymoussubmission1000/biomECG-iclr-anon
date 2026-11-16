"""ECG-only ablation of frozen barplot models; existing age/sex baseline is reused.
Uses original trainers, sampling, validation selection and model-specific budgets.
References: train_{pclr,jepa_resnet,ecgfounder,foundation}_unified_ecg1.py.
No pretraining. Outputs are isolated under cohort*/ablations/ecg_only/.
"""
import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
FAMILIES=('pclr','jepa','cpc_local','ecgfounder','ecgcpc','ecgjepa')
SEEDS=(12345,23451,34512,45123,51234)
LEADS=('I','II','V1','V2','V3','V4','V5','V6')


def ecg_only_class(base):
    import torch
    class ECGOnly(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if not self.frozen:raise ValueError('Ablation requires frozen encoder')
            old=self.head[0]
            class ECGHead(torch.nn.Sequential):
                def forward(self,features):
                    # Native trainers concatenate age/sex, including cached evaluation.
                    # Remove both before the first trainable layer, never zero-pad them.
                    return super().forward(features[...,:-2])
            self.head=ECGHead(torch.nn.Linear(old.in_features-2,old.out_features),*list(self.head.children())[1:])
    return ECGOnly


def run_worker(a):
    import torch
    from biomecg.study import pclr_ecg1 as io
    from biomecg.study import train_resnet_unified_ecg1 as shared
    cfg=io.source_config(a.site,a.seed);paths=cfg['dir_input'][cfg['run_from']][cfg['DB']]
    stem={'pclr':'pclr','jepa':'jepa_resnet','ecgfounder':'ecgfounder'}.get(a.family,'foundation')
    trainer=importlib.import_module('biomecg.study.train_'+stem+'_unified_ecg1')
    arch=f'pclr_{a.site}_frozen' if a.family=='pclr' else f'jepa_resnet_early_{a.site}_frozen' if a.family=='jepa' else 'pretrained_frozen'
    cli=['--site',a.site,'--seed',str(a.seed),'--labels_path',paths['labels'],'--ecg_path',paths['ecg'],'--horizon','365','--cohort_size',a.cohort,'--architecture',arch,'--num_workers',str(a.num_workers),'--cpu_threads','2']
    if stem=='foundation':
        cli+=['--family','ecgcpc' if a.family=='cpc_local' else a.family]
        if a.family=='cpc_local':cli+=['--cpc-local']
    else:cli+=['--model',a.model,'--bs','256','--lr_encoder','0','--lr_head','0.001','--epochs','200','--patience','20']
    sys.argv=[sys.argv[0],*cli]
    engine=shared if a.family in ('pclr','jepa') else trainer
    args=engine.parse_args()
    original=engine.output_dir(args)
    if not all((original/f).is_file() for f in ['best.pt','metrics.csv','predictions.csv']):
        print('PENDING combined reference:',original,flush=True);return
    # Match the completed reference's learning rate and effective batch exactly.
    import pandas as pd
    reference=pd.read_csv(original/'metrics.csv').iloc[0]
    for column,attribute in [('batch_size','bs'),('lr_head','lr_head')]:
        if column in reference and float(reference[column])!=float(getattr(args,attribute)):
            raise ValueError(f'Reference {column} differs: {original}')
    # Require the saved split; never create a new split for an ablation.
    split=shared.split_paths(args)[0]
    if not split.is_file():raise FileNotFoundError(split)
    target=original.parent.parent/'ablations/ecg_only'/original.parent.name/original.name
    signature=dict(protocol='barplot_ecg_only_v1',family=a.family,site=a.site,seed=a.seed,
        cohort=a.cohort,model=a.model,arguments=cli,split_sha256=io.sha256(split),
        reference_metrics_sha256=io.sha256(original/'metrics.csv'))
    # Keep historical signatures compatible when only DataLoader workers change.
    signature_cli = list(cli)
    signature_cli[signature_cli.index('--num_workers') + 1] = '2'
    signature['arguments'] = signature_cli
    target.mkdir(parents=True,exist_ok=True)
    with (target/'.ablation.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=target/'ablation_config.json';done=target/'ablation_completed.json'
        if config.exists() and json.loads(config.read_text())!=signature:raise ValueError(f'Configuration mismatch: {target}')
        if done.exists() and json.loads(done.read_text())==signature and all((target/f).is_file() for f in ['best.pt','metrics.csv','predictions.csv','predictions_patient.csv']):
            print('SKIP',target,flush=True);return
        if a.check_only:print('READY',target,flush=True);return
        io.atomic_json(config,signature)
        engine.output_dir=lambda _:target
        engine.parse_args=lambda:args
        if stem=='foundation':
            trainer.parse_args=lambda *unused:args
            trainer.FoundationMortalityModel=ecg_only_class(trainer.FoundationMortalityModel)
        else:engine.MortalityModel=ecg_only_class(engine.MortalityModel)
        trainer.main()
        io.atomic_json(done,signature)


def aggregate_founder(site,seed,cohort):
    import pandas as pd
    from biomecg.study import mortality_covariate_ablation_ecg1 as bio
    from biomecg.study import train_resnet_unified_ecg1 as shared
    base=shared.REFACTORED_ROOT/site/f'seed{seed}/runs/horizon365d/cohort{cohort}/ablations/ecg_only/ecgfounder_frozen'
    if not all((base/l/'ablation_completed.json').is_file() for l in LEADS):return
    frame=bio.mean_predictions([base/l/'predictions.csv' for l in LEADS])
    folder=base/'late';folder.mkdir(exist_ok=True)
    frame.to_csv(folder/'predictions.csv',index=False)
    metrics=dict(site=site,seed=seed,cohort_label=cohort,architecture='ecgfounder_frozen',model='late',variant='ecg_only')
    metrics.update(shared.calculate_metrics(frame.label.to_numpy(),frame.prediction.to_numpy(),'test_patient'))
    pd.DataFrame([metrics]).to_csv(folder/'metrics.csv',index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--site',choices=['all','mimic','code15'],default='all')
    p.add_argument('--family',choices=['all',*FAMILIES],default='all')
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--cohort',choices=['1000','8000','full']);p.add_argument('--model',default='early_fusion')
    a=p.parse_args()
    if a.worker:run_worker(a);return
    for site in ['mimic','code15'] if a.site=='all' else [a.site]:
        for family in FAMILIES if a.family=='all' else [a.family]:
            for cohort in ['1000','8000','full']:
                for model in LEADS if family=='ecgfounder' else ['early_fusion']:
                    command=[sys.executable,'-u',__file__,'--worker','--site',site,'--family',family,'--seed',str(a.seed),'--cohort',cohort,'--model',model,'--num-workers',str(a.num_workers)]
                    if a.check_only:command+=['--check-only']
                    subprocess.run(command,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),check=True)
                if family=='ecgfounder' and not a.check_only:aggregate_founder(site,a.seed,cohort)

if __name__=='__main__':main()
