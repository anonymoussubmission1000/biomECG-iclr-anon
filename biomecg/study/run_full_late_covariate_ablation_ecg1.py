"""Focused ablation: PCLR/JEPA late and external ECG-CPC, Full, both sites.
biomECG late and age/sex-only results already exist and are reused.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from biomecg.study import run_model_covariate_ablation_ecg1 as base

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=[12345,23451,34512],required=True);p.add_argument('--num-workers',type=int,default=2);p.add_argument('--check-only',action='store_true');a=p.parse_args()
 for site in ['mimic','code15']:
  for family in ['pclr','jepa','ecgcpc']:
   for model in base.LEADS if family!='ecgcpc' else ['early_fusion']:
    cmd=[sys.executable,'-u',str(ROOT/'biomecg/study/run_model_covariate_ablation_ecg1.py'),'--worker','--site',site,'--seed',str(a.seed),'--family',family,'--cohort','full','--model',model,'--num-workers',str(a.num_workers)]
    if a.check_only:cmd+=['--check-only']
    subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),check=True)
   if family!='ecgcpc' and not a.check_only:
    aggregate(site,a.seed,family)

def aggregate(site,seed,family):
 import pandas as pd
 from biomecg.study import mortality_covariate_ablation_ecg1 as bio
 from biomecg.study import train_resnet_unified_ecg1 as shared
 arch=f'pclr_{site}_frozen' if family=='pclr' else f'jepa_resnet_early_{site}_frozen'
 root=shared.REFACTORED_ROOT/site/f'seed{seed}/runs/horizon365d/cohortfull/ablations/ecg_only'/arch
 if not all((root/l/'ablation_completed.json').is_file() for l in base.LEADS):
  print('PENDING late aggregation',root,flush=True);return
 frame=bio.mean_predictions([root/l/'predictions.csv' for l in base.LEADS]);out=root/'late';out.mkdir(exist_ok=True)
 frame.to_csv(out/'predictions.csv',index=False);frame.to_csv(out/'predictions_patient.csv',index=False)
 metrics=dict(site=site,seed=seed,cohort_label='full',family=family,variant='ecg_only_late',**shared.calculate_metrics(frame.label.to_numpy(),frame.prediction.to_numpy(),'test_patient'))
 pd.DataFrame([metrics]).to_csv(out/'metrics.csv',index=False)
if __name__=='__main__':main()
