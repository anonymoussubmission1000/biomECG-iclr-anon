"""Test-only follow-up sensitivity. No training, no threshold selection.
Keep observed deaths <=365 days OR follow-up >=365 days; invalid negative
follow-up excluded separately. Recompute AUROC/AUPRC on matched test patients.
"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from biomecg.study import train_resnet_unified_ecg1 as shared
from biomecg.study import pclr_ecg1 as io
from biomecg.study.mortality_covariate_ablation_ecg1 import mean_predictions

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seeds',type=int,nargs='+',default=[12345,23451,34512]);p.add_argument('--all-configs',action='store_true',help='All ten configurations of figure 26 bis');a=p.parse_args();rows=[];missing=[]
 for site in ['mimic','code15']:
  cfg=io.source_config(site,a.seeds[0]);paths=cfg['dir_input'][cfg['run_from']][cfg['DB']];labels=shared.load_labels(Path(paths['labels']),Path(paths['ecg']),365)
  for seed in a.seeds:
   _,_,test=shared.load_cached_split(shared.split_paths(type('Args',(),dict(site=site,seed=seed,horizon=365))())[0],labels)
   expected=shared.rows_for_patients(labels,test)
   for cohort in ['8000','full']:
    root=shared.REFACTORED_ROOT/site/f'seed{seed}/runs/horizon365d/cohort{cohort}'
    groups={'biomECG late':[root/f'pretrained_{site}_frozen'/l for l in io.LEADS],
      'PCLR late':[root/f'pclr_{site}_frozen'/l for l in io.LEADS],
      'JEPA late':[root/f'jepa_resnet_early_{site}_frozen'/l for l in io.LEADS],
      'CPC local':[root/f'ecgcpc_local_{site}_frozen/native'],
      'ECG-CPC':[root/'ecgcpc_frozen/native']}
    if a.all_configs:
     for name,arch in [('biomECG',f'pretrained_{site}_frozen'),('PCLR',f'pclr_{site}_frozen'),('JEPA',f'jepa_resnet_early_{site}_frozen')]:
      groups[name+' early']=[root/arch/'early_fusion']
     groups['ECGFounder']=[root/'ecgfounder_frozen'/l for l in io.LEADS]
     groups['ECG-JEPA']=[root/'ecgjepa_frozen/native']
    for family,folders in groups.items():
     files=[f/'predictions.csv' for f in folders]
     if not all(f.is_file() for f in files):missing.append(dict(site=site,seed=seed,cohort=cohort,family=family));continue
     pred=mean_predictions(files).sort_values('row_index');idx=pred.row_index.to_numpy(dtype=int)
     if set(idx)!=set(expected):raise ValueError('Test split differs')
     truth=labels.iloc[idx]
     if pred.pid.astype(str).tolist()!=truth._pid.astype(str).tolist() or not np.array_equal(pred.label,truth._event):raise ValueError('Identity/label mismatch')
     duration=truth.TTE_Mortality.to_numpy(dtype=float);event=truth.E_Mortality.to_numpy(dtype=float)
     if not np.isfinite(duration).all() or not np.isin(event,[0,1]).all():raise ValueError('Invalid follow-up')
     keep=(duration>=0)&((duration>=365)|((event==1)&(duration<=365)))
     y=pred.label.to_numpy();score=pred.prediction.to_numpy()
     if (y[~keep & (duration>=0)]!=0).any():raise ValueError('Observed death excluded')
     for metric,fn in [('AUROC',roc_auc_score),('AUPRC',average_precision_score)]:
      original=fn(y,score);restricted=fn(y[keep],score[keep]);rows.append(dict(site=site,seed=seed,cohort=cohort,family=family,metric=metric,original=original,restricted=restricted,delta=restricted-original,n_original=len(y),n_retained=int(keep.sum()),early_censored=int(((duration>=0)&(duration<365)&(event==0)).sum()),invalid_negative_duration=int((duration<0).sum())))
 out=ROOT/'artifacts/paper/tables';out.mkdir(parents=True,exist_ok=True)
 prefix='followup_sensitivity_all_configs_seed'+ '-'.join(map(str,a.seeds)) if a.all_configs else 'followup_sensitivity_late_three_seeds'
 d=pd.DataFrame(rows);d.to_csv(out/(prefix+'.csv'),index=False)
 pd.DataFrame(missing).to_csv(out/(prefix+'_pending.csv'),index=False)
 if len(d):d.groupby(['site','cohort','family','metric']).agg(original=('original','mean'),restricted=('restricted','mean'),delta=('delta','mean'),n=('seed','nunique')).to_csv(out/(prefix+'_summary.csv'))
 print(f'{len(rows)} metric rows; {len(missing)} configurations pending. No refit.')
if __name__=='__main__':main()
