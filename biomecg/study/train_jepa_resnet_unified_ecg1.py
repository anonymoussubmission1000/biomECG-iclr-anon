"""JEPA-ResNet mortality downstream using the existing ECG1 trainer unchanged in its learning recipe.
Sources: https://arxiv.org/html/2605.12241v1 and
https://github.com/AI4HealthUOL/ecg-pretraining-strategies
Adaptations: README_jepa_resnet_ecg1.md.
"""
import fcntl
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from biomecg.study import train_resnet_unified_ecg1 as shared
from biomecg.study import jepa_resnet_ecg1 as p
from biomecg.study import pclr_ecg1 as io


def signature(args):
    config,provenance=p.resolve_downstream(args)
    result={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
            if k not in ('prepare_only','num_workers','cpu_threads','device')}
    result.update(protocol=p.PROTOCOL,pretraining_checkpoint_sha256=provenance['pretraining_checkpoint_sha256'],
        split_sha256=io.sha256(shared.split_paths(args)[0]))
    for key in ('ecg_path','labels_path'):
        path=getattr(args,key).resolve();result[key]=str(path)
        result[key+'_stat']=[path.stat().st_size,path.stat().st_mtime_ns]
    return result


def main():
    args=shared.parse_args()
    if args.architecture != f'jepa_resnet_early_{args.site}_frozen':
        raise ValueError('This JEPA-ResNet study only runs same-site frozen downstream')
    expected=signature(args);folder=shared.output_dir(args)
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        saved=folder/'jepa_resnet_run_config.json';marker=folder/'completed.json'
        if saved.exists() and json.loads(saved.read_text())!=expected:raise ValueError(f'Run config differs: {folder}')
        if marker.exists():
            if json.loads(marker.read_text())!=expected:raise ValueError(f'Completion config differs: {folder}')
            if all((folder/n).is_file() and (folder/n).stat().st_size for n in
                   ['best.pt','metrics.csv','predictions.csv','predictions_patient.csv']):
                print(f'SKIP {folder}',flush=True);return
        io.atomic_json(saved,expected)
        io.atomic_json(folder/'execution_precision.json', {
            'cuda_autocast': 'bfloat16', 'cpu': 'float32',
            'gradient_scaling': False,
            'reason': 'Stable downstream JEPA-ResNet evaluation'})
        # Reuse the exact historical train/validation/test loop, covariates, MLP,
        # loss, LR groups, early stopping and saved nested mortality split.
        shared.parse_args=lambda:args
        shared.main()
        if not args.prepare_only:io.atomic_json(marker,expected)

if __name__=='__main__':main()
