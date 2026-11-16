"""CPC from random initialization on the saved biomECG cohort1359.

Sources (benchmark commit 238409835ef55358a10bbc3459dfa9aaa91ad5e5):
https://github.com/AI4HealthUOL/ecg-fm-benchmarking/blob/main/code/conf/config_cpc_ecg_s4_heedb.yaml
https://github.com/AI4HealthUOL/ecg-fm-benchmarking/blob/main/code/clinical_ts/loss/selfsupervised.py
https://github.com/AI4HealthUOL/ecg-fm-benchmarking/blob/main/code/clinical_ts/template_model.py
Local adaptations: 8.192s complete cleaned ECGs, historical patient manifests,
validation negatives fixed, patience20, per-epoch resumable checkpoints.
See AUDIT_CPC_PRETRAINING.md. This is not the released external checkpoint.
"""
import fcntl
import json
import os
from pathlib import Path

import h5py
import numpy as np
import resampy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader
from biomecg.study import pclr_ecg1 as cohort
from biomecg.study.foundation_models_ecg1 import torch_cauchy_conj

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'cpc_local_ecg1_v1_full8192ms'
COMMIT = '238409835ef55358a10bbc3459dfa9aaa91ad5e5'
SAMPLES = 1966  # resampy: floor(4096 * 240 / 500)


def lead_indices(model='native'):
    if model == 'native': return list(range(12))
    if model == 'early_fusion': return cohort.INDICES.copy()
    if model not in cohort.LEADS: raise ValueError(f'Invalid CPC model: {model}')
    return [cohort.INDICES[cohort.LEADS.index(model)]]


def pretrain_dir(site, seed, model='native', training_seed=None):
    if model != 'native' or training_seed not in (None, seed):
        lead_indices(model)
        return ROOT/'results/CPC/ensemble_checkpoints'/f'{site}_cohort1359_seed{seed}'/model/f'trainseed{seed if training_seed is None else training_seed}' 
    return ROOT / 'results/CPC/checkpoints' / f'{site}_12l_cohort1359_seed{seed}'


def signature(site, seed, model='native', training_seed=None):
    _, manifest, path, source = cohort.context(site, seed)
    if [len(manifest[g+'_patients']) for g in ('train','val')] != [1087,272]:
        raise ValueError('Unexpected cohort1359 partition')
    ecg = Path(source['ecg']).resolve()
    result = dict(protocol=PROTOCOL, site=site, seed=seed, manifest=str(path),
        manifest_sha256=cohort.sha256(path), ecg_path=str(ecg),
        ecg_stat=[ecg.stat().st_size,ecg.stat().st_mtime_ns], benchmark_commit=COMMIT,
        fs=240,source_fs=500,source_samples=4096,samples=SAMPLES,leads=12,
        batch_size=64,epochs=200,patience=20,optimizer='AdamW',
        lr_head=1e-3,lr_s4=1e-4,lr_cnn=1e-5,weight_decay=1e-3,scheduler='constant',
        precision='float32',future_offset=14,negatives=128,temperature=1.,
        negative_sampling='same ECG, with replacement, excluding positive',
        loss_reduction='sum over time, mean over ECGs',normalization='none_raw_mV',
        sampling='each eligible ECG once per epoch',validation='fixed negatives; minimum CPC loss',
        architecture='CNN512x4-k3,1,1,1-stride2,1,1,1-S4x4-d512-state8-dropout0.2')
    if model != 'native' or training_seed not in (None, seed):
        result.update(model=model, training_seed=seed if training_seed is None else training_seed,
                      leads=len(lead_indices(model)), lead_indices=lead_indices(model))
    return result


class CPCEncoder(nn.Module):
    def __init__(self, input_channels=12):
        super().__init__()
        from clinical_ts.template_modules import ShapeConfig
        from clinical_ts.ts.encoder import RNNEncoder,RNNEncoderConfig
        from clinical_ts.ts.s4 import S4Predictor,S4PredictorConfig
        from clinical_ts.ts.s4_modules import s42
        s42.cauchy_conj = torch_cauchy_conj
        self.spec={'dim':512}
        shape=ShapeConfig(channels=input_channels,length=SAMPLES,sequence_last=True)
        self.conv=RNNEncoder(RNNEncoderConfig(features=[512]*4,kss=[3,1,1,1],strides=[2,1,1,1]),shape,None)
        self.temporal=S4Predictor(S4PredictorConfig(causal=True,state_dim=8,model_dim=512,
            backbone='s42',layers=4,dropout=.2),self.conv.get_output_shape())
        # Upstream S4 registers expanded buffers; materialize them so strict
        # checkpoint reload also works on CPU (before moving to CUDA).
        for module in self.modules():
            for name,value in module._buffers.items():
                if value is not None:module._buffers[name]=value.clone()

    def sequence(self,x):
        encoded=self.conv(seq=x)
        return encoded['seq'],self.temporal(**encoded)['seq']

    def forward(self,x):
        return self.sequence(x)[1].mean(dim=1)

    def depth_groups(self):
        return list(self.conv.parameters()),list(self.temporal.parameters())


class CPCModel(nn.Module):
    def __init__(self, input_channels=12):
        super().__init__();self.encoder=CPCEncoder(input_channels);self.head=nn.Linear(512,512,bias=False)
    def forward(self,x):
        targets,context=self.encoder.sequence(x)
        return self.head(context),targets
    def parameter_groups(self):
        return [dict(params=self.head.parameters(),lr=1e-3),
                dict(params=self.encoder.temporal.parameters(),lr=1e-4),
                dict(params=self.encoder.conv.parameters(),lr=1e-5)]


def cpc_loss(predicted, encoded, generator=None, chunk_size=16, offset=14, negatives=128):
    """Original CPCLoss, with bounded candidate memory and recomputed gradients.

    Neither targets nor predictions are detached. Chunking preserves the temporal
    sum, batch mean and negative distribution; avoids a ~15GiB candidate tensor.
    """
    b,t,d=encoded.shape
    if predicted.shape != encoded.shape or t<=offset or chunk_size<1:
        raise ValueError('Invalid CPC sequence/shape')
    total=predicted.new_zeros(())
    for start in range(0,t-offset,chunk_size):
        end=min(start+chunk_size,t-offset)
        pos=torch.arange(start+offset,end+offset,device=encoded.device)
        idx=torch.randint(t-1,(b,end-start,negatives),device=encoded.device,generator=generator)
        idx=idx+(idx>=pos[None,:,None])
        # Bind each chunk explicitly: backward recomputation must not capture
        # the final loop iteration's indices through late-bound closures.
        def block(pred,enc,ids,positions):
            batch=torch.arange(enc.shape[0],device=enc.device)[:,None,None]
            neg=enc[batch,ids]
            positive=enc[:,positions].unsqueeze(2)
            candidates=torch.cat([positive,neg],dim=2)
            scores=torch.einsum('btf,btnf->btn',pred,candidates)
            target=torch.zeros(scores.shape[:2],device=scores.device,dtype=torch.long)
            return F.cross_entropy(scores.reshape(-1,negatives+1),target.flatten(),reduction='sum')/b
        args=(predicted[:,start:end],encoded,idx,pos)
        part=checkpoint(block,*args,use_reentrant=False) if torch.is_grad_enabled() else block(*args)
        total=total+part
    return total


class SourceECGs(Dataset):
    def __init__(self,manifest,source,group):
        self.path=source['ecg'];self.file=None;self.owner=None
        with h5py.File(self.path,'r') as f:
            if f['ECG'].shape[1:]!=(12,4096) or int(f.attrs.get('sampling_rate',500))!=500:
                raise ValueError('Expected 12x4096 ECGs at 500Hz')
            lookup={str(sid):i for i,sid in enumerate(f['SID'][:])};pids=f['PID'][:]
            self.rows=[]
            for pid in manifest[group+'_patients']:
                for sid in manifest['valid_sids'][pid]:
                    row=lookup[str(sid)]
                    if str(pids[row])!=pid:raise ValueError('SID/PID mismatch')
                    self.rows.append(row)
        if len(set(self.rows))!=len(self.rows):raise ValueError('Duplicate ECG')
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        if self.file is None or self.owner!=os.getpid():
            self.close();self.file=h5py.File(self.path,'r');self.owner=os.getpid()
        x=self.file['ECG'][self.rows[index]].astype(np.float32)
        if not np.isfinite(x).all():raise ValueError('Non-finite ECG')
        x=resampy.resample(x,500,240,axis=-1)
        if x.shape!=(12,SAMPLES) or not np.isfinite(x).all():raise ValueError('Invalid resampled ECG')
        return torch.from_numpy(np.ascontiguousarray(x))
    def close(self):
        if self.file is not None:self.file.close();self.file=None
    def __getstate__(self):
        state=self.__dict__.copy();state['file']=None;state['owner']=None;return state
    def __del__(self):self.close()


def cached_signals(site,seed,group,expected,workers):
    """Disk memmap cache, atomic completion and provenance; no GPU residency."""
    _,manifest,_,source=cohort.context(site,seed)
    folder=ROOT/'results/CPC/cache'/f'{site}_seed{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{group}.npy';marker=folder/f'{group}.json'
    with (folder/f'{group}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if marker.exists():
            if json.loads(marker.read_text())!=expected or not path.exists():raise ValueError('CPC cache signature mismatch')
        else:
            ds=SourceECGs(manifest,source,group)
            temp=folder/f'{group}.tmp.npy'
            array=np.lib.format.open_memmap(temp,mode='w+',dtype='float32',shape=(len(ds),12,SAMPLES))
            offset=0
            for batch in DataLoader(ds,batch_size=32,num_workers=workers,shuffle=False):
                array[offset:offset+len(batch)]=batch.numpy();offset+=len(batch)
            if offset!=len(ds):raise ValueError('Missing cached ECGs')
            array.flush();del array;ds.close();temp.replace(path)
            cohort.atomic_json(marker,expected)
    return SignalCache(path)


class SignalCache(Dataset):
    def __init__(self,path,indices=None):
        self.array=np.load(path,mmap_mode='r');self.indices=indices
    def __len__(self):return len(self.array)
    def __getitem__(self,index):return torch.from_numpy((self.array[index] if self.indices is None else self.array[index,self.indices]).copy())


def provenance(site,seed,model='native',training_seed=None):
    expected=signature(site,seed,model,training_seed);folder=pretrain_dir(site,seed,model,training_seed)
    if not (folder/'completed.json').is_file() or json.loads((folder/'completed.json').read_text())!=expected:
        raise ValueError(f'Complete matching CPC pretraining required: {folder}')
    path=folder/'network.pt'
    return dict(pretraining_checkpoint=str(path),checkpoint_sha256=cohort.sha256(path),
                benchmark_commit=COMMIT,local_pretraining=expected)


def load_encoder(site,seed,model='native',training_seed=None):
    info=provenance(site,seed,model,training_seed)
    state=torch.load(info['pretraining_checkpoint'],map_location='cpu',weights_only=False)
    if state['signature']!=info['local_pretraining']:raise ValueError('CPC checkpoint signature mismatch')
    model=CPCModel(len(lead_indices(model)));model.load_state_dict(state['model'],strict=True)
    return model.encoder
