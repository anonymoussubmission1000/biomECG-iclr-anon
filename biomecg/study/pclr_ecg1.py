"""Matched-backbone PCLR on saved biomECG MIMIC/CODE-15 cohorts.

Sources:
https://doi.org/10.1371/journal.pcbi.1009862 (Diamant et al., 2022)
https://journals.plos.org/ploscompbiol/article/file?type=supplementary&id=10.1371/journal.pcbi.1009862.s002
https://github.com/broadinstitute/ml4h/blob/master/model_zoo/PCLR/build_model.py
https://github.com/mit-ccrg/PCLR

Official symmetric loss is mean(CE first views)+mean(CE second views).
Adaptations: our 8-lead 128D encoder, 256D projection, existing cleaned/z-scored
4096-point strips, eight patients/batch, five patient visits/epoch. Original PCLR
has a 320D encoder/projection and 512 patients/batch; its steps/epoch are unknown.
No claim of compute matching to historical biomECG is made for this protocol.
"""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (12345,23451,34512,45123,51234)
LEADS = ['I','II','V1','V2','V3','V4','V5','V6']
INDICES = [0,1,6,7,8,9,10,11]
PROTOCOL = 'pclr_ecg1_v2_five_patient_visits_batch8'


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)


def atomic_save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,tmp);tmp.replace(path)


def source_config(site,seed):
    from biomecg.core.dataset.mimic_cohort import configure_pretraining_cohort
    config=json.loads((ROOT/f'configs/study/{site}/{208 if site=="mimic" else 200}.json').read_text())
    config['seed']=seed
    for key,value in config['dir_input'][config['run_from']][config['DB']].items():
        config['dir_input'][config['run_from']][config['DB']][key]=str(ROOT/value)
    config['architecture']['leads']=LEADS.copy()
    config=configure_pretraining_cohort(config,1359)
    return config


def context(site,seed):
    from biomecg.core.dataset.mimic_cohort import prepare_cohort
    config=source_config(site,seed)
    manifest_path=ROOT/config['biometrics']['manifest_path'].format(seed=seed)
    if not manifest_path.is_file():raise FileNotFoundError(f'Existing biomECG manifest required: {manifest_path}')
    manifest,path=prepare_cohort(config,ROOT)  # validates saved input/split fingerprints
    source=config['dir_input'][config['run_from']][config['DB']]
    return config,manifest,path,source


def variant_leads(model='early_fusion'):
    if model == 'early_fusion': return LEADS.copy()
    if model not in LEADS: raise ValueError(f'Invalid lead: {model}')
    return [model]


def variant_seed(seed, training_seed=None):
    value = seed if training_seed is None else training_seed
    if value < 0: raise ValueError('Training seed must be nonnegative')
    return value


def variant_signature(signature, seed, model, training_seed):
    leads = variant_leads(model)
    training_seed = variant_seed(seed, training_seed)
    if model == 'early_fusion' and training_seed == seed: return signature
    signature = dict(signature, model=model, training_seed=training_seed, leads=leads)
    if 'architecture' in signature:
        signature['architecture'] = dict(signature['architecture'], leads=leads)
    if 'encoder_params' in signature:
        signature['encoder_params'] = 212992 if model == 'early_fusion' else 211648
    return signature


def pretrain_dir(site,seed,model='early_fusion',training_seed=None):
    if model == 'early_fusion' and variant_seed(seed,training_seed) == seed:
        return ROOT/f'results/PCLR/checkpoints/{site}_8l_cohort1359_seed{seed}'
    variant_leads(model)
    return ROOT/'results/PCLR/ensemble_checkpoints'/f'{site}_cohort1359_seed{seed}'/model/f'trainseed{variant_seed(seed,training_seed)}' 


def pretrain_signature(site,seed,model='early_fusion',training_seed=None):
    config,manifest,path,source=context(site,seed)
    result = dict(protocol=PROTOCOL,site=site,seed=seed,cohort_size=1359,
        manifest_sha256=sha256(path),manifest=str(path),source=source,
        architecture=config['architecture'],dropout=0.,projection=[128,256,256],
        patients_per_batch=8,epochs=50,optimizer='Adam',lr=.1,betas=[.9,.999],
        epsilon=1e-7,weight_decay=0.,temperature=.1,
        scheduler='epoch cosine: 0.1*(1+cos(pi*epoch/50))/2',
        patient_visits_per_epoch=5, sampling='five visits/patient/epoch; two independent ECG draws per visit with replacement',
        normalization='per lead z-score, epsilon 1e-6',checkpoint_selection='minimum fixed validation loss')
    return variant_signature(result,seed,model,training_seed)


class PCLRModel(nn.Module):
    def __init__(self, input_channels=8):
        super().__init__()
        from biomecg.core.architecture.resnet1d import SiameseResNet1d
        self.encoder=SiameseResNet1d(input_channels=input_channels,embedding_dim=256,
            fix_feature_dim=True,kernel_size=3,dilation_rate=1,dropout=0.).encoder
        # PCLR's two-layer nonlinear projection, adapted to our 128D encoder.
        self.projection=nn.Sequential(nn.Linear(128,256),nn.ReLU(),nn.Linear(256,256))
        for layer in self.projection:
            if isinstance(layer,nn.Linear):
                nn.init.kaiming_normal_(layer.weight,nonlinearity='relu');nn.init.zeros_(layer.bias)

    def forward(self,x):return self.projection(self.encoder(x))


def contrastive_loss(z,temperature=.1):
    if z.ndim!=2 or len(z)%2 or len(z)<4:raise ValueError('Need two views of at least two patients')
    z=F.normalize(z,dim=-1,eps=1e-12)
    logits=z@z.T/temperature
    logits=logits.masked_fill(torch.eye(len(z),device=z.device,dtype=torch.bool),-1e9)
    targets=(torch.arange(len(z),device=z.device)+len(z)//2)%len(z)
    # Exact official loss scale: CE(a->b)+CE(b->a), not half their sum.
    return 2*F.cross_entropy(logits,targets)


class PatientViews(Dataset):
    """Five visits per patient; independent ECG draws for each epoch and visit."""
    def __init__(self,manifest,source,group,seed,model='early_fusion'):
        self.indices=[INDICES[LEADS.index(lead)] for lead in variant_leads(model)]
        self.patients=list(manifest[f'{group}_patients']);self.seed=seed;self.epoch=0
        self.path=source['ecg'];self.file=None;self.owner=None
        with h5py.File(self.path,'r') as f:
            lookup={str(sid):i for i,sid in enumerate(f['SID'][:])}
            pids=f['PID'][:]
            self.rows=[]
            for pid in self.patients:
                rows=[lookup[str(sid)] for sid in manifest['valid_sids'][pid]]
                if len(rows)<2 or any(str(pids[r])!=pid for r in rows):raise ValueError(f'Invalid sessions: {pid}')
                self.rows.append(rows)

    def __len__(self):return 5*len(self.patients)
    def __getstate__(self):
        s=self.__dict__.copy();s['file']=None;s['owner']=None;return s
    def close(self):
        if self.file is not None:self.file.close();self.file=None
    def __del__(self):self.close()
    def __getitem__(self,index):
        visit,index=divmod(index,len(self.patients))
        if self.file is None or self.owner!=os.getpid():
            self.close();self.file=h5py.File(self.path,'r');self.owner=os.getpid()
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index,visit]))
        rows=rng.choice(self.rows[index],size=2,replace=True)
        views=[]
        for row in rows:
            x=self.file['ECG'][int(row)].astype(np.float32)[self.indices]
            center=x.shape[-1]//2;x=x[:,center-2048:center+2048]
            if x.shape!=(len(self.indices),4096) or not np.isfinite(x).all() or np.any(x.std(1)<=1e-6):
                raise ValueError(f'Invalid PCLR input: {self.path}/{row}')
            x=(x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-6)
            views.append(torch.from_numpy(x))
        return views[0],views[1],index



class PatientPassBatchSampler:
    """Five separate patient traversals; never repeat a patient within a batch."""
    def __init__(self,dataset,shuffle):
        self.dataset=dataset;self.shuffle=shuffle
    def __len__(self):
        return 5*((len(self.dataset.patients)+7)//8)
    def __iter__(self):
        n=len(self.dataset.patients)
        for visit in range(5):
            order=np.arange(n)
            if self.shuffle:
                rng=np.random.default_rng(np.random.SeedSequence([self.dataset.seed,self.dataset.epoch,visit,987]))
                rng.shuffle(order)
            for start in range(0,n,8):
                yield (order[start:start+8]+visit*n).tolist()


def resolve_downstream(args):
    if args.force_rebuild_split or args.model not in ['early_fusion',*LEADS] or args.horizon!=365:
        raise ValueError('PCLR requires existing 365-day splits and a supported lead selection')
    site=args.architecture.split('_')[1]
    if args.site!=site:raise ValueError('This PCLR study uses same-source downstream only')
    if args.mimic_pretraining_size!=1359:
        raise ValueError('PCLR requires cohort1359')
    config,manifest,path,source=context(site,args.seed)
    config['architecture']['leads']=variant_leads(args.model)
    for key, value in [('ecg',args.ecg_path),('labels',args.labels_path)]:
        if Path(source[key]).resolve()!=Path(value).resolve():raise ValueError('Downstream source mismatch')
    folder=pretrain_dir(site,args.seed,args.model,args.pretraining_training_seed);expected=pretrain_signature(site,args.seed,args.model,args.pretraining_training_seed)
    if not (folder/'completed.json').is_file():raise FileNotFoundError(folder/'completed.json')
    if json.loads((folder/'completed.json').read_text())!=expected:raise ValueError('PCLR config mismatch')
    checkpoint=folder/'network.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['signature']!=expected:raise ValueError('PCLR checkpoint provenance mismatch')
    return config,dict(pretraining_source=site,pretraining_seed=args.seed,
        pretraining_training_seed=variant_seed(args.seed,args.pretraining_training_seed),pretraining_n_train_val=1359,
        pretraining_checkpoint=str(checkpoint),pretraining_config=str(folder/'run_config.json'),
        pretraining_cohort_manifest=str(path),pretraining_objective='PCLR',
        pretraining_checkpoint_sha256=sha256(checkpoint))
