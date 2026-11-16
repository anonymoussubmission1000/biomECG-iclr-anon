"""JEPA objective with the exact biomECG eight-lead early ResNet18 encoder.

Sources: https://arxiv.org/html/2605.12241v1
https://github.com/AI4HealthUOL/ecg-pretraining-strategies
Pinned upstream commit: ec4e7eb744f979908cb649207840cc184af72587
Reference files: pretraining/code/clinical_ts/{mask/jepa_multiblock_mask.py,
loss/selfsupervised.py,ts/ema.py}, conf/config_jepa_ecg_s4.yaml.

Local adaptation, NOT the original ECG-JEPA model: 8 leads, 4096 samples/500Hz,
our unchanged fixed-width64 ResNet18, raw-input masking, proportionally scaled min_keep,
a small convolutional SSL predictor instead of S4, safe disjoint context,
visible-only input normalization, clean train-only BatchNorm calibration.
See README_jepa_resnet_ecg1.md for the complete reproducibility contract.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from biomecg.study import pclr_ecg1 as cohort

ROOT = Path(__file__).resolve().parents[2]
SEEDS = cohort.SEEDS
PROTOCOL = 'jepa_resnet_ecg1_v4_early64_fixedbudget_seeded'
UPSTREAM_COMMIT = 'ec4e7eb744f979908cb649207840cc184af72587'
SAMPLES, TOKENS, DIM, STRIDE = 4096, 128, 64, 32
BACKBONE_WIDTH = 64
PREDICTOR_DIM = 64
REPRESENTATION_DIM = 2 * DIM
ENCODER_PARAMS = 212992
DEFAULT_EPOCHS = 200  # local comparison budget, matching CPC maximum
PATIENCE = None  # no pretraining early stopping, as in upstream main_all.py


def new_encoder(input_channels=8):
    from biomecg.core.architecture.resnet1d import SiameseResNet1d
    encoder = SiameseResNet1d(input_channels=input_channels, embedding_dim=256,
                             fix_feature_dim=True, kernel_size=3,
                             dilation_rate=1, dropout=0.).encoder
    if sum(p.numel() for p in encoder.parameters()) != (ENCODER_PARAMS if input_channels == 8 else 211648):
        raise AssertionError('JEPA encoder must equal the biomECG early encoder')
    return encoder


def temporal_features(encoder, x):
    for layer in list(encoder.children())[:-1]:
        x = layer(x)
    return x.transpose(1, 2)  # [batch, 128, 64], before historical max+mean pooling


def pretrain_dir(site, seed, model='early_fusion', training_seed=None):
    cohort.variant_leads(model)
    if model != 'early_fusion' or cohort.variant_seed(seed,training_seed) != seed:
        return ROOT/'results/JEPAResNetEarly/ensemble_checkpoints'/f'{site}_cohort1359_seed{seed}'/model/f'trainseed{cohort.variant_seed(seed,training_seed)}'
    return ROOT / 'results/JEPAResNetEarly/checkpoints' / f'{site}_8l_cohort1359_seed{seed}'


def pretrain_signature(site, seed, epochs=DEFAULT_EPOCHS, model='early_fusion', training_seed=None):
    if seed not in SEEDS or site not in ('mimic', 'code15') or epochs < 1:
        raise ValueError('Invalid JEPA site/seed/epochs')
    config, manifest, path, source = cohort.context(site, seed)
    if [len(manifest[g+'_patients']) for g in ('train', 'val')] != [1087, 272]:
        raise ValueError('Expected cohort1359 = 1087 train + 272 validation')
    paths = {}
    for key in ('ecg', 'labels'):
        p = Path(source[key]).resolve()
        paths[key] = dict(path=str(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
    result = dict(protocol=PROTOCOL, upstream_commit=UPSTREAM_COMMIT,
                site=site, seed=seed, epochs=epochs, patience=PATIENCE, cohort_size=1359,
                manifest=str(path), manifest_sha256=cohort.sha256(path), inputs=paths,
                leads=cohort.LEADS, samples=SAMPLES, fs=500, encoder='biomECG_ResNet1d18_fixed64_exact',
                encoder_params=ENCODER_PARAMS, initialization="reseed after cache, before model",
                representation_dim=REPRESENTATION_DIM, tokens=TOKENS, temporal_dim=DIM,
                context_scale=[.85, 1.], target_scale=[.15, .20], target_blocks=4,
                min_keep=math.ceil(64*TOKENS/300), target_blocks_may_overlap=True,
                context_target_overlap=False, masking='waveform before normalization/encoder; context tokens only to predictor',
                normalization='teacher all samples; student visible samples only; per-lead zscore epsilon1e-8',
                predictor='4 residual Conv1d64 k3 dilations1,2,4,8 + LayerNorm/GELU + Conv1d64to64 k1; fixed sinusoidal positions',
                loss='masked smooth_l1 beta1; repeated targets weighted per block',
                target_normalization='two feature-wise LayerNorms without affine',
                ema_momentum=.9996, teacher_mode='eval on validation; parameter EMA; teacher BN updated on clean train only',
                batch_size=64, optimizer='AdamW', lr_predictor=.003,
                lr_residual=.0003, lr_stem=.00003, weight_decay=.001,
                scheduler='constant', precision='float32',
                sampling='all eligible train ECGs once per epoch; no biometric test patients',
                selection='minimum deterministic validation JEPA loss',
                downstream_encoder='minimum-val-loss online encoder after fixed budget, all BN calibrated on clean train only')
    return cohort.variant_signature(result,seed,model,training_seed)


def sample_masks(batch, generator, tokens=TOKENS):
    """Upstream block scales/overlapping target draws; forbid fallback leakage.

    min_keep scales from upstream's 64/300 tokens to our 28/128. Unlike upstream,
    insufficient context triggers a redraw, never a random fallback into targets.
    """
    min_keep = math.ceil(64*tokens/300)
    target_size = max(min_keep, int(tokens*(.15+.05*torch.rand((), generator=generator).item())))
    context_size = min(tokens-1, int(tokens*(.85+.15*torch.rand((), generator=generator).item())))
    positions = torch.arange(tokens)
    contexts, targets = [], []
    for _ in range(batch):
        for attempt in range(1000):
            starts = torch.randint(tokens-target_size+1, (4,), generator=generator)
            pred = (positions[None] >= starts[:, None]) & (positions[None] < starts[:, None]+target_size)
            start = int(torch.randint(tokens-context_size+1, (), generator=generator))
            context = (positions >= start) & (positions < start+context_size) & ~pred.any(0)
            if context.sum() >= min_keep:
                break
        else:
            raise RuntimeError('Cannot draw valid JEPA context; no leakage fallback allowed')
        contexts.append(context); targets.append(pred)
    return torch.stack(contexts), torch.stack(targets)


def normalize_signal(x, visible=None):
    if visible is None:
        return (x-x.mean(-1, keepdim=True))/(x.std(-1, keepdim=True, correction=0)+1e-8)
    # Mask before any statistics: changing hidden raw values cannot affect student input.
    visible = visible[:, None, :].expand_as(x)
    safe = x.masked_fill(~visible, 0.)
    count = visible.sum(-1, keepdim=True).clamp_min(1)
    mean = safe.sum(-1, keepdim=True)/count
    centered = (safe-mean).masked_fill(~visible, 0.)
    std = (centered.square().sum(-1, keepdim=True)/count).sqrt()
    return centered/(std+1e-8)


class TemporalPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(PREDICTOR_DIM))
        self.norms = nn.ModuleList([nn.LayerNorm(PREDICTOR_DIM) for _ in range(4)])
        self.convs = nn.ModuleList([nn.Conv1d(PREDICTOR_DIM, PREDICTOR_DIM, 3, padding=d, dilation=d)
                                   for d in (1, 2, 4, 8)])
        self.output = nn.Conv1d(PREDICTOR_DIM, DIM, 1)
        position = torch.arange(TOKENS)[:, None]
        freq = torch.exp(torch.arange(0, PREDICTOR_DIM, 2)*(-math.log(10000.)/PREDICTOR_DIM))
        pe = torch.zeros(TOKENS, PREDICTOR_DIM)
        pe[:, 0::2] = torch.sin(position*freq); pe[:, 1::2] = torch.cos(position*freq)
        self.register_buffer('position', pe)
        nn.init.normal_(self.mask_token, std=.02)

    def forward(self, features, context):
        z = torch.where(context[..., None], self.input(features), self.mask_token) + self.position
        for norm, conv in zip(self.norms, self.convs):
            z = z + conv(F.gelu(norm(z)).transpose(1, 2)).transpose(1, 2)
        return self.output(z.transpose(1, 2)).transpose(1, 2)


def jepa_loss(prediction, target, targets):
    if prediction.shape != target.shape or targets.shape != (len(prediction), 4, prediction.shape[1]):
        raise ValueError('JEPA loss shape mismatch')
    weights = targets.sum(1).to(prediction.dtype)
    if weights.sum() == 0:
        raise ValueError('Empty JEPA targets')
    return (F.smooth_l1_loss(prediction, target.detach(), reduction='none', beta=1.).mean(-1)*weights).sum()/weights.sum()


class JEPAResNet(nn.Module):
    def __init__(self, input_channels=8):
        super().__init__()
        self.input_channels = input_channels
        self.encoder = new_encoder(input_channels)
        self.teacher = deepcopy(self.encoder).requires_grad_(False).eval()
        self.predictor = TemporalPredictor()

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def parameter_groups(self):
        children = list(self.encoder.children())
        return [dict(params=self.predictor.parameters(), lr=.003, name='predictor'),
                dict(params=[p for m in children[4:] for p in m.parameters()], lr=.0003, name='residual'),
                dict(params=[p for m in children[:4] for p in m.parameters()], lr=.00003, name='stem')]

    @torch.no_grad()
    def update_teacher(self):
        for target, online in zip(self.teacher.parameters(), self.encoder.parameters()):
            target.lerp_(online, 1-.9996)
        # BN buffers are updated on CLEAN training signals in forward below.

    def forward(self, x, context, targets):
        if x.shape[1:] != (self.input_channels, SAMPLES):
            raise ValueError(f'Expected {self.input_channels} leads and {SAMPLES} samples')
        if context.shape != (len(x), TOKENS) or targets.shape != (len(x), 4, TOKENS):
            raise ValueError('Invalid JEPA mask shapes')
        if (context & targets.any(1)).any() or not context.any(1).all():
            raise ValueError('JEPA context overlaps targets or is empty')
        visible = context.repeat_interleave(STRIDE, -1)
        online_features = temporal_features(self.encoder, normalize_signal(x, visible))
        with torch.no_grad():
            # Clean teacher BN buffers use TRAIN data only, never validation.
            if self.training:
                self.teacher.train()
            teacher_features = temporal_features(self.teacher, normalize_signal(x))
            self.teacher.eval()
            target = F.layer_norm(teacher_features, (DIM,))
            target = F.layer_norm(target, (DIM,))
        predicted = self.predictor(online_features, context)
        loss = jepa_loss(predicted, target, targets)
        return loss, online_features.detach().mean(1), target.detach().mean(1)


class ECGRecords(Dataset):
    """Every eligible ECG once, using saved patient manifests; raw selected leads."""
    def __init__(self, manifest, source, group):
        self.path = source['ecg']; self.rows = []; self.file = None
        with h5py.File(self.path, 'r') as f:
            if f['ECG'].shape[1:] != (12, SAMPLES):
                raise ValueError('Expected original 12x4096 ECGs')
            if int(f.attrs.get('sampling_rate', 500)) != 500:
                raise ValueError('Expected original 500Hz ECGs')
            lookup = {str(s): i for i, s in enumerate(f['SID'][:])}; pids = f['PID'][:]
            for pid in manifest[group+'_patients']:
                for sid in manifest['valid_sids'][pid]:
                    row = lookup[str(sid)]
                    if str(pids[row]) != pid: raise ValueError('SID/PID mismatch')
                    self.rows.append(row)
        if len(set(self.rows)) != len(self.rows): raise ValueError('Duplicate ECG')

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        if self.file is None: self.file = h5py.File(self.path, 'r')
        x = self.file['ECG'][self.rows[index]][cohort.INDICES].astype(np.float32)
        if not np.isfinite(x).all(): raise ValueError('Nonfinite ECG')
        return torch.from_numpy(x)

    def close(self):
        if self.file is not None: self.file.close(); self.file = None

    def __getstate__(self):
        state = self.__dict__.copy(); state['file'] = None; return state


class CachedECGs(Dataset):
    def __init__(self, path):
        self.path = str(path); self.array = None
    def _open(self):
        if self.array is None: self.array = np.load(self.path, mmap_mode='r')
        return self.array
    def __len__(self): return len(self._open())
    def __getitem__(self, index): return torch.from_numpy(self._open()[index].copy())
    def __getstate__(self): return dict(path=self.path, array=None)


def cached_records(site, seed, group, signature, workers):
    from torch.utils.data import DataLoader
    expected = {k: signature[k] for k in ('protocol', 'manifest_sha256', 'inputs', 'leads', 'samples')}
    expected['group'] = group
    key = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()[:16]
    directory = ROOT / 'results/JEPAResNet/cache' / f'{site}_seed{seed}_{key}'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{group}.npy'; marker = directory / f'{group}.json'
    if marker.exists():
        if json.loads(marker.read_text()) != expected or not path.exists(): raise ValueError('Cache mismatch')
        return CachedECGs(path)
    _, manifest, _, source = cohort.context(site, seed)
    dataset = ECGRecords(manifest, source, group)
    tmp = path.with_suffix('.tmp.npy')
    array = np.lib.format.open_memmap(tmp, mode='w+', dtype='float32', shape=(len(dataset), 8, SAMPLES))
    offset = 0
    for x in DataLoader(dataset, batch_size=64, num_workers=workers):
        array[offset:offset+len(x)] = x.numpy(); offset += len(x)
    array.flush(); del array; dataset.close()
    if offset != len(dataset): raise ValueError('Incomplete cache')
    tmp.replace(path); cohort.atomic_json(marker, expected)
    return CachedECGs(path)


def resolve_downstream(args):
    if args.force_rebuild_split or args.model not in ['early_fusion',*cohort.LEADS] or args.horizon != 365:
        raise ValueError('JEPA requires existing 365-day splits and a supported lead selection')
    if args.architecture != f'jepa_resnet_early_{args.site}_frozen':
        raise ValueError('JEPA study supports same-site frozen only')
    if args.mimic_pretraining_size != 1359:
        raise ValueError('JEPA requires cohort1359')
    config, _, manifest_path, source = cohort.context(args.site, args.seed)
    for key, path in [('ecg', args.ecg_path), ('labels', args.labels_path)]:
        if Path(source[key]).resolve() != Path(path).resolve(): raise ValueError('Downstream source path mismatch')
    config['architecture']['leads'] = cohort.variant_leads(args.model)
    folder = pretrain_dir(args.site, args.seed,args.model,args.pretraining_training_seed)
    saved = json.loads((folder/'completed.json').read_text())
    expected = pretrain_signature(args.site, args.seed, epochs=saved['epochs'],model=args.model,training_seed=args.pretraining_training_seed)
    if saved != expected: raise ValueError('JEPA pretraining configuration changed')
    state = torch.load(folder/'network.pt', map_location='cpu', weights_only=True)
    if state['signature'] != expected or not state.get('bn_calibrated', False):
        raise ValueError('Invalid or uncalibrated JEPA checkpoint')
    return config, dict(pretraining_source=args.site, pretraining_seed=args.seed,
        pretraining_training_seed=cohort.variant_seed(args.seed,args.pretraining_training_seed), pretraining_n_train_val=1359,
        pretraining_checkpoint=str(folder/'network.pt'), pretraining_config=str(folder/'run_config.json'),
        pretraining_cohort_manifest=str(manifest_path), pretraining_objective='JEPA-ResNet',
        pretraining_checkpoint_sha256=cohort.sha256(folder/'network.pt'))


class SelectedLeads(Dataset):
    def __init__(self, dataset, model):
        self.dataset=dataset
        self.indices=[cohort.LEADS.index(lead) for lead in cohort.variant_leads(model)]
    def __len__(self): return len(self.dataset)
    def __getitem__(self,index): return self.dataset[index][self.indices]
