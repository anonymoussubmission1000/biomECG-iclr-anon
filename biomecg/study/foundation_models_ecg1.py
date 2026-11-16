"""Native ECG foundation encoders for the ECG1 mortality protocol.

Sources (vendored verbatim at foundation_assets/benchmark, see UPSTREAM.json):
Benchmark: https://github.com/AI4HealthUOL/ecg-fm-benchmarking
Paper: https://arxiv.org/abs/2509.25095 (Tables 1, 32, 33)
CPC: benchmark code/clinical_ts/ts/{encoder,s4}.py and CPCWrapper.get_params
JEPA: https://github.com/sehunfromdaegu/ECG_JEPA
MERL: https://github.com/cheliu-computation/MERL-ICML2024
ST-MEM: https://github.com/vuno/ST-MEM

Use the benchmark's pooled features and depth-dependent LR groups, but our
common mortality MLP plus age/sex. No attention head, no lead-wise ensemble.
All encoder state keys/shapes and optimizer parameter coverage are checked.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'foundation_assets'
BENCHMARK = ASSETS / 'benchmark'
sys.path.insert(0, str(BENCHMARK / 'code'))
SPECS = {
    'ecgcpc': dict(fs=240, samples=600, dim=512, checkpoint='ecgcpc/last_11597276.ckpt'),
    'ecgjepa': dict(fs=250, samples=2500, dim=768, checkpoint='ecgjepa/multiblock_epoch100.pth'),
    'merl': dict(fs=500, samples=1250, dim=512, checkpoint='merl/res18_best_encoder.pth'),
    'stmem': dict(fs=250, samples=600, dim=768, checkpoint='stmem/st_mem_vit_base_full.pth'),
}


def checkpoint_path(family):
    return ASSETS / 'checkpoints' / SPECS[family]['checkpoint']


@lru_cache(maxsize=16)
def _checkpoint_digest(path, size, mtime_ns):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def provenance(family):
    path = checkpoint_path(family)
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(f'{path}: run biomecg/study/setup_ecg_foundation_models.py first')
    stat = path.stat()
    digest = _checkpoint_digest(path, stat.st_size, stat.st_mtime_ns)
    return dict(foundation_model=family, pretraining_checkpoint=str(path),
                checkpoint_sha256=digest,
                benchmark_commit=json.loads((BENCHMARK/'UPSTREAM.json').read_text())['commit'])


def strict_load(module, state):
    expected = module.state_dict()
    missing, extra = set(expected)-set(state), set(state)-set(expected)
    wrong = [k for k in set(expected)&set(state) if expected[k].shape != state[k].shape]
    if missing or extra or wrong:
        raise ValueError(f'Checkpoint mismatch: missing={sorted(missing)}, extra={sorted(extra)}, shapes={wrong}')
    # S4 initializes expanded buffers with shared storage; materialize before copy.
    for submodule in module.modules():
        for name, value in submodule._buffers.items():
            if value is not None:
                submodule._buffers[name] = value.clone()
    module.load_state_dict(state, strict=True)


def torch_cauchy_conj(v, z, w):
    """Exact conjugate-pair Cauchy sum, PyTorch autograd; avoids a CUDA extension.

    N=8 in this checkpoint makes the explicit sum small. Algebraically identical
    to the upstream KeOps expression; no approximation or learned-weight change.
    """
    return (v.unsqueeze(-1) / (z - w.unsqueeze(-1)) +
            v.conj().unsqueeze(-1) / (z - w.conj().unsqueeze(-1))).sum(dim=-2)


class FoundationEncoder(nn.Module):
    def __init__(self, family):
        super().__init__()
        self.family = family
        self.spec = SPECS[family]
        path = checkpoint_path(family)
        if not path.is_file():
            raise FileNotFoundError(path)
        # Public author checkpoints (CPC includes configuration dataclasses).
        state = torch.load(path, map_location='cpu', weights_only=False)
        if family == 'ecgjepa':
            from clinical_ts.models.ecg_foundation_models.ecg_jepa.ecg_jepa import ecg_jepa
            self.network = ecg_jepa(encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=16,
                predictor_embed_dim=384, predictor_depth=6, predictor_num_heads=12,
                c=8, pos_type='sincos', mask_scale=(0, 0), leads=list(range(8))).encoder
            strict_load(self.network, state['encoder'])
        elif family == 'merl':
            from clinical_ts.models.ecg_foundation_models.merl.resnet1d import ResNet18
            self.network = ResNet18(num_classes=1)
            self.network.linear = nn.Identity()
            strict_load(self.network, {k:v for k,v in state.items() if not k.startswith('linear.')})
        elif family == 'stmem':
            from clinical_ts.models.ecg_foundation_models.st_mem.encoder.st_mem_vit import st_mem_vit_base
            self.network = st_mem_vit_base(num_leads=12)
            strict_load(self.network, {k.removeprefix('encoder.'):v for k,v in state['model'].items()
                                       if k.startswith('encoder.')})
        else:
            from clinical_ts.template_modules import ShapeConfig
            from clinical_ts.ts.encoder import RNNEncoder, RNNEncoderConfig
            from clinical_ts.ts.s4 import S4Predictor, S4PredictorConfig
            # Preserve the pretrained S4 kernel length / C correction (omega stores L/2+1 nodes).
            kernel_length = 2 * (state['state_dict']['ts_encoder.predictor.predictor.s4_layers.0.kernel.kernel.omega'].shape[0]-1)
            shape = ShapeConfig(channels=12, length=kernel_length*2, sequence_last=True)
            from clinical_ts.ts.s4_modules import s42
            s42.cauchy_conj = torch_cauchy_conj
            self.conv = RNNEncoder(RNNEncoderConfig(features=[512]*4, kss=[3,1,1,1],
                strides=[2,1,1,1]), shape, None)
            self.temporal = S4Predictor(S4PredictorConfig(causal=True, state_dim=8, model_dim=512,
                backbone='s42'), self.conv.get_output_shape())
            weights = state['state_dict']
            strict_load(self.conv, {k.removeprefix('ts_encoder.encoder.'):v for k,v in weights.items()
                                   if k.startswith('ts_encoder.encoder.')})
            strict_load(self.temporal, {k.removeprefix('ts_encoder.predictor.'):v for k,v in weights.items()
                                       if k.startswith('ts_encoder.predictor.')})

    def forward(self, x):
        if self.family == 'ecgjepa':
            return self.network.representation(x[:, [0,1,6,7,8,9,10,11]])
        if self.family == 'ecgcpc':
            return self.temporal(**self.conv(seq=x))['seq'].mean(dim=1)
        return self.network(x)

    def depth_groups(self):
        """Same grouping as benchmark get_params, with exhaustive coverage checks."""
        early, late = [], []
        if self.family == 'ecgcpc':
            early, late = list(self.conv.parameters()), list(self.temporal.parameters())
        else:
            for name, p in self.network.named_parameters():
                if not p.requires_grad:
                    continue
                if self.family == 'merl':
                    low = name.startswith(('conv1.', 'bn1.', 'layer1.', 'layer2.'))
                    high = name.startswith(('layer3.', 'layer4.'))
                elif self.family == 'ecgjepa':
                    low = name == 'pos_embed' or name.startswith('W_P.') or any(
                        name.startswith(f'encoder_blocks.blocks.{i}.') for i in range(3))
                    high = name.startswith('norm.') or any(
                        name.startswith(f'encoder_blocks.blocks.{i}.') for i in range(3,12))
                else:
                    # Correct names relative to the encoder (upstream wrapper uses an extra encoder. prefix).
                    low = name in ('pos_embedding','sep_embedding') or name.startswith(
                        ('lead_embeddings.','to_patch_embedding.')) or any(name.startswith(f'block{i}.') for i in range(3))
                    high = name.startswith('norm.') or any(name.startswith(f'block{i}.') for i in range(3,12))
                if not (low or high):
                    raise ValueError(f'Unassigned {self.family} parameter: {name}')
                (early if low else late).append(p)
        early = [p for p in early if p.requires_grad]
        late = [p for p in late if p.requires_grad]
        actual = [id(p) for p in early+late]
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        if len(actual) != len(set(actual)) or set(actual) != expected or not early or not late:
            raise ValueError('Incomplete or overlapping depth LR groups')
        return early, late


class FoundationMortalityModel(nn.Module):
    def __init__(self, encoder, frozen):
        super().__init__()
        self.encoder, self.frozen = encoder, frozen
        self.head = nn.Sequential(nn.Linear(encoder.spec['dim']+2,256),nn.ReLU(),
                                  nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
        if frozen:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, ecg, metadata):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            embedding = self.encoder(ecg)
        if embedding.ndim != 2 or embedding.shape[1] != self.encoder.spec['dim']:
            raise ValueError(f'Unexpected feature shape {embedding.shape}')
        return self.head(torch.cat([embedding,metadata],dim=1)).squeeze(1)

    def parameter_groups(self, lr_head):
        groups = [dict(params=list(self.head.parameters()), lr=lr_head, name='head')]
        if not self.frozen:
            early, late = self.encoder.depth_groups()
            groups += [dict(params=early,lr=lr_head*0.01,name='early_encoder'),
                       dict(params=late,lr=lr_head*0.1,name='late_encoder')]
        return groups
