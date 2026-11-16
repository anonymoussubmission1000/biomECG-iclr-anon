"""Load anonymous tensor-only local SSL weights for representation extraction.

Sources: model implementations and adaptation summaries in pclr_ecg1.py,
jepa_resnet_ecg1.py and cpc_local_ecg1.py. Pretraining heads are discarded.
"""
from pathlib import Path
import torch

def load_encoder(path, family, leads=8):
    """Return an eval-mode encoder; caller applies the documented preprocessing."""
    state=torch.load(Path(path),map_location='cpu',weights_only=True)
    if not all(isinstance(v,torch.Tensor) for v in state.values()):
        raise ValueError('Expected a tensor-only state dictionary')
    if family=='jepa':
        from .jepa_resnet_ecg1 import new_encoder
        encoder=new_encoder(leads);encoder.load_state_dict(state,strict=True)
    elif family=='pclr':
        from .pclr_ecg1 import PCLRModel
        model=PCLRModel(leads);model.load_state_dict(state,strict=True);encoder=model.encoder
    elif family=='cpc':
        from .cpc_local_ecg1 import CPCModel
        model=CPCModel(leads);model.load_state_dict(state,strict=True);encoder=model.encoder
    else:
        raise ValueError('Use the biometric model loader for biometric checkpoints')
    return encoder.eval()
