"""Content-addressed frozen ECG embeddings; no changes to head training batches."""
import fcntl
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def cached_frozen_loaders(model, loaders, args, device, cache_root):
    if not model.frozen:
        raise ValueError('Embedding cache requires a frozen encoder')
    # Extraction must not consume the RNG stream used by head training/shuffling.
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    training = model.training
    result = []
    try:
        digest = hashlib.sha256()
        for name, value in model.encoder.state_dict().items():
            value = value.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        encoder_sha = digest.hexdigest()
        model.eval()
        for original in loaders:
            dataset = original.dataset
            identity = dict(protocol='frozen_embeddings_v1', encoder_sha256=encoder_sha,
                            batch_size=args.bs, precision=str(getattr(model, 'amp_dtype', torch.float16)),
                            device_type=device.type, torch_version=str(torch.__version__),
                            leads=list(map(int, dataset.lead_indices)))
            for key in ('ecg_path', 'labels_path'):
                path = Path(getattr(args, key)).resolve()
                stat = path.stat()
                identity[key] = [str(path), stat.st_size, stat.st_mtime_ns]
            identity['rows_sha256'] = hashlib.sha256(np.asarray(dataset.rows, dtype=np.int64).tobytes()).hexdigest()
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            cache_root = Path(cache_root)
            cache_root.mkdir(parents=True, exist_ok=True)
            path = cache_root / f'{key}.npz'
            with (cache_root / f'{key}.lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not path.exists():
                    extraction = DataLoader(dataset, batch_size=args.bs, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=device.type == 'cuda')
                    chunks = [[], [], [], []]
                    with torch.no_grad():
                        for ecg, metadata, target, rows in extraction:
                            with torch.autocast(device_type=device.type,
                                                dtype=getattr(model, 'amp_dtype', torch.float16),
                                                enabled=device.type == 'cuda'):
                                features = model.encoder(ecg.to(device, non_blocking=True))
                            for sink, tensor in zip(chunks, (features.float().cpu(), metadata, target, rows)):
                                sink.append(tensor)
                    arrays = [torch.cat(chunk).numpy() for chunk in chunks]
                    if not np.isfinite(arrays[0]).all():
                        raise ValueError('Nonfinite frozen embeddings')
                    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
                    try:
                        with temporary.open('wb') as stream:
                            np.savez(stream, features=arrays[0], metadata=arrays[1], targets=arrays[2],
                                     rows=arrays[3], identity=json.dumps(identity, sort_keys=True))
                        os.replace(temporary, path)
                    finally:
                        temporary.unlink(missing_ok=True)
                    print(f'Created frozen embedding cache: {path}', flush=True)
                else:
                    print(f'Reusing frozen embedding cache: {path}', flush=True)
                with np.load(path, allow_pickle=False) as archive:
                    if json.loads(str(archive['identity'])) != identity:
                        raise ValueError(f'Cache provenance mismatch: {path}')
                    if not np.array_equal(archive['rows'], np.asarray(dataset.rows)):
                        raise ValueError(f'Cache row mismatch: {path}')
                    tensors = [torch.from_numpy(archive[name].copy()) for name in
                               ('features', 'metadata', 'targets', 'rows')]
                result.append(DataLoader(TensorDataset(*tensors), batch_size=args.bs,
                                         shuffle=isinstance(original.sampler, torch.utils.data.RandomSampler),
                                         num_workers=0, pin_memory=device.type == 'cuda'))
        model.cached_embeddings = True
        return tuple(result)
    finally:
        model.train(training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
