"""Fetch public author checkpoints linked by the pinned ECG FM benchmark.
Sources: https://github.com/AI4HealthUOL/ecg-fm-benchmarking#-checkpoints
JEPA: https://github.com/sehunfromdaegu/ECG_JEPA
MERL: https://github.com/cheliu-computation/MERL-ICML2024
ST-MEM: https://github.com/vuno/ST-MEM
Source implementations are vendored separately; see foundation_assets/benchmark/UPSTREAM.json.
"""
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from biomecg.study.foundation_models_ecg1 import ASSETS, SPECS, provenance


def main():
    import gdown
    root=ASSETS/'checkpoints'
    cpc=root/'ecgcpc';cpc.mkdir(parents=True,exist_ok=True)
    if not (cpc/'last_11597276.ckpt').is_file():
        data=urllib.request.urlopen('https://ndownloader.figshare.com/files/58173919').read()
        if hashlib.md5(data).hexdigest()!='7cdd0d54786b4d98248afc6dce63beea':
            raise ValueError('CPC archive checksum differs from Figshare publication')
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name in archive.namelist():
                if Path(name).name in ('last_11597276.ckpt','config_last_11597276_ckpt.yaml'):
                    (cpc/Path(name).name).write_bytes(archive.read(name))
    ids={'ecgjepa':'1gMOT4xjQQg0GZkY1iE6NuDzua4ALw00l',
         'stmem':'14nScwPk35sFi8wc-cuLJLqudVwynKS0n','merl':'1NG__hPjm8_5c9BiUmCzjlC2Og6ArCtww'}
    for family,id in ids.items():
        path=root/SPECS[family]['checkpoint'];path.parent.mkdir(parents=True,exist_ok=True)
        if not path.is_file():
            temp=path.with_suffix('.download')
            result=gdown.download(id=id,output=str(temp),quiet=False)
            if result is None:raise RuntimeError(f'Download failed: {family}')
            temp.replace(path)
    manifest={family:provenance(family) for family in SPECS}
    (root/'foundation_sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for family,info in manifest.items():print(family,info['checkpoint_sha256'])


if __name__=='__main__':main()
