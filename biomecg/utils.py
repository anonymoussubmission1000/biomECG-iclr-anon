"""Small utilities used by biometric pretraining."""

import json
import random
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from biomecg.paths import ROOT


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_base_path():
    return str(ROOT)


def create_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(dct, path, with_date=False):
    if with_date:
        raise ValueError("Use an explicit output filename")
    Path(path).write_text(json.dumps(dct, indent=2) + "\n")


def print_config_dict(config):
    print(json.dumps(config, indent=2))


def print_dataloader_info(name, loader):
    print(f"{name}: {len(loader.dataset)} pairs; {len(loader)} batches")


def plot_training_curves(config, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in ("Training", "Validation"):
        ax.plot([point[0] for point in config["log"]["history_loss"][key]], label=key)
    ax.set(xlabel="Epoch", ylabel="Loss")
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)
