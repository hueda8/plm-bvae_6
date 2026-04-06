# PyTorch-compatible hparams loader for Python 3.12
import sys
import yaml
import getpass
import os
from typing import Iterable, Optional, Set

def _flatten_hparams(d: dict, prefix: str = '', sep: str = '_', preserve_keys: Optional[Iterable[str]] = None) -> dict:
    """Recursively flatten a nested dict by joining keys with *sep*.

    This allows the YAML config file to use nested sections for readability
    while all downstream code continues to use flat keys unchanged.

    Example::

        {'plm': {'backend': 'esm2', 'batch_size': 8}, 'seed': 42}
        → {'plm_backend': 'esm2', 'plm_batch_size': 8, 'seed': 42}
    """
    out: dict = {}
    preserve_set: Set[str] = set(preserve_keys or [])
    for k, v in d.items():
        full_key = prefix + sep + k if prefix else k
        if full_key in preserve_set and isinstance(v, dict):
            out[full_key] = v
            continue
        if isinstance(v, dict):
            out.update(_flatten_hparams(v, full_key, sep, preserve_keys=preserve_set))
        else:
            out[full_key] = v
    return out

def replace_names(where, name, target):
    if isinstance(where, str) and name in where:
        return where.replace(name, target)
    if isinstance(where, dict):
        for k in list(where.keys()):
            where[k] = replace_names(where[k], name, target)
        return where
    if isinstance(where, list):
        for i in range(len(where)):
            where[i] = replace_names(where[i], name, target)
        return where
    return where

def loader_func(hparams_path, is_parent=False):
    try:
        with open(hparams_path, 'r') as f:
            hp = _flatten_hparams(yaml.safe_load(f), preserve_keys={"lr_schedule", "early_stopping"})
    except Exception:
        print('Cannot load', hparams_path)
        raise RuntimeError('cannot load hparams')

    if 'derive_from' in hp:
        parent = loader_func(hp['derive_from'], is_parent=True)
        parent.update(hp)
        hp = parent

    if ('user_name' not in hp) or hp['user_name'] == 'default':
        hp['user_name'] = getpass.getuser()

    # Defaults
    hp.setdefault('input_is_precomputed', True)
    hp.setdefault('embedding_dim', 2560)
    hp.setdefault('decoder_embedding_dim', hp.get('hidden_size', 2))
    hp.setdefault('vocab_size', 23)

    hp.setdefault('word_dropout', 0.0)
    hp.setdefault('label_smoothing', 0.0)
    hp.setdefault('seed', 42)
    hp.setdefault('input_noise_std', 0.0) # 'input_noise_std': 0.01,
    hp.setdefault('input_dropout_rate', 0.0) # 'input_dropout_rate': 0.10,
    hp.setdefault('early_stopping', None) # "early_stopping": {"patience": 10, "min_delta": 0.001},
    hp.setdefault('lr_schedule', None) # "lr_schedule": {"mode": "cosine", "warmup_epochs": 4, "min_lr_ratio": 0.1}, or {"mode": "plateau", "factor": 0.5, "patience": 3, "min_lr": 1e-6},
    hp.setdefault('use_preproj_mlp', False)
    hp.setdefault('preproj_hidden', None)
    hp.setdefault('deterministic', False)
    hp.setdefault('resume_checkpoint', None)
    hp.setdefault('compat_plm', True)
    hp.setdefault('select_indices', None)
    hp.setdefault('allow_sidecar_indices', False)
    hp.setdefault('decode_strategy', 'greedy')
    hp.setdefault('decode_top_p', 0.9)
    hp.setdefault('decode_temperature', 1.0)

    if not is_parent:
        for p1 in list(hp.keys()):
            hp = replace_names(hp, '<' + p1 + '>', str(hp[p1]))
    return hp

# If running this module directly, allow CLI-based loading as before.
if __name__ == "__main__":
    try:
        hparams = loader_func(sys.argv[1])
    except Exception:
        hparams = 'not specified'

def PrintHparamsInfo(hparams):
    ESCAPE_INFO = '\033[38;5;209m'
    ESCAPE_TITLE = '\033[38;5;123m'
    ESCAPE_DATA = '\033[38;5;72m'
    ESCAPE_FILE = '\033[38;5;118m'
    ESCAPE_OFF = '\033[0m'
    import __main__
    if isinstance(hparams, dict) and 'model_name' in hparams:
        print(ESCAPE_TITLE + 'Running ' + ESCAPE_FILE +  getattr(__main__, '__file__', '<stdin>') + ESCAPE_TITLE + '; model: ' + ESCAPE_INFO + str(hparams['model_name']))
        print(ESCAPE_OFF)
