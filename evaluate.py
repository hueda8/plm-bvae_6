import os
import json
import numpy as np
from tqdm import tqdm

import torch
from featurizer import batch_generator
from model import SeqModel
from load_hparams import loader_func, PrintHparamsInfo  # ← hparams ではなく loader_func を使う
import random
import argparse

def set_randomseeds(s: int, deterministic: bool = False):
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    # Optional: HuggingFace transformers
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(s)
    except Exception:
        pass
    # Deterministic behavior
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams_path", type=str, required=True)
    parser.add_argument("--checkpoint_kind", type=str, default="best", choices=["best", "last"],
                        help="Which checkpoint to load from hparams['model_path']: 'best' or 'last'.")
    args = parser.parse_args()

    hparams = loader_func(args.hparams_path)
    PrintHparamsInfo(hparams)

    # 事前計算済み埋め込みのみを使用するように強制
    hparams['input_is_precomputed'] = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed_value = int(hparams.get('seed', 42))
    deterministic = bool(hparams.get('deterministic', False))
    set_randomseeds(seed_value, deterministic=deterministic)

    if device.type == 'cuda':
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(hparams.get('allow_tf32', True))
        except Exception:
            pass

    model = SeqModel(hparams).to(device)
    ckpt_dir = os.path.join(hparams['model_path'], args.checkpoint_kind)
    model_path = os.path.join(ckpt_dir, 'model.pt')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    print(f"[EVAL] Missing keys: {len(missing)}")
    print(f"[EVAL] Unexpected keys: {len(unexpected)}")

    model.eval()
    print(f"[EVAL] Loaded checkpoint '{args.checkpoint_kind}': {model_path}")
    
    metrics = {}
    gen = batch_generator
    num_batches = sum(1 for _ in gen(hparams, 'test'))
    print(num_batches, 'batches')
    
    for mode in ['eval', 'eval_sample']:
        for batch in tqdm(gen(hparams, 'test'), total=num_batches, desc=f"test {mode}"):
            batch_x, batch_y = batch
            x = torch.from_numpy(batch_x).float().to(device)
            y = torch.from_numpy(batch_y).long().to(device)
            with torch.no_grad():
                _, losses = model(x, mode, y)
            for l, v in losses.items():
                key = l + ('_s' if mode == 'eval_sample' else '')
                metrics[key] = metrics.get(key, []) + [float(v.detach().cpu().item())]

    metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
    print('test loss:', metrics)

    os.makedirs(hparams['output_path'], exist_ok=True)
    with open(os.path.join(hparams['output_path'], 'test_metrics.txt'), 'w') as f:
        f.write(json.dumps(metrics))


