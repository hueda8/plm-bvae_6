import os
import numpy as np
from tqdm import tqdm
import torch
import argparse

from featurizer import Encoder
from model import SeqModel
from load_hparams import loader_func, PrintHparamsInfo

def _flatten_int_list(arr) -> list:
    """Normalize tensor/ndarray/nested lists to a flat Python list[int]."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().view(-1).tolist()
    elif isinstance(arr, np.ndarray):
        arr = arr.reshape(-1).tolist()
    # Now arr is a (possibly nested) Python list
    try:
        flat = np.array(arr, dtype=object).reshape(-1).tolist()
    except Exception:
        flat = list(arr)
    # cast to int
    out = []
    for v in flat:
        try:
            out.append(int(v))
        except Exception:
            # if it's a singleton list like [x]
            try:
                out.append(int(np.array(v).reshape(()).item()))
            except Exception:
                continue
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams_path", type=str, required=True, help="Path to hparams YAML")
    parser.add_argument("--checkpoint_kind", type=str, default="last", choices=["best", "last"],
                        help="Which checkpoint to load from hparams['model_path']: 'best' or 'last'.")
    # 追加: decode オプション
    parser.add_argument("--decode_strategy", type=str, default="greedy", choices=["greedy", "top_p"])
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    
    args = parser.parse_args()

    hparams = loader_func(args.hparams_path)
    PrintHparamsInfo(hparams)

    # CLIが指定されていたら hparams を上書き
    if args.decode_strategy is not None:
        hparams["decode_strategy"] = args.decode_strategy
    if args.top_p is not None:
        hparams["decode_top_p"] = float(args.top_p)
    if args.temperature is not None:
        hparams["decode_temperature"] = float(args.temperature)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = SeqModel(hparams).to(device)
    ckpt_dir = os.path.join(hparams['model_path'], args.checkpoint_kind)
    model_path = os.path.join(ckpt_dir, 'model.pt')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    print(f"[DECODE] Missing keys: {len(missing)}")
    print(f"[DECODE] Unexpected keys: {len(unexpected)}")

    model.eval()
    print(f"[DECODE] Loaded checkpoint '{args.checkpoint_kind}': {model_path}")
    
    text_encoder = Encoder(hparams)

    # Resolve I/O paths
    vectors_path = os.path.join(hparams['output_path'], 'vectors.txt')
    # Create output file
    opath = os.path.join(hparams['output_path'], 'decoded.txt')
    os.makedirs(hparams['output_path'], exist_ok=True)

    # Decode sequences
    with torch.no_grad():
        with open(vectors_path, 'r') as f, open(opath, 'w') as ofile:
            for line in tqdm(f, desc='decode'):
                vec = np.array([float(val) for val in line.strip().split()], dtype=np.float32)
                z = torch.from_numpy(vec[None, :]).to(device)
                out = model(z, 'decode')
                ints = _flatten_int_list(out[0])
                sentence = text_encoder.decode(ints)
                ofile.write(sentence + '\n')

    print('Saved as', opath)


