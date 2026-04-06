import os
import numpy as np
from tqdm import tqdm
import torch
import argparse

from featurizer import batch_generator, prepare_trimmed_cache
from model import SeqModel
from load_hparams import loader_func, PrintHparamsInfo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams_path", type=str, required=True, help="Path to hparams YAML")
    parser.add_argument("--checkpoint_kind", type=str, default="last", choices=["best", "last"],
                        help="Which checkpoint to load from hparams['model_path']: 'best' or 'last'.")
    args = parser.parse_args()

    hparams = loader_func(args.hparams_path)
    PrintHparamsInfo(hparams)

    # 強制的に precomputed を使用
    hparams['input_is_precomputed'] = True

    static_idx = hparams.get("select_indices", None)
    use_static = (static_idx is not None and len(static_idx) > 0)
    use_sidecar = bool(hparams.get("allow_sidecar_indices", False))
    need_trim_cache = use_static or use_sidecar

    if need_trim_cache:
        test_cache_dir = prepare_trimmed_cache(hparams, mode="test", force_rebuild=False)
        hparams["embeddings_path"] = os.path.dirname(test_cache_dir)
        print(f"[TRIM CACHE][AUTO][ENCODE] embeddings_path -> {hparams['embeddings_path']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SeqModel(hparams).to(device)
    ckpt_dir = os.path.join(hparams['model_path'], args.checkpoint_kind)
    model_path = os.path.join(ckpt_dir, 'model.pt')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    print(f"[ENCODE] Missing keys: {len(missing)}")
    print(f"[ENCODE] Unexpected keys: {len(unexpected)}")

    model.eval()
    print(f"[ENCODE] Loaded checkpoint '{args.checkpoint_kind}': {model_path}")

    os.makedirs(hparams['output_path'], exist_ok=True)
    out_path = os.path.join(hparams['output_path'], 'vectors.txt')

    gen = batch_generator  # precomputed 専用

    with open(out_path, 'w') as of, torch.no_grad():
        for batch in tqdm(gen(hparams, 'test', keep_order=True), desc='encode'):
            batch_x, _ = batch
            x = torch.from_numpy(batch_x).float().to(device)
            latent, _ = model(x, 'encode')
            vecs = latent.detach().cpu().numpy()
            if hparams.get('latent_type') == 'binary':
                vecs = (vecs * 1.1).astype('int32')
            for v in vecs:
                of.write(' '.join(str(float(a)) for a in v) + '\n')

    print("Saved latent vectors to", out_path)


