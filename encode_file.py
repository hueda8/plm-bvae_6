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
    hparams["selection_already_applied"] = True

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
    lora_dir = os.path.join(ckpt_dir, 'lora_plm')
    ae_path = os.path.join(ckpt_dir, 'ae_head.pt')

    if os.path.isfile(model_path):
        ckpt = torch.load(model_path, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
        print(f"[ENCODE] Missing keys: {len(missing)}")
        print(f"[ENCODE] Unexpected keys: {len(unexpected)}")
        print(f"[ENCODE] Loaded checkpoint '{args.checkpoint_kind}': {model_path}")
    else:
        # LoRA-only bundle: lora_plm + ae_head.pt
        if not (bool(hparams.get("lora_enable", False)) and os.path.isdir(lora_dir) and os.path.isfile(ae_path)):
            raise FileNotFoundError(
                f"Neither full checkpoint nor LoRA bundle found.\n"
                f"  model.pt: {model_path}\n"
                f"  lora_plm: {lora_dir}\n"
                f"  ae_head.pt: {ae_path}"
            )

        from peft import PeftModel
  
        base_plm = model.plm_extractor.plm
        model.plm_extractor.plm = PeftModel.from_pretrained(base_plm, lora_dir).to(device)

        ae_ckpt = torch.load(ae_path, map_location=device)
        missing, unexpected = model.load_state_dict(ae_ckpt["ae"], strict=False)
        print(f"[ENCODE] (LoRA+AE) Missing keys: {len(missing)}")
        print(f"[ENCODE] (LoRA+AE) Unexpected keys: {len(unexpected)}")
        print(f"[ENCODE] Loaded LoRA bundle '{args.checkpoint_kind}': {lora_dir} + {ae_path}")

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

