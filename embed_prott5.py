import os
import yaml
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import List
from collections import Counter
from transformers import T5Tokenizer, T5EncoderModel
from load_hparams import _flatten_hparams

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
MAX_TOKENS = 1022  # ProtT5 でも安全側の上限

def clean_sequence(seq: str, mode: str = "prott5"):
    raw = seq.strip().upper().replace(" ", "")
    if not raw:
        return ""
    if mode == "strict":
        return "".join([c if c in VALID_AA else 'X' for c in raw])
    elif mode == "prott5":
        # ProtT5推奨: 非標準は X で統一
        return "".join([c if c in VALID_AA else 'X' for c in raw])
    else:
        raise ValueError(f"Unknown clean mode: {mode}")

def to_spaced(seq: str) -> str:
    # ProtT5 は各アミノ酸を空白区切りでトークン化するのが基本
    return " ".join(list(seq))

def _resolve_amp_dtype(hp_amp_dtype: str) -> torch.dtype:
    s = str(hp_amp_dtype).lower()
    if s in ("fp16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported amp_dtype '{hp_amp_dtype}'. Use one of: fp16, bf16, fp32")

def load_model(encoder_model: str, tokenizer_model: str, amp_dtype: torch.dtype, device: str, lora_path: str = None, state_path: str = None):
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_model, do_lower_case=False)
    if device == "cuda" and amp_dtype in (torch.float16, torch.bfloat16):
        model = T5EncoderModel.from_pretrained(encoder_model, torch_dtype=amp_dtype)
    else:
        model = T5EncoderModel.from_pretrained(encoder_model)
    if state_path:
        sd = torch.load(state_path, map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[FullFT] Loaded ProtT5 state from {state_path}. Missing: {len(missing)} Unexpected: {len(unexpected)}")
    if lora_path:
        if not _HAS_PEFT:
            raise ImportError("peft is required to load LoRA adapters.")
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"[LoRA] Loaded adapters from {lora_path}")
    model.to(device); model.eval()
    emb_dim = int(model.config.d_model)
    return tokenizer, model, emb_dim

def _remove_special_tokens(hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer, seq_len: int) -> torch.Tensor:
    keep = attention_mask.bool()
    special = torch.tensor(
        [tid in tokenizer.all_special_ids for tid in input_ids.tolist()],
        device=input_ids.device,
        dtype=torch.bool,
    )
    keep = keep & (~special)
    out = hidden[keep]
    if out.shape[0] != seq_len:
        raise RuntimeError(f"Length mismatch after special-token removal: got={out.shape[0]} expected={seq_len}")
    return out

@torch.inference_mode()
def embed_single_chunk(seq: str, tokenizer, model, device: str, add_special_tokens: bool, amp_dtype: torch.dtype) -> np.ndarray:
    spaced = to_spaced(seq)
    toks = tokenizer(spaced, return_tensors="pt", add_special_tokens=add_special_tokens, padding=False, truncation=False)
    toks = {k: v.to(device) for k, v in toks.items()}
    use_autocast = (device == 'cuda' and amp_dtype in (torch.float16, torch.bfloat16))
    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_autocast):
        reps = model(**toks).last_hidden_state.squeeze(0)
    reps = _remove_special_tokens(hidden=reps, input_ids=toks["input_ids"].squeeze(0), attention_mask=toks["attention_mask"].squeeze(0), tokenizer=tokenizer, seq_len=len(seq))
    return reps.detach().cpu().float().numpy().astype(np.float32)

def run(input_file: str,
        out_dir: str,
        batch_size: int,
        device: str,
        amp_dtype: torch.dtype,
        skip_existing: bool,
        allow_diff: bool,
        add_special_tokens: bool,
        clean_mode: str,
        encoder_model: str,
        tokenizer_model: str,
        lora_path: str = None,
        state_path: str = None):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)

    tokenizer, model, emb_dim = load_model(encoder_model, tokenizer_model, amp_dtype, device, lora_path=lora_path, state_path=state_path)

    sequences: List[str] = []
    with open(input_file) as f:
        for line in f:
            sequences.append(clean_sequence(line, mode=clean_mode))

    normal_idx = [i for i, s in enumerate(sequences) if 0 < len(s) <= MAX_TOKENS]
    empty_idx = [i for i, s in enumerate(sequences) if len(s) == 0]
    long_idx = [i for i, s in enumerate(sequences) if len(s) > MAX_TOKENS]  # チャンク分割にまわす

    batches: List[List[int]] = []
    buf: List[int] = []
    for i in normal_idx:
        buf.append(i)
        if len(buf) >= batch_size:
            batches.append(buf); buf = []
    if buf: batches.append(buf)

    existing = set()
    if skip_existing:
        existing = {int(fn.split('.')[0]) for fn in os.listdir(out_dir) if fn.endswith(".npy") and fn.split('.')[0].isdigit()}

    diff_counter = Counter()
    use_autocast = (device == 'cuda' and amp_dtype in (torch.float16, torch.bfloat16))

    with torch.inference_mode():
        for batch in tqdm(batches, desc="Embedding (normal: ProtT5)"):
            need = [i for i in batch if i not in existing]
            if not need: continue
            seqs = [sequences[i] for i in batch]
            spaced = [" ".join(list(s)) for s in seqs]
            toks = tokenizer(spaced, return_tensors="pt", add_special_tokens=add_special_tokens, padding=True, truncation=False)
            toks = {k: v.to(device) for k, v in toks.items()}
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_autocast):
                reps = model(**toks).last_hidden_state  # [B, Tpad, D]
            attn = toks["attention_mask"]; lengths = attn.sum(dim=1).tolist()
            for bi, tok_len in enumerate(lengths):
                gidx = batch[bi]
                if gidx in existing: continue
                seq_len = len(seqs[bi])
                arr = reps[bi, :tok_len, :]
                arr = _remove_special_tokens(hidden=arr, input_ids=toks["input_ids"][bi, :tok_len], attention_mask=toks["attention_mask"][bi, :tok_len], tokenizer=tokenizer, seq_len=seq_len)
                diff_counter[arr.shape[0] - seq_len] += 1
                np.save(os.path.join(out_dir, f"{gidx}.npy"), arr.detach().cpu().numpy().astype(np.float32))

    # 長い配列は単純にチャンク分割（MAX_TOKENS ごと）
    for idx in tqdm(long_idx, desc="Embedding (long: ProtT5)"):
        if idx in existing: continue
        seq = sequences[idx]
        arrays = []
        for i in range(0, len(seq), MAX_TOKENS):
            part = seq[i:i+MAX_TOKENS]
            arr = embed_single_chunk(part, tokenizer, model, device, add_special_tokens, amp_dtype)
            arrays.append(arr)
        arr = np.concatenate(arrays, axis=0) if arrays else np.zeros((0, emb_dim), dtype=np.float32)
        np.save(os.path.join(out_dir, f"{idx}.npy"), arr)
        diff_counter[arr.shape[0] - len(seq)] += 1

    post = Counter()
    for i, seq in enumerate(sequences):
        path = os.path.join(out_dir, f"{i}.npy")
        if not os.path.isfile(path): continue
        emb = np.load(path, mmap_mode="r")
        post[emb.shape[0] - len(seq)] += 1

    print("[CHECK] diff_counter (during save):", dict(diff_counter))
    print("[CHECK] post verification diffs:", dict(post))
    if not allow_diff and any(k != 0 for k in post):
        raise RuntimeError(f"[ASSERT] Non-zero diffs remain: {post}")
    print(f"[DONE] Saved embeddings to {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--data_path", type=str, default=None)
    ap.add_argument("--embeddings_path", type=str, default=None)
    
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--amp_dtype", type=str, choices=["fp16", "bf16", "fp32"], default=None)
    ap.add_argument("--lora_path", type=str, default=None)
    ap.add_argument("--state_path", type=str, default=None)
    ap.add_argument("--hparams_path", type=str, default="./experiment_configs/binary.yml")
    ap.add_argument("--lora_enable", dest="lora_enable", action="store_true")
    ap.set_defaults(lora_enable=None)
    ap.add_argument("--full_finetune_enable", dest="full_finetune_enable", action="store_true")
    ap.set_defaults(full_finetune_enable=None)
    ap.add_argument("--compat_plm", dest="compat_plm", action="store_true")
    ap.set_defaults(compat_plm=None)
    
    args = ap.parse_args()

    # Load and apply hyperparameters from YAML if provided
    hp = {}
    if args.hparams_path is not None:
        try:
            with open(args.hparams_path, 'r') as f:
                hp = _flatten_hparams(
                    yaml.safe_load(f),
                    preserve_keys={"lr_schedule", "early_stopping"}
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load hparams file '{args.hparams_path}': {e}")
        
    # Extract PLM-related keys from YAML
    cfg = {
        "data_path": hp.get("data_path", args.data_path),
        "embeddings_path": hp.get("embeddings_path", args.embeddings_path),
        "encoder_model": hp.get("plm_encoder_model", "Rostlab/prot_t5_xl_uniref50"),
        "tokenizer_model": hp.get("plm_tokenizer_model", hp.get("plm_encoder_model", "Rostlab/prot_t5_xl_uniref50")),
        "add_special_tokens": hp.get("plm_add_special_tokens", False),
        "clean_mode": hp.get("plm_clean_mode", "prott5"),
        "allow_diff": hp.get("plm_allow_diff", False),
        "batch_size": hp.get("plm_batch_size", 8),
        "device": hp.get("plm_device", "auto"),
        "amp_dtype_str": hp.get("amp_dtype", "fp16"),
        "skip_existing": not hp.get("plm_no_skip_existing", False),
        "lora_path": hp.get("lora_inference_path", args.lora_path) or None,
        "state_path": hp.get("full_state_path", args.state_path) or None,
        "lora_enable": bool(hp.get("lora_enable", False)),
        "full_finetune_enable": bool(hp.get("full_finetune_enable", False)),
        "compat_plm": bool(hp.get("compat_plm", False)),
    }

    # CLI明示指定のみ上書き
    if args.data_path is not None:
        cfg["data_path"] = args.data_path
    if args.embeddings_path is not None:
        cfg["embeddings_path"] = args.embeddings_path
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.amp_dtype is not None:
        cfg["amp_dtype_str"] = args.amp_dtype
    if args.lora_path is not None:
        cfg["lora_path"] = args.lora_path
    if args.state_path is not None:
        cfg["state_path"] = args.state_path
    if args.lora_enable is not None:
        cfg["lora_enable"] = args.lora_enable
    if args.full_finetune_enable is not None:
        cfg["full_finetune_enable"] = args.full_finetune_enable
    if args.compat_plm is not None:
        cfg["compat_plm"] = args.compat_plm

    if cfg["compat_plm"]:
        cfg["add_special_tokens"] = True
        cfg["clean_mode"] = "prott5"

    if not cfg["lora_enable"]:
        cfg["lora_path"] = None
    if not cfg["full_finetune_enable"]:
        cfg["state_path"] = None

    if not cfg["data_path"]:
        raise ValueError("data_path is required (set in YAML or pass --data_path).")
    if not cfg["embeddings_path"]:
        raise ValueError("embeddings_path is required (set in YAML or pass --embeddings_path).")
    input_file = os.path.join(cfg["data_path"], f"{args.mode}.txt")
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    out_dir = os.path.join(cfg["embeddings_path"], args.mode)

    amp_dtype = _resolve_amp_dtype(cfg["amp_dtype_str"])

    # Print summary log
    print(f"[HP] Using hparams file: {args.hparams_path}")
    print(f"[HP]   data_path               = {cfg['data_path']}")
    print(f"[HP]   plm_encoder_model       = {cfg['encoder_model']}")
    print(f"[HP]   plm_tokenizer_model     = {cfg['tokenizer_model']}")
    print(f"[HP]   plm_add_special_tokens  = {cfg['add_special_tokens']}")
    print(f"[HP]   plm_clean_mode          = {cfg['clean_mode']}")
    print(f"[HP]   plm_allow_diff          = {cfg['allow_diff']}")
    print(f"[HP]   plm_batch_size          = {cfg['batch_size']}")
    print(f"[HP]   plm_device              = {cfg['device']}")
    print(f"[HP]   amp_dtype               = {cfg['amp_dtype_str']}")
    print(f"[HP]   skip_existing           = {cfg['skip_existing']}")
    print(f"[HP]   lora_enable             = {cfg['lora_enable']}")
    print(f"[HP]   full_finetune_enable    = {cfg['full_finetune_enable']}")
    print(f"[HP]   lora_inference_path     = {cfg['lora_path']}")
    print(f"[HP]   full_state_path         = {cfg['state_path']}")

    run(
        input_file=input_file,
        out_dir=out_dir,
        batch_size=cfg["batch_size"],
        device=cfg["device"],
        amp_dtype=amp_dtype,
        skip_existing=cfg["skip_existing"],
        allow_diff=cfg["allow_diff"],
        add_special_tokens=cfg["add_special_tokens"],
        clean_mode=cfg["clean_mode"],
        encoder_model=cfg["encoder_model"],
        tokenizer_model=cfg["tokenizer_model"],
        lora_path=cfg["lora_path"],
        state_path=cfg["state_path"],
    )

if __name__ == "__main__":
    main()
