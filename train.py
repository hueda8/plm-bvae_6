# PyTorch training script with AMP, pLM LoRA/full-FT options, freeze schedule, and param-group LR
# Refactored for readability and clearer separation of concerns.
# Added: set_randomseeds() to fix RNG seeds across libraries.

import os
import json
import glob
import math
from typing import Dict, Tuple, Optional, List
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import random  # ADD: for Python RNG

from featurizer import batch_generator, batch_generator_text, prepare_trimmed_cache
from model import SeqModel
from load_hparams import loader_func, PrintHparamsInfo
import argparse

# ============================================================
# Seeding utilities
# ============================================================

def set_randomseeds(s: int, deterministic: bool = False):
    """
    Fix RNG seeds across numpy, random, torch (CPU/GPU), and optionally transformers/accelerate.
    If deterministic=True, enable deterministic algorithms (may reduce performance on GPU).
    """
    # Python / env
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)

    # Numpy
    np.random.seed(s)

    # PyTorch
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


# ============================================================
# Environment / Reproducibility Setup
# ============================================================

def setup_environment(hp: dict) -> torch.device:
    """Set device, seed, and TF32 flags."""
    PrintHparamsInfo(hp)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count()
    print(f"[DEVICE] selected_device={device}")
    print(f"[DEVICE] torch.cuda.is_available()={cuda_available}, torch.cuda.device_count()={cuda_count}")

    if cuda_available and device.type == "cuda":
        try:
            cur_idx = torch.cuda.current_device()
            print(f"[DEVICE] current_device_index={cur_idx}")
            print(f"[DEVICE] current_device_name={torch.cuda.get_device_name(cur_idx)}")
        except Exception as e:
            print(f"[DEVICE][WARN] Failed to query CUDA device details: {e}")
    else:
        print("[DEVICE][WARN] CUDA is not available. Training will run on CPU.")

    # Use hparams['seed'] if available; default to 42
    seed_value = int(hp.get('seed', 42))
    # Deterministic can be toggled via hparams; default False
    deterministic = bool(hp.get('deterministic', False))
    set_randomseeds(seed_value, deterministic=deterministic)

    if device.type == 'cuda':
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(hp.get('allow_tf32', True))
        except Exception:
            pass
    return device


# ============================================================
# Embeddings management (precomputed mode)
# ============================================================

def _list_npy(dir_path: str):
    if not os.path.isdir(dir_path):
        return []
    return sorted(glob.glob(os.path.join(dir_path, "*.npy")))

def _count_lines(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)

def ensure_embeddings(hp: dict, modes: Tuple[str, ...] = ("train", "dev")):
    """
    Check-only: Verify that precomputed embeddings exist for each split in `modes`.
    - Counts lines in {data_path}/{mode}.txt
    - Counts .npy files in {embeddings_path}/{mode}
    If embeddings are missing, raise FileNotFoundError (do NOT generate).
    """
    if "data_path" not in hp:
        raise KeyError("hparams must include 'data_path'.")
    if "embeddings_path" not in hp:
        raise KeyError("hparams must include 'embeddings_path'.")

    for mode in modes:
        txt_path = os.path.join(hp["data_path"], f"{mode}.txt")
        out_dir = os.path.join(hp["embeddings_path"], mode)

        need = _count_lines(txt_path)
        if need == 0:
            print(f"[EMB] Skip {mode}: no lines in {txt_path}")
            continue

        have = len(_list_npy(out_dir))
        if have >= need:
            print(f"[EMB] OK {mode}: {have}/{need} embeddings exist at {out_dir}")
            continue

        raise FileNotFoundError(
            f"[EMB] Missing embeddings for mode='{mode}': found {have}/{need} .npy files in {out_dir}. "
            f"Please generate embeddings in advance (e.g., run embedding script) before training."
        )

def _list_idx_files(dir_path: str, pattern: str = "*.idx.npy") -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    return sorted(glob.glob(os.path.join(dir_path, pattern)))

def check_sidecar_indices_consistency(hp: dict, modes: Tuple[str, ...] = ("train", "dev")):
    """
    When allow_sidecar_indices is true and static_idx is not set,
    check that the number of sidecar index files equals the number of sequences
    (lines) in data_path/{mode}.txt for the given modes.
    If strict_sidecar_indices is true, raise an error on mismatch; otherwise warn.
    """
    allow = bool(hp.get("allow_sidecar_indices", False))
    static_idx = hp.get("select_indices", None)
    strict = bool(hp.get("strict_sidecar_indices", True))

    if not allow:
        return
    if static_idx is not None and len(static_idx) > 0:
        # static indices override sidecar; skip checking
        return

    for mode in modes:
        txt_path = os.path.join(hp["data_path"], f"{mode}.txt")
        idx_dir = os.path.join(hp.get("embeddings_path", ""), mode, "indices")

        if not os.path.isfile(txt_path):
            msg = f"[IDX CHECK] data file missing for mode='{mode}': {txt_path}"
            if strict:
                raise FileNotFoundError(msg)
            else:
                print(msg)
                continue

        need = _count_lines(txt_path)
        have = len(_list_idx_files(idx_dir))

        if need == 0:
            print(f"[IDX CHECK] mode='{mode}': no lines in {txt_path} (skip)")
            continue

        if have != need:
            msg = (f"[IDX CHECK] mode='{mode}': indices count mismatch "
                   f"(found {have} files in {idx_dir}, expected {need} from {txt_path}).")
            if strict:
                raise RuntimeError(msg + " Set strict_sidecar_indices=false to downgrade to warning.")
            else:
                print("[WARN]", msg)
        else:
            print(f"[IDX CHECK] mode='{mode}': OK ({have}/{need}) sidecar indices at {idx_dir}")

# ============================================================
# Optimizer Utilities
# ============================================================

def _no_decay_param(name: str) -> bool:
    lname = name.lower()
    return (
        lname.endswith('bias')
        or 'norm' in lname
        or 'embedding' in lname
    )

def _is_lora_param(name: str) -> bool:
    return ('lora_' in name) or ('.lora_' in name)

def _toggle_plm_requires_grad(model: torch.nn.Module, require: bool):
    """Toggle requires_grad for parameters under plm_extractor.plm.*"""
    for n, p in model.named_parameters():
        if n.startswith('plm_extractor.plm'):
            p.requires_grad = require

def _collect_param_groups(model: torch.nn.Module,
                          use_precomputed: bool,
                          initial_lr: float,
                          plm_lr: float,
                          wd_main: float,
                          wd_plm: float,
                          wd_lora: float,
                          include_plm: bool) -> List[dict]:
    """Create param groups for AdamW with tags, optionally including PLM/LoRA groups."""
    named_params = list(model.named_parameters())

    plm_trainable, plm_trainable_no_decay = [], []
    lora_params, lora_params_no_decay = [], []
    main_params, main_params_no_decay = [], []

    for n, p in named_params:
        if not p.requires_grad:
            # Skip frozen params
            continue
        if n.startswith('plm_extractor.plm'):
            if not include_plm:
                # Exclude PLM groups until unfreeze time
                continue
            if _is_lora_param(n):
                (lora_params_no_decay if _no_decay_param(n) else lora_params).append(p)
            else:
                (plm_trainable_no_decay if _no_decay_param(n) else plm_trainable).append(p)
        else:
            (main_params_no_decay if _no_decay_param(n) else main_params).append(p)

    param_groups: List[dict] = []
    if main_params:
        param_groups.append({'params': main_params, 'lr': initial_lr, 'weight_decay': wd_main, 'tag': 'main'})
    if main_params_no_decay:
        param_groups.append({'params': main_params_no_decay, 'lr': initial_lr, 'weight_decay': 0.0, 'tag': 'main_nd'})
    if include_plm:
        if lora_params:
            param_groups.append({'params': lora_params, 'lr': plm_lr, 'weight_decay': wd_lora, 'tag': 'lora'})
        if lora_params_no_decay:
            param_groups.append({'params': lora_params_no_decay, 'lr': plm_lr, 'weight_decay': 0.0, 'tag': 'lora_nd'})
        if plm_trainable:
            param_groups.append({'params': plm_trainable, 'lr': plm_lr, 'weight_decay': wd_plm, 'tag': 'plm'})
        if plm_trainable_no_decay:
            param_groups.append({'params': plm_trainable_no_decay, 'lr': plm_lr, 'weight_decay': 0.0, 'tag': 'plm_nd'})

    return param_groups

def build_optimizer(model: torch.nn.Module,
                    hp: dict,
                    use_precomputed: bool,
                    initial_lr: float,
                    plm_lr: float,
                    wd_main: float,
                    wd_plm: float,
                    wd_lora: float,
                    include_plm: bool) -> torch.optim.Optimizer:
    """Construct AdamW optimizer with parameter grouping; do NOT rebuild later."""
    param_groups = _collect_param_groups(
        model=model,
        use_precomputed=use_precomputed,
        initial_lr=initial_lr,
        plm_lr=plm_lr,
        wd_main=wd_main,
        wd_plm=wd_plm,
        wd_lora=wd_lora,
        include_plm=include_plm
    )
    opt = torch.optim.AdamW(param_groups)
    return opt


# ============================================================
# Runtime Parameter Extraction
# ============================================================

def extract_runtime_params(hp: dict) -> Dict[str, any]:
    """Collect runtime-related parameters with expanded amp_dtype parsing."""
    amp_mode_str = str(hp.get('amp_dtype', 'fp16')).lower()
    if amp_mode_str in ('fp16', 'half'):
        amp_dtype = torch.float16
    elif amp_mode_str in ('bf16', 'bfloat16'):
        amp_dtype = torch.bfloat16
    elif amp_mode_str in ('fp32', 'float32'):
        amp_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported amp_dtype '{amp_mode_str}'. Use one of: fp16, bf16, fp32")

    return {
        'use_precomputed': bool(hp.get('input_is_precomputed', True)),
        'freeze_plm_epochs': int(hp.get('freeze_plm_epochs', 0)),
        'initial_lr': float(hp.get('learning_rate', 5e-4)),
        'plm_lr': float(hp.get('plm_learning_rate', hp.get('learning_rate', 5e-5))),
        'wd_main': float(hp.get('weight_decay', 0.01)),
        'wd_plm': float(hp.get('full_weight_decay', 0.01)),
        'wd_lora': float(hp.get('lora_weight_decay', 0.0)),
        'use_amp': bool(hp.get('use_amp', True)),
        'amp_dtype': amp_dtype,
        'amp_mode_str': amp_mode_str,
        'grad_accum_steps': int(hp.get('grad_accum_steps', 1)),
        'clip_grad_norm': float(hp.get('clip_grad_norm', 2.0)),
        'num_epochs': int(hp.get('num_epochs', 10)),
        'early_conf': hp.get('early_stopping', None),
        'lr_schedule_conf': hp.get('lr_schedule', None),
        'resume_checkpoint': hp.get('resume_checkpoint', None),
    }


# ============================================================
# Scheduler builder (shared)
# ============================================================

def build_scheduler(optimizer: torch.optim.Optimizer,
                    rt: Dict[str, any],
                    last_epoch: Optional[int] = None) -> Tuple[Optional[object], Optional[str]]:
    """Create LR scheduler for current optimizer param_groups. Supports 'plateau' and 'cosine'."""
    lr_conf = rt['lr_schedule_conf']
    scheduler = None
    scheduler_mode: Optional[str] = None
    if isinstance(lr_conf, dict):
        mode = lr_conf.get('mode', 'cosine').lower()
        if mode == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=int(lr_conf.get('patience', 3)),
                factor=float(lr_conf.get('factor', 0.5)),
                min_lr=float(lr_conf.get('min_lr', 1e-6))
            )
            scheduler_mode = 'plateau'
        
        elif mode == 'cosine':
            warmup_epochs = int(lr_conf.get('warmup_epochs', 0))
            total_epochs = rt['num_epochs']
            min_lr_ratio_conf = lr_conf.get('min_lr_ratio', None)

            # Persist initial_lr per group
            for g in optimizer.param_groups:
                if 'initial_lr' not in g:
                    g['initial_lr'] = g['lr']

            def make_group_lambda(init_lr: float):
                if min_lr_ratio_conf is not None:
                    # Ratio mode: same ratio for all groups
                    min_lr_ratio = float(min_lr_ratio_conf)
                # Clamp for safety
                min_lr_ratio = max(min(min_lr_ratio, 1.0), 0.0)

                def f(ep: int):
                    # Warmup
                    if ep < warmup_epochs:
                        return (ep + 1) / max(warmup_epochs, 1)
                    # Cosine progress
                    progress = (ep - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
                    progress = min(max(progress, 0.0), 1.0)
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
                return f

            lambdas = [make_group_lambda(g['initial_lr']) for g in optimizer.param_groups]
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
            if last_epoch is not None:
                # Ensure we continue schedule from the correct epoch index
                scheduler.last_epoch = int(last_epoch)
            scheduler_mode = 'cosine'
        else:
            print(f"[WARN] Unknown lr_schedule mode '{mode}' - no scheduler will be used.")
    return scheduler, scheduler_mode


# ============================================================
# Checkpoint utilities
# ============================================================

def _extract_non_plm_state_dict(model: torch.nn.Module) -> dict:
    """
    Save only non-PLM params (AE side):
      - encoder_stack
      - latent_layer
      - decoder_embeddings
      - decoder
      - etc.
    Excludes: plm_extractor.plm.*
    """
    out = {}
    for k, v in model.state_dict().items():
        if not k.startswith("plm_extractor.plm."):
            out[k] = v.detach().cpu()
    return out

def save_checkpoint_bundle(model: torch.nn.Module, hp: dict, kind: str = "last"):
    """
    Save a bundle of:
      - LoRA mode:
          * lora_plm/   (adapter only)
          * ae_head.pt  (non-PLM params + hparams)
      - non-LoRA mode:
          * model.pt (full SeqModel state_dict + hparams)
      - full-FT sidecar:
          * plm_full.pt (optional, existing behavior)
    kind:
      - "best": saved when dev improves
      - "last": saved at the end of training
    """
    if kind not in ("best", "last"):
        raise ValueError(f"Unsupported checkpoint kind '{kind}'. Use one of: best, last")

    base_dir = os.path.join(hp['model_path'], "best" if kind == "best" else "last")
    os.makedirs(base_dir, exist_ok=True)

    is_lora = bool(getattr(getattr(model, "plm_extractor", None), "lora_enable", False))
    is_fullft = bool(getattr(getattr(model, "plm_extractor", None), "full_finetune_enable", False))

    if is_lora:
        # 1) LoRA adapter only
        if hasattr(model.plm_extractor.plm, "save_pretrained"):
            lora_dir = os.path.join(base_dir, "lora_plm")
            os.makedirs(lora_dir, exist_ok=True)
            model.plm_extractor.plm.save_pretrained(lora_dir)
            print(f"[CKPT:{kind}] Saved LoRA adapters -> {lora_dir}")
        else:
            raise RuntimeError("LoRA is enabled but plm has no save_pretrained().")

        # 2) AE side only (exclude plm_extractor.plm.*)
        ae_path = os.path.join(base_dir, "ae_head.pt")
        ae_state = _extract_non_plm_state_dict(model)
        torch.save({"ae": ae_state, "hparams": hp}, ae_path)
        print(f"[CKPT:{kind}] Saved AE head -> {ae_path}")

    else:
        # model.pt (full SeqModel state_dict + hparams)
        model_path = os.path.join(base_dir, 'model.pt')
        torch.save({'model': model.state_dict(), 'hparams': hp}, model_path)
        print(f"[CKPT:{kind}] Saved SeqModel -> {model_path}")

    # full-FT sidecar (existing)
    try:
        if hasattr(model, 'plm_extractor') and model.plm_extractor:
            if is_fullft:
                plm_path = os.path.join(base_dir, 'plm_full.pt')
                torch.save(model.plm_extractor.plm.state_dict(), plm_path)
                print(f"[CKPT:{kind}] Saved full pLM weights -> {plm_path}")
    except Exception as e:
        print(f"[WARN] Failed to save pLM adapters/weights: {e}")


# ============================================================
# Model / Optimizer Creation
# ============================================================

def create_model_and_optimizer(device: torch.device, hp: dict, rt: Dict[str, any]) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Optional[object], Optional[str]]:
    """Instantiate model, optionally resume, and build initial optimizer & scheduler."""
    model = SeqModel(hp).to(device)

    if rt['resume_checkpoint'] and os.path.isfile(rt['resume_checkpoint']):
        ckpt = torch.load(rt['resume_checkpoint'], map_location=device)
        missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
        print(f"[RESUME] Loaded checkpoint: {rt['resume_checkpoint']}")
        if missing:
            print(f"[RESUME] Missing keys: {missing}")
        if unexpected:
            print(f"[RESUME] Unexpected keys: {unexpected}")

    # Freeze PLM params initially if we plan to unfreeze later
    include_plm_initial = (not rt['use_precomputed']) and (rt['freeze_plm_epochs'] == 0)
    if (not rt['use_precomputed']) and (rt['freeze_plm_epochs'] > 0):
        _toggle_plm_requires_grad(model, require=False)

    optimizer = build_optimizer(
        model=model,
        hp=hp,
        use_precomputed=rt['use_precomputed'],
        initial_lr=rt['initial_lr'],
        plm_lr=rt['plm_lr'],
        wd_main=rt['wd_main'],
        wd_plm=rt['wd_plm'],
        wd_lora=rt['wd_lora'],
        include_plm=include_plm_initial
    )

    scheduler, scheduler_mode = build_scheduler(optimizer, rt, last_epoch=None)
    return model, optimizer, scheduler, scheduler_mode


# ============================================================
# One Epoch Execution
# ============================================================

def _add_plm_groups_if_needed(epoch: int,
                              model: torch.nn.Module,
                              optimizer: torch.optim.Optimizer,
                              hp: dict,
                              rt: Dict[str, any],
                              scheduler,
                              scheduler_mode: Optional[str]) -> Tuple[torch.optim.Optimizer, Optional[object], Optional[str], bool]:
    """
    At the unfreeze epoch, toggle PLM requires_grad to True and add PLM/LoRA param groups
    to the existing optimizer via add_param_group. Rebuild scheduler to align with new groups.
    Returns: (optimizer, scheduler, scheduler_mode, changed_flag)
    """
    changed = False
    # Only if using PLM online and at the configured unfreeze epoch
    if (not rt['use_precomputed']) and hasattr(model, 'plm_extractor') and model.plm_extractor is not None:
        if epoch == rt['freeze_plm_epochs']:
            # If PLM/LoRA groups already exist, skip
            has_plm_groups = any(g.get('tag', '').startswith('plm') or g.get('tag', '').startswith('lora')
                                 for g in optimizer.param_groups)
            if not has_plm_groups:
                # Unfreeze
                _toggle_plm_requires_grad(model, require=True)
                # Build PLM/LoRA param groups and add to optimizer
                plm_groups = _collect_param_groups(
                    model=model,
                    use_precomputed=rt['use_precomputed'],
                    initial_lr=rt['initial_lr'],
                    plm_lr=rt['plm_lr'],
                    wd_main=rt['wd_main'],
                    wd_plm=rt['wd_plm'],
                    wd_lora=rt['wd_lora'],
                    include_plm=True
                )
                # Only add groups that target PLM/LoRA; main groups already exist
                for g in plm_groups:
                    if g.get('tag', '').startswith('plm') or g.get('tag', '').startswith('lora'):
                        # Ensure 'initial_lr' for cosine scheduler
                        if 'initial_lr' not in g:
                            g['initial_lr'] = g['lr']
                        optimizer.add_param_group(g)
                # Rebuild scheduler to include new groups; preserve progress by setting last_epoch=epoch-1
                scheduler, scheduler_mode = build_scheduler(optimizer, rt, last_epoch=epoch - 1)
                changed = True
    return optimizer, scheduler, scheduler_mode, changed

def run_epoch(split: str,
              epoch: int,
              model: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              hp: dict,
              rt: Dict[str, any],
              device: torch.device,
              scaler: Optional[torch.cuda.amp.GradScaler],
              scheduler,
              scheduler_mode: Optional[str]) -> Tuple[Dict[str, float], torch.optim.Optimizer, Optional[object], Optional[str]]:
    """
    Run one epoch over the given split.
    Returns:
      - metrics: Dict[str, float]
      - optimizer: possibly updated optimizer (param groups added at unfreeze)
      - scheduler: possibly rebuilt scheduler to align with updated optimizer
      - scheduler_mode: same or updated mode
    """
    is_train = split == 'train'
    model.train(is_train)
    metrics: Dict[str, List[float]] = {}
    gen = batch_generator if rt['use_precomputed'] else batch_generator_text

    # Add PLM/LoRA param groups at the unfreeze epoch without rebuilding the optimizer
    if is_train:
        optimizer, scheduler, scheduler_mode, _ = _add_plm_groups_if_needed(
            epoch, model, optimizer, hp, rt, scheduler, scheduler_mode
        )

    accum = 0
    for batch in tqdm(gen(hp, split), desc=f"{split} epoch {epoch}", leave=False):
        if rt['use_precomputed']:
            batch_x, batch_y = batch
            x = torch.from_numpy(batch_x).float().to(device)
            select_indices_batch = None
        else:
            # テキストモードでは select_indices_batch が第三戻り値として返る
            if len(batch) == 3:
                seq_list, batch_y, select_indices_batch = batch
            else:
                raise ValueError(
                    "batch_generator_text must yield a 3-tuple: "
                    "(seq_list, batch_y, select_indices_batch). "
                    f"Got len(batch)={len(batch)}."
                )
            x = seq_list
        y = torch.from_numpy(batch_y).long().to(device)

        if is_train and accum == 0:
            optimizer.zero_grad(set_to_none=True)

        autocast_enabled = (device.type == 'cuda'
                            and rt['use_amp']
                            and rt['amp_dtype'] != torch.float32)

        with torch.autocast(device_type=device.type if device.type == 'cuda' else 'cpu',
                            dtype=rt['amp_dtype'],
                            enabled=autocast_enabled):
            _, losses = model(x, 'train' if is_train else 'eval', y,
                              select_indices_batch=select_indices_batch)
            loss = losses['nll']
            if hp.get('use_kl', False) and 'kl' in losses:
                kl_coef = float(hp.get('kl_coef', 0.01))
                loss = loss + losses['kl'] * kl_coef

        if is_train:
            if scaler is not None and autocast_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum += 1
            if accum >= rt['grad_accum_steps']:
                # クリップ対象パラメータ収集（requires_grad かつ grad が付与されているもの）
                params_to_clip = [
                    p
                    for group in optimizer.param_groups
                    for p in group['params']
                    if p.requires_grad and p.grad is not None
                ]
                if scaler is not None and autocast_enabled:
                    # AMP: 実際の勾配値に戻す
                    scaler.unscale_(optimizer)
                    if params_to_clip:  # 空リスト回避
                        clip_grad_norm_(params_to_clip, rt['clip_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if params_to_clip:
                        clip_grad_norm_(params_to_clip, rt['clip_grad_norm'])
                    optimizer.step()
                accum = 0

        for k, v in losses.items():
            v_float = float(v.detach().cpu().item()) if isinstance(v, torch.Tensor) else float(v)
            metrics.setdefault(k, []).append(v_float)

    metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
    
    # 全グループの学習率ログ（tag付き）
    for g in optimizer.param_groups:
        tag = g.get('tag', None)
        if tag is not None:
            metrics[f'lr_{tag}'] = float(g['lr'])

    # 監視用: 正則化相当値（regval_{tag}）を追加
    with torch.no_grad():
        for g in optimizer.param_groups:
            tag = g.get('tag', None)
            if tag is None:
                continue
            wd = float(g.get('weight_decay', 0.0))
            if wd == 0.0:
                metrics[f'regval_{tag}'] = 0.0
                continue
            l2_sum = 0.0
            for p in g['params']:
                if p is None or not hasattr(p, "data"):
                    continue
                t = p.detach()
                l2_sum += float(t.pow(2).sum().cpu().item())
            metrics[f'regval_{tag}'] = 0.5 * wd * l2_sum
    
    metrics['amp_dtype'] = rt['amp_mode_str']
    return metrics, optimizer, scheduler, scheduler_mode


# ============================================================
# Scheduler / Checkpoint Handling
# ============================================================

def handle_scheduler_and_checkpoint(epoch: int,
                                    dev_stats: Dict[str, float],
                                    best_dev: float,
                                    best_epoch: int,
                                    optimizer: torch.optim.Optimizer,
                                    scheduler,
                                    scheduler_mode: Optional[str],
                                    hp: dict,
                                    early_conf,
                                    epochs_no_improve: int,
                                    use_early: bool,
                                    es_min_delta: float,
                                    model: torch.nn.Module,
                                    device: torch.device) -> Tuple[float, int, int, bool]:
    """Update LR scheduler, save checkpoint if improved, manage early stopping."""
    dev_score = dev_stats.get('nll', dev_stats.get('ppl', float('inf')))

    if scheduler is not None:
        if scheduler_mode == 'plateau':
            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(dev_score)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < prev_lr - 1e-12:
                print(f"[LR REDUCE] lr -> {new_lr:.6g}")
        elif scheduler_mode == 'cosine':
            scheduler.step()
            print(f"[LR COSINE] epoch {epoch} lrs: " +
                  ", ".join(f"{g.get('tag','?')}={g['lr']:.6g}" for g in optimizer.param_groups if 'lr' in g))

    improved = dev_score < (best_dev - (es_min_delta if use_early else 0.0))
    if improved:
        best_dev = dev_score
        best_epoch = epoch
        epochs_no_improve = 0
        # Save BEST bundle (separate folder)
        save_checkpoint_bundle(model, hp, kind="best")
        print("Saved BEST checkpoint bundle. dev:", best_dev)
    else:
        if use_early:
            epochs_no_improve += 1
            if epochs_no_improve >= early_conf.get('patience', 10):
                print(f"Early stopping triggered at epoch {epoch}.")
                return best_dev, best_epoch, epochs_no_improve, True

    return best_dev, best_epoch, epochs_no_improve, False


# ============================================================
# Main Training Loop
# ============================================================

def main():
    device = setup_environment(hparams)
    rt = extract_runtime_params(hparams)

    # on-the-fly (LoRA/full-FT) 用 selection cache ログ
    if not rt['use_precomputed']:
        static_idx = hparams.get("select_indices", None)
        use_static = (static_idx is not None and len(static_idx) > 0)
        use_sidecar = bool(hparams.get("allow_sidecar_indices", False))
        use_text_sel_cache = bool(hparams.get("use_text_selection_cache", True))
        cache_root = hparams.get(
            "text_selection_cache_root",
            os.path.join(hparams.get("embeddings_path", None), "text_sel_cache")
        )
        force_rebuild = bool(hparams.get("text_selection_cache_force_rebuild", False))

        if use_static or use_sidecar:
            print("[TEXT SEL CACHE][AUTO] enabled because select_indices/sidecar is used.")
            print(f"[TEXT SEL CACHE][AUTO] use_text_selection_cache -> {use_text_sel_cache}")
            print(f"[TEXT SEL CACHE][AUTO] root -> {cache_root}")
            print(f"[TEXT SEL CACHE][AUTO] force_rebuild -> {force_rebuild}")
        else:
            print("[TEXT SEL CACHE][AUTO] disabled (no select_indices and allow_sidecar_indices=false).")

    # New: preflight check for sidecar indices consistency
    check_sidecar_indices_consistency(hparams, modes=("train", "dev", "test"))

    if rt['use_precomputed']:
        static_idx = hparams.get("select_indices", None)
        use_static = (static_idx is not None and len(static_idx) > 0)
        use_sidecar = bool(hparams.get("allow_sidecar_indices", False))
        need_trim_cache = use_static or use_sidecar

        if need_trim_cache:
            # force_rebuild は原則 false 固定（「1回作って再利用」）
            train_cache_dir = prepare_trimmed_cache(hparams, mode="train", force_rebuild=False)
            dev_cache_dir = prepare_trimmed_cache(hparams, mode="dev", force_rebuild=False)
            orig_embeddings_path = hparams["embeddings_path"]

            # featurizer.batch_generator が embeddings_path/{mode} を読む仕様なので、
            # ルートを trim_cache/<cache_key> に差し替える (train/dev)
            hparams["embeddings_path"] = os.path.dirname(train_cache_dir)
            hparams["selection_already_applied"] = True
            hparams["source_embeddings_path_for_indices"] = orig_embeddings_path
            print(f"[TRIM CACHE][AUTO] enabled because select_indices/sidecar is used.")
            print(f"[TRIM CACHE][AUTO] embeddings_path -> {hparams['embeddings_path']}")
        else:
            print("[TRIM CACHE][AUTO] disabled (no select_indices and allow_sidecar_indices=false).")

        ensure_embeddings(hparams, modes=("train", "dev"))

        if "embedding_dim" not in hparams or hparams["embedding_dim"] is None:
            raise KeyError(
                "hparams['embedding_dim'] is required when input_is_precomputed=true. "
                "Please set 'embedding_dim' explicitly in your hparams file."
            )

    os.makedirs(hparams['model_path'], exist_ok=True)

    model, optimizer, scheduler, scheduler_mode = create_model_and_optimizer(device, hparams, rt)

    scaler = torch.amp.GradScaler('cuda', enabled=(rt['use_amp']
                                                   and device.type == 'cuda'
                                                   and rt['amp_dtype'] == torch.float16))

    early_conf = rt['early_conf']
    use_early = isinstance(early_conf, dict)
    es_min_delta = float(early_conf.get('min_delta', 1e-3)) if use_early else 0.0

    train_metrics_file = open(os.path.join(hparams['model_path'], 'train_metrics.txt'), 'w')
    dev_metrics_file = open(os.path.join(hparams['model_path'], 'dev_metrics.txt'), 'w')

    best_dev = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    o_train = open('./train_loss.txt', 'w')
    o_dev = open('./dev_loss.txt', 'w')

    effective_use_amp = bool(rt['use_amp'] and device.type == 'cuda' and rt['amp_dtype'] != torch.float32)
    print(f"\n[INFO] Mixed precision mode: {rt['amp_mode_str']} (use_amp={effective_use_amp})")
    if scheduler_mode:
        print(f"[INFO] LR scheduler mode: {scheduler_mode}")

    for epoch in range(rt['num_epochs']):
        print('\nEpoch', epoch)

        # Train epoch
        train_stats, optimizer, scheduler, scheduler_mode = run_epoch('train', epoch, model, optimizer, hparams, rt, device, scaler, scheduler, scheduler_mode)
        train_metrics_file.write(json.dumps(train_stats) + '\n'); train_metrics_file.flush()
        o_train.write(str(train_stats) + '\n'); o_train.flush()

        # Dev epoch
        dev_stats, optimizer, scheduler, scheduler_mode = run_epoch('dev', epoch, model, optimizer, hparams, rt, device, scaler, scheduler, scheduler_mode)
        dev_metrics_file.write(json.dumps(dev_stats) + '\n'); dev_metrics_file.flush()
        o_dev.write(str(dev_stats) + '\n'); o_dev.flush()

        print('\ntrain stats:', train_stats, flush=True)
        print('dev stats:  ', dev_stats, flush=True)

        best_dev, best_epoch, epochs_no_improve, early_stop = handle_scheduler_and_checkpoint(
            epoch=epoch,
            dev_stats=dev_stats,
            best_dev=best_dev,
            best_epoch=best_epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_mode=scheduler_mode,
            hp=hparams,
            early_conf=early_conf if use_early else {},
            epochs_no_improve=epochs_no_improve,
            use_early=use_early,
            es_min_delta=es_min_delta,
            model=model,
            device=device
        )
        if early_stop:
            break

    o_train.close(); o_dev.close()
    train_metrics_file.close(); dev_metrics_file.close()

    # Save LAST bundle at the end of training (separate from BEST)
    save_checkpoint_bundle(model, hparams, kind="last")
    if not use_early:
        print(f"Saved LAST checkpoint bundle. Best dev nll={best_dev:.6f} at epoch {best_epoch}")
    else:
        print(f"Saved EARLY STOP checkpoint bundle. Best dev nll={best_dev:.6f} at epoch {best_epoch}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams_path", type=str, required=True)
    args = parser.parse_args()
    hparams = loader_func(args.hparams_path)

    main()



