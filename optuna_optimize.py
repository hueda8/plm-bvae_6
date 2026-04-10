import os
import yaml
import json
import argparse
import math
from typing import Dict, Any, Optional, Tuple, List

import optuna
from optuna.samplers import TPESampler

import torch

# Reuse training utilities from train.py without invoking its CLI main
from train import (
    setup_environment,
    extract_runtime_params,
    create_model_and_optimizer,
    run_epoch,
    ensure_embeddings,
)
from featurizer import prepare_trimmed_cache
from load_hparams import loader_func, PrintHparamsInfo


def load_hparams(base_path: Optional[str]) -> Dict[str, Any]:
    """Load base hparams YAML, or fallback to default loaded in code."""
    if base_path is None:
        raise RuntimeError("hparams_path must be provided for optuna_optimize.")
    hp = loader_func(base_path)
    # Apply model_name into path placeholders
    model_name = str(hp.get("model_name", "optuna_run"))
    hp["model_path"] = hp.get("model_path", "./model_data/<model_name>/").replace("<model_name>", model_name)
    hp["output_path"] = hp.get("output_path", "./model_output/<model_name>/").replace("<model_name>", model_name)
    return hp

def _snap_hidden_size_for_heads(base_hs: int, heads: int, lo: int = 128, hi: int = 1024, step: int = 128) -> int:
    # 刻み幅 step と heads の最小公倍数 L の倍数に丸める
    L = math.lcm(step, int(heads))
    # lo 以上 hi 以下の範囲に収めつつ、base_hs に最も近い倍数へスナップ（丸め方は運用に応じて変更可）
    min_m = max(1, math.ceil(lo / L))
    max_m = max(1, hi // L)
    m = int(round(base_hs / L))
    m = min(max(m, min_m), max_m)
    return int(L * m)

def suggest_hparams(
    trial: optuna.trial.Trial,
    base_hp: Dict[str, Any],
    study_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Copy base hparams and apply Optuna suggestions.
    Supports both Trial (during optimization) and FrozenTrial (after optimization).
    If trial is FrozenTrial (no trial.study), study_name must be passed explicitly (will fallback otherwise).
    """
    hp = dict(base_hp)

    # =========================
    # Hyperparameter search space
    # =========================
    hp["learning_rate"] = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
    hp["dropout_rate"] = trial.suggest_float("dropout_rate", 0.0, 0.3, step=0.1)
    hp["label_smoothing"] = trial.suggest_float("label_smoothing", 0.0, 0.1, step=0.05)
    hp["grad_accum_steps"] = trial.suggest_int("grad_accum_steps", 1, 4)
    hp["clip_grad_norm"] = trial.suggest_float("clip_grad_norm", 0.5, 5.0, step=0.5)
    hp["input_noise_std"] = trial.suggest_float("input_noise_std", 0.0, 0.05, step=0.01)
    hp["input_dropout_rate"] = trial.suggest_float("input_dropout_rate", 0.0, 0.2, step=0.1)
    # pLMへの入力は全長が基本
    hp["word_dropout"] = base_hp.get("word_dropout", 0.0)


    # Encoder configuration
    enc_type_choices = ["rnn_bidi"] #, "transformer", "rnn", "rnn_bidi_onlyfirst"]
    hp["encoder_type"] = base_hp.get("encoder_type", "rnn_bidi")
    # hp["encoder_type"] = trial.suggest_categorical("encoder_type", enc_type_choices)
    hp["encoder_layers"] = trial.suggest_int("encoder_layers", 1, 4)
    # 新規パラメータ名で常に Int 分布に統一（衝突を避ける）
    base_hs = trial.suggest_int("hidden_size_base", 128, 512, step=128)
    if hp["encoder_type"] in ("rnn", "rnn_bidi", "rnn_bidi_onlyfirst"):
        hp["use_preproj_mlp"] = trial.suggest_categorical("use_preproj_mlp", [True, False])
        hp["hidden_size"] = base_hs
    else:
        heads = trial.suggest_int("transformer_heads", 2, 8)
        hp["transformer_heads"] = heads
        hp["transformer_inner_dim"] = trial.suggest_int("transformer_inner_dim", 512, 2048, step=128)
        # スナップして制約を満たす hidden_size を導出（prune しない）
        hp["hidden_size"] = _snap_hidden_size_for_heads(base_hs, heads, 128, 1024, 128)

    # とりあえず、hp["latent_type"] = binary で固定
    hp["latent_type"] = base_hp.get("latent_type", "binary")
    hp["latent_size"] = base_hp.get("latent_size", 64)
    # # Latent configuration
    # latent_choices = ["bottleneck", "vae", "binary", "gumbel"]
    # hp["latent_type"] = trial.suggest_categorical("latent_type", latent_choices)
    # if hp["latent_type"] in ("bottleneck", "vae", "binary", "gumbel"):
    #     hp["latent_size"] = trial.suggest_int("latent_size", 64, 1052, step=32)
    #     # optional KL usage for binary/gumbel/vae
    #     hp["use_kl"] = trial.suggest_categorical("use_kl", [True, False])
    #     hp["kl_coef"] = trial.suggest_float("kl_coef", 1e-4, 5e-2, log=True)

    # Decoder configuration
    hp["decoder_layers"] = trial.suggest_int("decoder_layers", 1, 4)
    # Keep decoder_embedding_dim small if labels are AA indices; use base value or small dims
    hp["decoder_embedding_dim"] = base_hp.get("decoder_embedding_dim", 2)
    # hp["decoder_embedding_dim"] = trial.suggest_int("decoder_embedding_dim", 2, 32, step=6)
    hp["concat_latent_to_words"] = base_hp.get("concat_latent_to_words", True)
    # hp["concat_latent_to_words"] = trial.suggest_categorical("concat_latent_to_words", [True, False])

    # Training runtime: 探索の初期は、探索範囲が小さい方が公平に他のパラメータを判断できる
    hp["num_epochs"] = base_hp.get("num_epochs", 300)
    # hp["num_epochs"] = trial.suggest_int("num_epochs", 100, 400, step=100)
    # AMP dtype stays consistent
    hp["amp_dtype"] = base_hp.get("amp_dtype", "fp32")
    # hp["amp_dtype"] = trial.suggest_categorical("amp_dtype", ["fp16", "bf16", "fp32"])
    hp["use_amp"] = base_hp.get("use_amp", True)
    hp["allow_tf32"] = base_hp.get("allow_tf32", True)

    # pLM config — keep backend stable from base hparams
    backend = base_hp.get("plm_backend", None)
    if backend is None or (isinstance(backend, str) and backend.strip() == ""):
        raise ValueError("plm_backend must be set in base hparams (e.g., 'esm2').")

    hp["plm_backend"] = backend
    hp["plm_encoder_model"] = base_hp.get("plm_encoder_model")
    hp["plm_tokenizer_model"] = base_hp.get("plm_tokenizer_model", hp["plm_encoder_model"])
    hp["plm_add_special_tokens"] = base_hp.get("plm_add_special_tokens", True)
    hp["plm_clean_mode"] = base_hp.get("plm_clean_mode", "strict")
    hp["plm_allow_diff"] = base_hp.get("plm_allow_diff", False)
    hp["plm_batch_size"] = base_hp.get("plm_batch_size", 8)
    hp["plm_device"] = base_hp.get("plm_device", "auto")
    hp["plm_no_skip_existing"] = base_hp.get("plm_no_skip_existing", True)
    hp["compat_plm"] = base_hp.get("compat_plm", True)

    # LoRA/Full-FT parameters remain aligned to base unless you want to search them
    # hp["plm_learning_rate"] = trial.suggest_float("plm_learning_rate", 1e-5, 1e-3, log=True)
    hp["lora_enable"] = base_hp.get("lora_enable", False)
    hp["full_finetune_enable"] = base_hp.get("full_finetune_enable", False)
    # hp["lora_enable"] = trial.suggest_categorical("lora_enable", [True, False])
    # hp["full_finetune_enable"] = trial.suggest_categorical("full_finetune_enable", [True, False])
    # hp["lora_r"] = trial.suggest_categorical("lora_r", [1, 2, 4, 8, 16, 64])
    # hp["lora_alpha"] = trial.suggest_categorical("lora_alpha", [1, 4, 16, 32, 64])
    # hp["lora_dropout"] = trial.suggest_float("lora_dropout", 0, 0.05, step=0.01)
    # hp["lora_bias"] = base_hp.get("lora_bias", "none")
    # hp["lora_target_modules"] = base_hp.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "out_proj"])

    # Weight decay groups
    hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    # hp["full_finetune_enable"] = True の場合
    # hp["full_weight_decay"] = trial.suggest_categorical("full_weight_decay", [0.0, 1e-4, 5e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    # hp["lora_weight_decay"] = base_hp.get("lora_weight_decay", 0.0)

    # Misc
    hp["seed"] = base_hp.get("seed", 42)
    hp["tokens_per_batch"] = base_hp.get("tokens_per_batch", 250)
    hp["buffer_size"] = base_hp.get("buffer_size", 100)
    hp["max_input_length"] = base_hp.get("max_input_length", 100)
    hp["max_output_length"] = base_hp.get("max_output_length", 100)
    hp["vocab_size"] = base_hp.get("vocab_size", 23)

    # Precomputed embeddings setting (keep from base)
    hp["input_is_precomputed"] = base_hp.get("input_is_precomputed", True)
    if hp["input_is_precomputed"]:
        hp["embeddings_path"] = base_hp.get("embeddings_path", "./embeddings/")
        hp["embedding_dim"] = base_hp.get("embedding_dim", 2560)
    else:
        # On-the-fly mode uses PLM directly; embedding_dim derived from PLM hidden size
        hp.pop("embedding_dim", None)

    # Early stopping and LR schedule can be kept or simplified
    hp["early_stopping"] = base_hp.get("early_stopping", None)
    # hp["early_conf"] = trial.suggest_categorical("early_conf", ["early_stopping", None])
    # hp["early_stopping"] = base_hp.get("early_stopping", [{"patience": 10, "min_delta": 1e-3})
    hp["lr_schedule"] = base_hp.get("lr_schedule", None)
    # hp["lr_schedule_conf"] = trial.suggest_categorical("lr_schedule_conf", ["lr_schedule", None])
    # hp["lr_schedule"] = base_hp.get("lr_schedule", {"mode": "cosine", "warmup_epochs": 4, "min_lr_ratio": 0.05})
    # hp["lr_schedule"] = base_hp.get("lr_schedule", {"mode": "plateau", "factor": 0.5, "patience": 3, "min_lr": 1e-6})

    # Freeze behavior for PLM
    # hp["freeze_plm_epochs"] = base_hp.get("freeze_plm_epochs", 0)
    ## hp["lora_enable"] = True or hp["full_finetune_enable"] = True の場合
    # hp["freeze_plm_epochs"] = trial.suggest_categorical("freeze_plm_epochs", [0, 2, 5, 10, 20])

    # Study name handling (FrozenTrial には study が無い)
    if study_name is None:
        # Trial オブジェクトなら getattr(trial, "study", None) を試す
        st = getattr(trial, "study", None)
        if st is not None:
            study_name = st.study_name
        else:
            study_name = "optuna"

    model_name = f"optuna_{study_name}_trial_{trial.number}"
    hp["model_name"] = model_name
    hp["model_path"] = os.path.join(base_hp.get("model_path", "./model_data/optuna/"), model_name)
    hp["output_path"] = os.path.join(base_hp.get("output_path", "./model_output/optuna/"), model_name)
    # 追加: インデックス用サブディレクトリ
    hp["select_indices"] = base_hp.get("select_indices", None)
    hp["allow_sidecar_indices"] = base_hp.get("allow_sidecar_indices", False)

    # Data paths remain aligned to base
    hp["data_path"] = base_hp.get("data_path")

    return hp

def handle_scheduler_and_checkpoint_opt(epoch: int,
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
                                        es_min_delta: float) -> Tuple[float, int, int, bool]:
    """
    Scheduler更新のみ。checkpoint保存は行わない (Optuna用)
    """
    dev_score = dev_stats.get('nll', dev_stats.get('ppl', float('inf')))

    if scheduler is not None:
        if scheduler_mode == 'plateau':
            scheduler.step(dev_score)
        elif scheduler_mode == 'cosine':
            scheduler.step()

    improved = dev_score < (best_dev - (es_min_delta if use_early else 0.0))
    if improved:
        best_dev = dev_score
        best_epoch = epoch
        epochs_no_improve = 0
    else:
        if use_early:
            epochs_no_improve += 1
            if epochs_no_improve >= early_conf.get('patience', 10):
                return best_dev, best_epoch, epochs_no_improve, True
    return best_dev, best_epoch, epochs_no_improve, False

def _save_best_so_far_from_trial(study: optuna.Study,
                                 trial: optuna.Trial,
                                 base_hp: Dict[str, Any],
                                 best_dev: float) -> Optional[str]:
    """
    study.best_trial に依存せず、「今回の trial」からモデルで使用する全ハイパーパラメータを生成して保存する。
    """
    if study is None or trial is None:
        return None

    # FrozenTrial のため study_name を明示的に渡して merged hparams を作る
    full_model_hparams = suggest_hparams(trial, base_hp, study_name=study.study_name)

    base_model_dir = base_hp.get("model_path", "./model_data/optuna/")
    out_dir = os.path.join(base_model_dir, f"optuna_{study.study_name}")
    os.makedirs(out_dir, exist_ok=True)

    out_yaml = os.path.join(out_dir, "best_so_far_model_hparams.yml")
    payload = {
        "study_name": study.study_name,
        "best_trial_number": int(trial.number),
        "best_dev_nll": float(best_dev),
        "model_hparams": full_model_hparams,
    }
    with open(out_yaml, "w") as f:
        yaml.dump(payload, f, allow_unicode=True, sort_keys=False)
    return out_yaml

def _print_trial_summary(trial: optuna.Trial,
                         best_dev: float,
                         hp: Dict[str, Any],
                         best_epoch: int,
                         base_hp: Dict[str, Any]):
    """
    これまでの trial で最も良かった dev nll とハイパーパラメータセット、
    各 trial 終了時に その trial で得られた dev の nll とハイパーパラメータセットを標準出力へ表示
    """
    # これまでのベスト（study.best_trial）を安全に参照
    st = getattr(trial, "study", None)
    if st is not None:
        try:
            # 現在のスタディベスト
            bt = st.best_trial
            if bt is not None and bt.value is not None:
                print(f"[OPTUNA][BEST SO FAR] Trial #{bt.number} dev NLL: {bt.value:.6f}")
                print(f"[OPTUNA][BEST SO FAR] Params: {json.dumps(bt.params, ensure_ascii=False)}", "\n")
            else:
                print("[OPTUNA][BEST SO FAR] Not available yet.", "\n")
        except Exception as e:
            # ここで例外（Record does not exist 等）が出ても保存判定には影響させない
            print(f"[OPTUNA][BEST SO FAR] Not available yet. Reason: {e}", "\n")

        # 「これまでで最良の dev nll」を基準に保存判定
        # 公式の最新ベスト値（存在しない場合は今回の trial の best_dev を代用）
        try:
            current_global_best = float(st.best_value)
        except Exception:
            current_global_best = float(best_dev)

        # 前回保存したベスト値（system_attrs に保持）
        last_saved_best = st.system_attrs.get("best_so_far_value", None)

        # 今回の trial の結果が初回保存（last_saved_best が未設定）か、
        # これまでで最良の dev nll（current_global_best）より改善していれば保存
        should_save = (last_saved_best is None) or (best_dev < current_global_best)

        if should_save:
            # study.best_trial に依存しない保存（今回の trial から完全ハイパラ生成）
            out_path = _save_best_so_far_from_trial(st, trial, base_hp, best_dev)
            st.set_system_attr("best_so_far_value", float(current_global_best))
            st.set_system_attr("best_so_far_trial", int(trial.number))
            if out_path:
                print(f"[OPTUNA] Best improved by trial #{trial.number}; saved to: {out_path}", "\n")
    else:
        print("[OPTUNA][BEST SO FAR] Not available yet.", "\n")

    # 現在のトライアルの結果
    print(f"[OPTUNA][TRIAL {trial.number}] Trial dev NLL: {best_dev:.6f} (best epoch {best_epoch})")
    print(f"[OPTUNA][TRIAL {trial.number}] Suggested params: {json.dumps(trial.params, ensure_ascii=False)}", "\n")

    # user_attr にメタ情報を残しておく
    trial.set_user_attr("trial_dev_nll", best_dev)
    trial.set_user_attr("best_epoch", best_epoch)

def objective(trial: optuna.Trial, base_hp: Dict[str, Any]) -> float:
    """Optuna objective: minimize dev nll."""
    hp = suggest_hparams(trial, base_hp)

    # Environment and device setup
    device = setup_environment(hp)

    # Precompute embeddings if needed and infer embedding_dim
    if bool(hp.get("input_is_precomputed", False)):
        static_idx = hp.get("select_indices", None)
        use_static = static_idx is not None and len(static_idx) > 0
        use_sidecar = bool(hp.get("allow_sidecar_indices", False))
        need_trim_cache = use_static or use_sidecar

        if need_trim_cache:
            train_cache_dir = prepare_trimmed_cache(hp, mode="train", force_rebuild=False)
            dev_cache_dir = prepare_trimmed_cache(hp, mode="dev", force_rebuild=False)
            orig_embeddings_path = hp["embeddings_path"]
            hp["embeddings_path"] = os.path.dirname(train_cache_dir)
            hp["selection_already_applied"] = True
            hp["source_embeddings_path_for_indices"] = orig_embeddings_path
            print(f"[TRIM CACHE][AUTO] enabled because select_indices/sidecar is used.")
            print(f"[TRIM CACHE][AUTO] embeddings_path -> {hp['embeddings_path']}")
        else:
            print("[TRIM CACHE][AUTO] disabled (no select_indices and allow_sidecar_indices=false).")

        ensure_embeddings(hp, modes=("train", "dev"))
        if "embedding_dim" not in hp or hp["embedding_dim"] is None:
            raise KeyError(
                "hparams['embedding_dim'] is required when input_is_precomputed=true. "
                "Please set 'embedding_dim' explicitly in your hparams file."
            )

    # Create model, optimizer, scheduler
    rt = extract_runtime_params(hp)
    model, optimizer, scheduler, scheduler_mode = create_model_and_optimizer(device, hp, rt)

    # AMP scaler
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(rt["use_amp"] and device.type == "cuda" and rt["amp_dtype"] == torch.float16),
    )

    # Train epochs
    best_dev = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(rt["num_epochs"]):
        # Train
        train_stats, optimizer, scheduler, scheduler_mode = run_epoch("train", epoch, model, optimizer, hp, rt, device, scaler, scheduler, scheduler_mode)

        # Dev
        dev_stats, optimizer, scheduler, scheduler_mode = run_epoch("dev", epoch, model, optimizer, hp, rt, device, scaler, scheduler, scheduler_mode)

        # Report intermediate metrics to Optuna
        dev_nll = float(dev_stats.get("nll", float("inf")))
        trial.report(dev_nll, step=epoch)

        # Pruning support
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Scheduler and checkpoint handling; also sets best_dev and tracks early stop conditions
        best_dev, best_epoch, epochs_no_improve, early_stop = handle_scheduler_and_checkpoint_opt(
            epoch=epoch,
            dev_stats=dev_stats,
            best_dev=best_dev,
            best_epoch=best_epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_mode=scheduler_mode,
            hp=hp,
            early_conf=rt["early_conf"] if isinstance(rt["early_conf"], dict) else {},
            epochs_no_improve=epochs_no_improve,
            use_early=isinstance(rt["early_conf"], dict),
            es_min_delta=float(rt["early_conf"].get("min_delta", 1e-3)) if isinstance(rt["early_conf"], dict) else 0.0,
        )
        if early_stop:
            print(f"[OPTUNA][TRIAL {trial.number}] Early stop at epoch {epoch}")
            break

    _print_trial_summary(trial, best_dev, hp, best_epoch, base_hp)
    # Return the best dev nll observed during this trial
    return float(best_dev)

def write_trials_log(study: optuna.Study, base_hp: Dict[str, Any]) -> str:
    """
    全 trial のハイパーパラメータと結果(nll)を txt にログ。
    1行1JSON（trial_number, value, params など）。
    """
    log_dir = os.path.join(base_hp.get("model_path", "./model_data/optuna/"), f"optuna_{study.study_name}")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "optuna_trials_log.txt")

    existing_numbers = set()
    best_so_far: Optional[float] = None

    # 既存ログがあれば読み込み（重複回避 & 既存best継承）
    if os.path.isfile(log_path):
        with open(log_path, "r") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                num = obj.get("number")
                val = obj.get("value")
                if isinstance(num, int):
                    existing_numbers.add(num)
                # 既存best更新
                if isinstance(val, str):
                    try:
                        fval = float(val)
                        if best_so_far is None or fval < best_so_far:
                            best_so_far = fval
                    except Exception:
                        pass

    # 追記モード
    with open(log_path, "a") as f:
        # Trial番号順で安定
        for t in sorted(study.trials, key=lambda x: x.number):
            if t.number in existing_numbers:
                continue
            val = t.value
            if val is None:
                formatted = None
                is_best = None
            else:
                formatted = f"{val:.6f}"
                if best_so_far is None or val < best_so_far:
                    best_so_far = val
                    is_best = "(best)"
                else:
                    is_best = ""
            row = {
                "number": t.number,
                "value": f"{formatted} {is_best}".strip(),
                "params": t.params,
                # "user_attrs": t.user_attrs,
                # "system_attrs": t.system_attrs,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OPTUNA] Trials log written: {log_path}")
    return log_path

def main():
    parser = argparse.ArgumentParser(description="Optuna optimization for plm-bvae.")
    parser.add_argument("--hparams_path", type=str, default="experiment_configs/binary.yml", help="Base hparams YAML.")
    parser.add_argument("--study_name", type=str, default="plm_bvae_opt", help="Optuna study name.")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_plm_bvae.db", help="Optuna storage.")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials to run.")
    parser.add_argument("--pruner", type=str, default="median", choices=["none", "median", "successivehalving"], help="Pruner choice.")
    parser.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random"], help="Sampler choice.")
    args = parser.parse_args()

    base_hp = load_hparams(args.hparams_path)

    # Set up sampler and pruner
    if args.sampler == "tpe":
        sampler = TPESampler(seed=int(base_hp.get("seed", 42)))
    else:
        sampler = optuna.samplers.RandomSampler(seed=int(base_hp.get("seed", 42)))

    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=30, interval_steps=5)
    elif args.pruner == "successivehalving":
        pruner = optuna.pruners.SuccessiveHalvingPruner(reduction_factor=4, min_resource=30, min_early_stopping_rate=0)
    else:
        pruner = optuna.pruners.NopPruner()

    # Create or load study with SQLite storage for persistence and resumption
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    print(f"[OPTUNA] Starting optimization on storage: {args.storage} (study: {args.study_name})")
    try:
        study.optimize(lambda t: objective(t, base_hp), n_trials=args.n_trials, gc_after_trial=True)
    except KeyboardInterrupt:
        print("[OPTUNA] Interrupted by user. Study state preserved in storage.")

    log_path = write_trials_log(study, base_hp)

    print(f"[OPTUNA] Best trial: #{study.best_trial.number}")
    print(f"[OPTUNA] Best value (dev nll): {study.best_value:.6f}")
    print("[OPTUNA] Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  - {k}: {v}")

    # Generate best hparams safely (pass study.study_name explicitly)
    best_hp = suggest_hparams(study.best_trial, base_hp, study_name=study.study_name)
    os.makedirs(best_hp["model_path"], exist_ok=True)
    out_yaml = os.path.join(best_hp["model_path"], "best_hparams_optuna.yml")
    with open(out_yaml, "w") as f:
        yaml.dump(best_hp, f, allow_unicode=True, sort_keys=False)
    print(f"[OPTUNA] Saved best hparams to: {out_yaml}")
    print(f"[OPTUNA] Trials log to: {log_path}")

if __name__ == "__main__":
    main()
