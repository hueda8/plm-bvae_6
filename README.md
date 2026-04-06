# pLM-bVAE

This repository provides a protein sequence modeling framework combining:
- Pretrained Protein Language Models (PLMs: ESM2 / ProtT5 / Ankh) via a unified feature extractor with optional LoRA or full fine-tuning
- Flexible encoder stacks (`rnn`, `rnn_bidi`, `rnn_bidi_onlyfirst`, `transformer`)
- Multiple latent representations (`bottleneck`, `vae`, `binary`, `gumbel`)
- A lightweight decoder for sequence modeling / reconstruction
- Mixed-precision (AMP) training, scheduled learning rate, early stopping
- Optuna-based hyperparameter optimization persisted in SQLite
- Optional simulated / quantum annealing sampling workflow (legacy WS-MOQA concept)
- Precomputed embedding pipeline (faster iteration) and on-the-fly PLM extraction

> Original WS-MOQA description: a whole-spectrum approach balancing biological sequence activity and prediction stability when sampling. This README has been updated to reflect the current code base.

---

## 1. Environment Setup

Create the conda environment:

```bash
conda env create -n moqa -f moqa_env.yml
conda activate moqa
```

Required core packages (already in `moqa_env.yml`):
- `torch` / `cuda` (GPU recommended)
- `transformers`
- `peft` (only if using LoRA)
- `optuna`
- `tqdm`
- `numpy`
- `pandas` (if extending data processing)

Optional (for advanced sampling):
- `dimod`, `dwave-system` (if you enable D-Wave or quantum annealing)
- `accelerate` (the code auto-attempts seeding if installed)

---

## 2. Data Layout

Expected files (from `experiment_configs/binary.yml`):
- `data_path` points to a directory containing:
  - `train.txt`
  - `dev.txt`
  - (optional) `test.txt`
  - Each line: raw amino acid sequence (uppercase; ambiguous residues handled during cleaning)

Precomputed embeddings (if `input_is_precomputed=true`):
```
embeddings/
  train/0.npy 1.npy ...
  dev/0.npy ...
  test/...
```
Each `.npy` file must be shape `[L, D]` (sequence length × embedding_dim). Embedding generation scripts create these automatically if absent.

---

## 3. Configuration

Primary hyperparameters file:
```
experiment_configs/binary.yml
```

Notable keys (excerpt):
- Model core:
  - `encoder_type`: `rnn_bidi` (others: `rnn`, `rnn_bidi_onlyfirst`, `transformer`)
  - `encoder_layers`, `hidden_size`
  - `latent_type`: `binary` (others: `bottleneck`, `vae`, `gumbel`)
  - `latent_size`
  - `decoder_layers`, `decoder_embedding_dim`
- Training:
  - `num_epochs`, `learning_rate`, `plm_learning_rate`
  - `label_smoothing`
  - `clip_grad_norm`, `grad_accum_steps`
  - `early_stopping` (patience / min_delta)
  - `lr_schedule` (supports `"cosine"` or `"plateau"`)
- PLM integration:
  - `plm_backend`: `esm2` | `prott5` | `ankh`
  - `plm_encoder_model`, `plm_tokenizer_model`
  - `plm_add_special_tokens`, `plm_strip_end`
  - `compat_plm` (normalizes tokenization defaults)
  - `freeze_plm_epochs` (freeze then unfreeze schedule)
  - `lora_enable`, `full_finetune_enable` (mutually exclusive)
- Embedding mode:
  - `input_is_precomputed`: `true` uses `embeddings/` structure
  - `embedding_dim`: must match produced features (e.g. 2560 for ESM2 3B)
- Misc:
  - `tokens_per_batch` (dynamic packing to maximize token utilization)
  - `buffer_size` (sequence buffering before forming batches)
  - `vocab_size`: should match AA vocab (23: PAD, EOS, UNK + 20 residues)
  - `seed`: reproducibility
  - `amp_dtype`: `fp16` | `bf16` | `fp32`, with `use_amp`

---

## 4. Feature Extractor (PLM Layer)

Implemented in `model.py` as `HFProteinFeatureExtractor`:
- Valid backends: `esm2`, `prott5`, `ankh`
- Automatically handles:
  - Tokenization spacing for ProtT5 / Ankh (per-residue space separated)
  - Special tokens trimming (length differences of +1/+2)
  - Optional gradient checkpointing (`plm_grad_checkpointing`)
  - LoRA: activates PEFT modules only (`lora_enable=true`)
  - Full fine-tuning: all parameters set `requires_grad=True` (`full_finetune_enable=true`)

Embedding paths:
- Precomputed mode: skip PLM forward pass, load saved `.npy`
- On-the-fly mode: supply raw sequence list to the model forward

---

## 5. Encoder / Latent / Decoder

### EncoderStack
- RNN variants pack variable lengths (GRU). Bidirectional handling:
  - `rnn_bidi`: all layers bidi
  - `rnn_bidi_onlyfirst`: only first layer bidi
- Transformer variant:
  - PositionalEncoding + stack of `MultiheadAttention` + FeedForward blocks

### LatentLayer
- `bottleneck`: linear projection
- `vae`: outputs `mu`, `logsigma` with KL loss
- `binary`: straight-through Bernoulli sampling
- `gumbel`: 2-class relaxed sampling (Gumbel-Softmax)
- Optional inversion via `invert` index (experiments)

### Decoder
- GRU-based, supports concatenating latent to every time step (`concat_latent_to_words`)
- Training alignment: `[PAD]->x1, x1->x2, ..., xN->EOS` with label smoothing

---

## 6. Losses & Metrics
- Primary objective: normalized NLL (`losses['nll']`)
- Perplexity (`losses['ppl']`)
- Optional KL regularization (`losses['kl']`) if `use_kl=true`
- Length masking includes EOS step (`lengths + 1`)

---

## 7. Training

Basic run (precomputed embeddings mode):

```bash
python train.py
# or explicitly
python train.py experiment_configs/binary.yml  # if you adapt train.py to accept path
```

Ensure embeddings exist first (automatic if missing and `input_is_precomputed=true`). To regenerate manually:

```bash
python embed_esm2.py \
  --data_path ./data/train.txt \
  --embeddings_path ./embeddings \
  --mode train \
  --compat_plm
```

(Equivalent scripts: `embed_prott5.py`, `embed_ankh.py`)

### Reproducibility
Seeds fixed via `setup_environment()` → Python `random`, NumPy, Torch CPU/GPU, optional `transformers` and `accelerate`. Determinism toggled by `deterministic` in hparams.

### Mixed Precision
Enabled if `use_amp=true`. Dtype controlled by `amp_dtype`.

---

## 8. Hyperparameter Optimization (Optuna)

File: `optuna_optimize.py`

Objective: minimize dev NLL (`best_dev` tracked across epochs).

SQLite persistence (resume/review):
```bash
# (Optional) Initialize DB on HPC node/scratch
python create_optuna_db.py --use-tmpdir --bootstrap-optuna

# Run optimization (20 trials example)
python optuna_optimize.py \
  --hparams_path experiment_configs/binary.yml \
  --study_name plm_bvae_opt \
  --storage sqlite:///optuna_plm_bvae.db \
  --n_trials 20 \
  --sampler tpe \
  --pruner median
```

Resume later (same study/storage): re-run the command; completed trials remain.

Artifacts:
- Per-trial model checkpoints under `model_path/optuna_<study>_trial_<n>/`
- Best trial merged hparams saved as `best_hparams_optuna.yml`

Search space adjustable inside `suggest_hparams()`.

---

## 9. LoRA / Full Fine-Tuning

In `model.py`:
- Mutually exclusive: set only one of `lora_enable` or `full_finetune_enable`
- On improvement (dev NLL), trainer saves:
  - LoRA adapters → `model_path/lora_plm/`
  - Full FT weights → `model_path/plm_full.pt`
- Freezing schedule via `freeze_plm_epochs` (epochs where PLM params frozen before rebuild of optimizer).

---

## 10. Precomputed vs On-the-Fly

| Mode | hparam `input_is_precomputed` | Input to `SeqModel.forward` | Embedding source | When to use |
|------|------------------------------|------------------------------|------------------|-------------|
| Precomputed | true | Tensor `[B,T,D]` | `.npy` files | Faster iteration / fixed embeddings |
| On-the-fly | false | List[str] sequences | PLM forward pass | Joint tuning / LoRA / full FT |

Switch by editing `experiment_configs/*.yml`.

---

## 11. Embedding Scripts

- `embed_esm2.py` (ESM2)
- `embed_prott5.py` (ProtT5)
- `embed_ankh.py` (Ankh / AutoModel)

Common arguments:
```bash
--data_path <file>          # text file with sequences (one per line)
--embeddings_path <dir>     # root directory; script creates <mode>/ subfolder
--mode train|dev|test
--encoder_model <HF name>
--tokenizer_model <HF name>
--add_special_tokens
--lora_path <adapters dir>
--state_path <full weights .pt>
--compat_plm                # normalize tokenization defaults
```

Scripts automatically:
- Batch sequences
- Handle long sequences (chunking)
- Strip special tokens to match raw sequence length
- Produce `.npy` per line index

---

## 12. Sampling (Legacy WS-MOQA)

If you reintroduce binary quadratic sampling (e.g., simulated annealing / D-Wave):

1. Create `vectors.txt` with initial binary strings under:
   ```
   model_output/binary/vectors.txt
   ```
2. Run sampler (if `sampler.py` provided in your local branch):
   ```bash
   python sampler.py
   ```
3. For D-Wave usage, replace simulated annealing sampler with:
   ```python
   bqm = dimod.BinaryQuadraticModel(model)
   sampler = EmbeddingComposite(
       DWaveSampler(endpoint='https://cloud.dwavesys.com/sapi',
                    token='YOUR_PASSCODE',
                    solver='Advantage_system4.1')
   )
   ```

(This functionality may require re-adding `sampler.py` / QM adapters if removed.)

---

## 13. Common Issues & Tips

| Issue | Cause | Fix |
|-------|-------|-----|
| `ValueError: plm_backend is required but not set` | Missing backend in hparams | Set `"plm_backend": "esm2"` (or valid option) |
| `Unexpected embedding length ...` | Tokenization diff not in {0,1,2} | Verify `plm_add_special_tokens` & `strip_end` correctness |
| CUDA OOM | Large batch / model size | Lower `tokens_per_batch` or use `use_half` (AMP) |
| Slow Optuna trials | High `num_epochs` or large PLMs | Start with smaller search range / fewer epochs | 
| No embeddings found & precomputed mode enabled | `.npy` files missing | Run appropriate `embed_*.py` script or disable precomputed mode |

---

## 14. Extending

- Add new PLM backend: implement detection & model load logic in `HFProteinFeatureExtractor`
- Add new latent type: extend `LatentLayer.forward`
- Add metrics: modify end of training loop in `run_epoch`
- Add new optimizer strategy: replace `build_optimizer` or integrate schedulers

---

## 15. Reproducibility Checklist

- Set `seed` in hparams
- Avoid changing environment mid-run
- Record `best_hparams_optuna.yml`
- Pin model revision via explicit `plm_encoder_model` tag or commit hash (HF hub)

---

## 16. Example Minimal Run

```bash
# 1. Activate environment
conda activate moqa

# 2. (Optional) Generate embeddings
python embed_esm2.py \
  --data_path /path/to/train.txt \
  --embeddings_path ./embeddings \
  --mode train --compat_plm

python embed_esm2.py \
  --data_path /path/to/dev.txt \
  --embeddings_path ./embeddings \
  --mode dev --compat_plm

# 3. Train
python train.py

# 4. Optimize hyperparameters
python optuna_optimize.py --n_trials 10 --storage sqlite:///optuna_plm_bvae.db

# 5. Inspect best trial output
cat model_data/optuna_plm_bvae_opt_trial_<N>/best_hparams_optuna.yml
```

---

## 17. Directory Structure (Typical)

```
.
├── model.py
├── train.py
├── optuna_optimize.py
├── create_optuna_db.py
├── featurizer.py
├── embed_esm2.py
├── embed_prott5.py
├── embed_ankh.py
├── load_hparams.py
├── experiment_configs/
│   └── binary.yml
├── embeddings/
│   ├── train/
│   └── dev/
├── model_data/
│   └── binary/
├── model_output/
│   └── binary/
└── README.md
```

---

## 18. License & Citation

(Insert your licensing terms here—e.g. MIT, Apache 2.0, or institutional notice.)

If you use WS-MOQA / plm-bvae_1 in publications, please cite the original work and any pretrained models (ESM2, ProtT5, Ankh) per their authors’ guidelines.

---

## 19. Contributing

1. Fork & create a feature branch.
2. Add tests or minimal repro scripts if introducing new components.
3. Ensure style & typing consistency (consider running `ruff` or `black` if configured).
4. Submit a Pull Request with clear description & benchmark differences.

---

## 20. Acknowledgments

- Facebook / Meta AI for ESM2
- RostLab for ProtT5
- Contributors of Ankh / protein BERT variants
- PEFT maintainers for LoRA tooling
- Optuna team for hyperparameter optimization framework

---

Feel free to open issues for:
- Additional latent formulations
- New PLM backends
- Performance regressions
- Sampling pipeline clarification

Enjoy modeling!

