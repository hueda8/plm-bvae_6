# PyTorch reimplementation for Python 3.12 + with pLM(+LoRA/full-FT) and AMP-friendly hooks
from typing import Tuple, Dict, Optional, List, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoTokenizer, EsmModel, T5EncoderModel
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


def lengths_to_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    if max_len is None:
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    rng = torch.arange(max_len, device=lengths.device)
    return rng.unsqueeze(0) < lengths.unsqueeze(1)


def step_function(x: torch.Tensor, dtype=None) -> torch.Tensor:
    out = (torch.sign(x) + 1.0) / 2.0
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype)
    return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: (d_model // 2)])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.lin1 = nn.Linear(d_model, dim_feedforward)
        self.lin2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff = self.lin2(self.dropout2(F.gelu(self.lin1(x))))
        x = self.norm2(x + ff)
        state = x.mean(dim=1)
        return x, state


class EncoderStack(nn.Module):
    def __init__(self, hparams: dict, input_dim: int):
        super().__init__()
        self.hp = hparams
        enc_type = self.hp['encoder_type']
        hs = int(self.hp['hidden_size'])
        nl = int(self.hp['encoder_layers'])
        dr = float(self.hp.get('dropout_rate', 0.0))

        self.kind = enc_type
        self.dropout = nn.Dropout(dr) if dr > 0.0 else nn.Identity()

        if enc_type in ['rnn', 'rnn_bidi', 'rnn_bidi_onlyfirst']:
            use_preproj = bool(self.hp.get('use_preproj_mlp', False))
            hp_preproj_hidden = self.hp.get('preproj_hidden', None)
            if hp_preproj_hidden is not None:
                preproj_hidden = int(hp_preproj_hidden)
            else:
                mid_hidden = hs + (input_dim - hs) / 2.0  # 平均値（hs と input_dim の中間）
                preproj_hidden = int(max(min(mid_hidden, max(hs, input_dim)), min(hs, input_dim)))
            preproj_dropout = float(self.hp.get('dropout_rate', 0.0))
            if use_preproj and (input_dim != hs):
                self.preproj = nn.Sequential(
                    nn.Linear(input_dim, preproj_hidden),
                    nn.GELU(),
                    nn.Dropout(preproj_dropout),
                    nn.Linear(preproj_hidden, hs),
                    nn.LayerNorm(hs),
                )
                in_dim = hs
            else:
                self.preproj = nn.Identity()
                in_dim = input_dim

            self.rnns = nn.ModuleList()
            self.bidi_flags = []
            for i in range(nl):
                if enc_type == 'rnn':
                    bidi = False
                elif enc_type == 'rnn_bidi':
                    bidi = True
                elif enc_type == 'rnn_bidi_onlyfirst':
                    bidi = (i == 0)
                else:
                    raise ValueError(f"Unsupported encoder_type: {enc_type}")
                self.bidi_flags.append(bidi)
                gru = nn.GRU(
                    input_size=in_dim,
                    hidden_size=hs,
                    batch_first=True,
                    bidirectional=bidi,
                )
                self.rnns.append(gru)
                in_dim = hs * (2 if bidi else 1)
            self.out_dim = in_dim

        elif enc_type in ['transformer']:
            in_dim = input_dim
            self.proj_in = nn.Linear(in_dim, hs) if in_dim != hs else nn.Identity()
            self.pos = PositionalEncoding(hs, max_len=int(self.hp.get('max_output_length', 1024)))
            self.blocks = nn.ModuleList(
                [TransformerEncoderBlock(hs,
                                         int(self.hp['transformer_heads']),
                                         int(self.hp['transformer_inner_dim']),
                                         float(self.hp.get('dropout_rate', 0.0))) for _ in range(nl)]
            )
            self.out_dim = hs
        else:
            raise ValueError(f'Unsupported encoder_type: {enc_type}')

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        if self.kind in ['rnn', 'rnn_bidi', 'rnn_bidi_onlyfirst']:
            outputs = self.preproj(x)
            states = []
            for gru, bidi in zip(self.rnns, self.bidi_flags):
                packed = nn.utils.rnn.pack_padded_sequence(outputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
                out_packed, h = gru(packed)
                out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=T)
                outputs = self.dropout(out)
                h_cat = torch.cat([h[-2], h[-1]], dim=-1) if bidi else h[-1]
                states.append(h_cat)
            context = outputs
            state = torch.cat(states, dim=-1) if len(states) > 1 else states[0]
            return context, state
        elif self.kind in ['transformer']:
            pad_mask = ~lengths_to_mask(lengths, T)
            h = self.pos(self.proj_in(x))
            cur_state = None
            for blk in self.blocks:
                h, cur_state = blk(h, pad_mask)
            return h, cur_state
        else:
            raise RuntimeError('Unreachable')

def _resolve_lora_targets_from_model(mod: nn.Module) -> List[str]:
    candidate_sets = [
        ["query", "key", "value", "dense"], # ESM2, Ankh
        ["q", "k", "v", "o"], # ProtT5
        ["q_proj", "k_proj", "v_proj", "out_proj"],
    ]
    names = set(n.split('.')[-1] for n, _ in mod.named_modules())
    for c in candidate_sets:
        if any(t in names for t in c):
            return c
    return candidate_sets[0]


class HFProteinFeatureExtractor(nn.Module):
    """
    AutoModel/AutoTokenizer ベースの汎用抽出器（ESM2, ProtT5, Ankh に対応）。
    - backend: "esm2", "prott5", "ankh"
    - プロット用 / 推論用のクリーニングルールを embed_* 実装に合わせて追加
    - LoRA / full-FT に対応（peft 必須）
    """
    VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
    AMBIGUOUS = {"U", "Z", "O", "B", "J"}

    def __init__(self, hparams: dict, device: torch.device):
        super().__init__()
        if not _HAS_TRANSFORMERS:
            raise ImportError("transformers is required for HFProteinFeatureExtractor.")

        self.hp = hparams
        self.device = device
        backend_raw = self.hp.get("plm_backend", None)
        if backend_raw is None:
            raise ValueError("plm_backend is required. Choose one of: esm2, prott5, ankh.")
        self.backend = str(backend_raw).lower()

        self.encoder_model = self.hp.get("plm_encoder_model", None)
        if self.encoder_model is None:
            raise ValueError("encoder_model is required but not set.")
        self.tokenizer_model = self.hp.get("plm_tokenizer_model", self.encoder_model)
        self.add_special_tokens = bool(self.hp.get("plm_add_special_tokens", True))

        # compat_plm によるショートカット: clean_mode をバックエンドに合わせる
        compat_plm = bool(self.hp.get("compat_plm", True))
        clean_mode = self.hp.get("plm_clean_mode", "strict")
        if compat_plm:
            self.add_special_tokens = True
            # バックエンドから適切な clean_mode を決定
            if self.backend == "esm2":
                clean_mode = "esm"
            elif self.backend == "prott5":
                clean_mode = "prott5"
            elif self.backend == "ankh":
                clean_mode = "ankh"
            else:
                raise ValueError(f"Unknown backend '{self.backend}' for compat_plm shortcut.")
        self.clean_mode = clean_mode

        # トークン化モード：prott5/ankh は空白区切りが基本、esm2 は非空白
        self.use_spaced = bool(self.hp.get("plm_tokenize_spaced", self.backend in ("prott5", "ankh", "protbert")))

        self.lora_enable = bool(self.hp.get("lora_enable", False))
        self.full_finetune_enable = bool(self.hp.get("full_finetune_enable", False))
        if self.lora_enable and self.full_finetune_enable:
            raise ValueError("lora_enable and full_finetune_enable cannot both be true.")

        allow_tf32 = bool(self.hp.get("allow_tf32", True))
        try:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        except Exception:
            pass

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model, do_lower_case=False)

        # backend に応じてベースモデルを選択
        if self.backend == "esm2":
            base = EsmModel.from_pretrained(self.encoder_model)
            hidden_size = int(base.config.hidden_size)

        elif self.backend == "prott5":
            base = T5EncoderModel.from_pretrained(self.encoder_model)
            hidden_size = int(base.config.d_model)

        elif self.backend == "ankh":
            base = T5EncoderModel.from_pretrained(self.encoder_model)
            hidden_size = int(base.config.d_model)

        else:
            raise ValueError(f"Unsupported plm_backend: {self.backend}")

        # 勾配チェックポイント（plm_grad_checkpointing）
        if bool(self.hp.get("plm_grad_checkpointing", False)) and hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable()
            if hasattr(base.config, "use_cache"):
                base.config.use_cache = False

        # LoRA / full-FT
        if self.lora_enable:
            if not _HAS_PEFT:
                raise ImportError("peft is required for LoRA.")
            for p in base.parameters():
                p.requires_grad = False
            target_modules = list(self.hp.get("lora_target_modules", _resolve_lora_targets_from_model(base)))
            r = int(self.hp.get("lora_r", 8))
            alpha = int(self.hp.get("lora_alpha", 16))
            dropout = float(self.hp.get("lora_dropout", 0.05))
            bias = self.hp.get("lora_bias", "none")
            try:
                lcfg = LoraConfig(
                    r=r, lora_alpha=alpha, target_modules=target_modules,
                    lora_dropout=dropout, bias=bias, task_type=TaskType.FEATURE_EXTRACTION
                )
            except Exception:
                lcfg = LoraConfig(
                    r=r, lora_alpha=alpha, target_modules=target_modules,
                    lora_dropout=dropout, bias=bias
                )
            self.plm = get_peft_model(base, lcfg)
        else:
            self.plm = base
            req_grad = True if self.full_finetune_enable else False
            for p in self.plm.parameters():
                p.requires_grad = req_grad

        self.plm.to(device)
        self.hidden_size = hidden_size

    # ==== 追加: embed_*.py と同等のクリーニング関数 ====
    def clean_sequence(self, seq: str) -> str:
        raw = seq.strip().upper().replace(" ", "")
        if not raw:
            return ""
        m = self.clean_mode
        if m == "strict":
            # 全非標準を X
            return "".join([c if c in self.VALID_AA else 'X' for c in raw])
        elif m == "esm":
            # U/Z/O -> X, その他非標準は原文維持 (ESM tokenizer 側で <unk> など処理)
            return "".join([c if (c in self.VALID_AA) else ('X' if c in {"U", "Z", "O"} else c) for c in raw])
        elif m == "prott5":
            # ProtT5 推奨: 非標準は全て X
            return "".join([c if c in self.VALID_AA else 'X' for c in raw])
        elif m == "ankh":
            # Ankh: 非標準を X
            return "".join([c if c in self.VALID_AA else 'X' for c in raw])
        else:
            raise ValueError(f"Unknown clean mode: {m}")

    def _remove_special_tokens(self, hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        keep = attention_mask.bool()
        special = torch.tensor([tid in self.tokenizer.all_special_ids for tid in input_ids.tolist()], device=input_ids.device, dtype=torch.bool)
        keep = keep & (~special)
        out = hidden[keep]
        if out.size(0) != seq_len:
            raise RuntimeError(f"Length mismatch after special-token removal: got={out.size(0)} expected={seq_len}")
        return out

    def _embed_single_chunk(self, text: str, seq_len: int, training: bool) -> torch.Tensor:
        toks = self.tokenizer(text, return_tensors="pt", add_special_tokens=self.add_special_tokens, padding=False, truncation=False)
        toks = {k: v.to(self.device) for k, v in toks.items()}

        with torch.set_grad_enabled(training):
            reps = self.plm(**toks).last_hidden_state.squeeze(0)  # [T, D]

        reps = self._remove_special_tokens(hidden=reps, input_ids=toks["input_ids"].squeeze(0), attention_mask=toks["attention_mask"].squeeze(0), seq_len=seq_len)
        return reps

    def forward(self, seqs: List[str], seq_lengths: Optional[torch.Tensor] = None, mode: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        cleaned = [self.clean_sequence(s) for s in seqs]
        if seq_lengths is None:
            seq_lens = [len(s) for s in cleaned]
        else:
            seq_lens = [int(x) for x in seq_lengths.detach().cpu().tolist()]

        training = (mode == "train" and (self.lora_enable or self.full_finetune_enable))
        self.plm.train(training)

        arrays: List[torch.Tensor] = []
        max_chunk_len = 1022
        for seq, seq_len in zip(cleaned, seq_lens):
            if seq_len == 0:
                arrays.append(torch.zeros(0, self.hidden_size, device=self.device, dtype=torch.float32))
                continue

            if len(seq) <= max_chunk_len:
                text = " ".join(list(seq)) if self.use_spaced else seq
                arr = self._embed_single_chunk(text, seq_len, training)
                arrays.append(arr)
            else:
                parts = [seq[i:i + max_chunk_len] for i in range(0, len(seq), max_chunk_len)]
                part_arrays = []
                for p in parts:
                    p_text = " ".join(list(p)) if self.use_spaced else p
                    part_arrays.append(self._embed_single_chunk(p_text, len(p), training))

                arr = torch.cat(part_arrays, dim=0)
                if arr.size(0) != seq_len:
                    raise RuntimeError(f"[LONG SEQ MISMATCH] seq_len={seq_len} emb={arr.size(0)}")
                arrays.append(arr)

        B = len(arrays)
        D = self.hidden_size
        maxL = max(seq_lens) if seq_lens else 0
        if maxL == 0:
            embedded = torch.zeros(B, 0, D, device=self.device, dtype=torch.float32)
            lengths = torch.zeros(B, dtype=torch.long, device=self.device)
            return embedded, lengths

        out_dtype = arrays[0].dtype if arrays else torch.float32
        padded = torch.zeros(B, maxL, D, device=self.device, dtype=out_dtype)
        for i, a in enumerate(arrays):
            L = a.size(0)
            if L > 0:
                padded[i, :L, :] = a
        lengths = torch.tensor(seq_lens, dtype=torch.long, device=self.device)
        return padded, lengths

class LatentLayer(nn.Module):
    def __init__(self, hparams: dict, enc_state_dim: int):
        super().__init__()
        self.hp = hparams
        lt = self.hp['latent_type']
        ls = int(self.hp.get('latent_size', enc_state_dim))
        self.latent_type = lt
        self.latent_size = ls

        if lt == 'bottleneck':
            self.linear = nn.Linear(enc_state_dim, ls, bias=False)
        elif lt == 'vae':
            self.mu = nn.Linear(enc_state_dim, ls)
            self.logsigma = nn.Linear(enc_state_dim, ls)
        elif lt in ['binary', 'gumbel']:
            self.logits = nn.Linear(enc_state_dim, ls)
        elif lt is None:
            pass
        else:
            raise ValueError(f'Unsupported latent_type: {lt}')

    def forward(self, enc_state: torch.Tensor, mode: str, invert: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss: Dict[str, torch.Tensor] = {}
        lt = self.latent_type
        if lt == 'bottleneck':
            z = self.linear(enc_state)
            if invert is not None:
                inv_mask = F.one_hot(invert, num_classes=self.latent_size).float()
                z = (1 - inv_mask) * z + inv_mask * (-z)
            return z, loss
        elif lt == 'vae':
            mu = self.mu(enc_state); logsigma = self.logsigma(enc_state)
            sigma = torch.exp(logsigma)
            kl = 0.5 * (mu.pow(2).sum(-1) + sigma.pow(2).sum(-1)) - logsigma.sum(-1) - self.latent_size / 2.0
            loss['kl'] = kl.mean()
            if invert is not None:
                inv_mask = F.one_hot(invert, num_classes=self.latent_size).float()
                mu = (1 - inv_mask) * mu + inv_mask * (-mu)
            z = mu + sigma * torch.randn_like(sigma) if mode in ['train','eval_sample'] else mu
            return z, loss
        elif lt in ['binary', 'gumbel']:
            logits = self.logits(enc_state)
            if invert is not None:
                inv_mask = F.one_hot(invert, num_classes=self.latent_size).float()
                logits = (1 - inv_mask) * logits + inv_mask * (-logits)
            prob = torch.sigmoid(logits)

            if mode in ['train','eval_sample']:
                if lt == 'binary':
                    u = torch.rand_like(prob)
                    hard = (prob - u >= 0).float()
                    z = hard + prob - prob.detach()
                else:
                    e = logits.unsqueeze(-1)
                    g1 = -torch.log(-torch.log(torch.rand_like(e) + 1e-12) + 2e-12)
                    g2 = -torch.log(-torch.log(torch.rand_like(e) + 1e-12) + 2e-12)
                    cat = torch.softmax(torch.cat([e + g1, -e + g2], dim=-1), dim=-1)
                    z = cat[..., 0]
            else:
                z = (prob >= 0.5).float()

            if bool(self.hp.get('use_kl', False)):
                q = 1.0 - prob
                kl = (prob * torch.log(prob + 1e-12) + q * torch.log(q + 1e-12) - math.log(0.5)).sum(-1)
                loss['kl'] = kl.mean()
            return z, loss
        elif lt is None:
            return enc_state, loss
        else:
            raise RuntimeError('Unreachable')


class Decoder(nn.Module):
    def __init__(self, hparams: dict, emb_weights: nn.Embedding):
        super().__init__()
        self.hp = hparams
        self.emb = emb_weights
        hs = int(self.hp['hidden_size'])
        nl = int(self.hp['decoder_layers'])
        dr = float(self.hp.get('dropout_rate', 0.0))
        self.word_dropout_p = float(self.hp.get('word_dropout', 0.0)) if self.hp.get('word_dropout') else 0.0
        self.concat_latent = bool(self.hp.get('concat_latent_to_words', False))
        in_dim = int(self.hp.get('decoder_embedding_dim', hs)) + (
            int(self.hp.get('latent_size', 0))
            if self.concat_latent and self.hp.get('encoder_type') is not None
            else 0
        )

        latent_in_dim = int(self.hp.get('latent_size', 0)) if self.hp.get('encoder_type') is not None else 0
        if latent_in_dim > 0:
            self.init_proj = nn.Linear(latent_in_dim, hs)
        else:
            self.init_proj = None

        self.rnn = nn.GRU(input_size=in_dim, hidden_size=hs, num_layers=nl, batch_first=True)
        self.dropout = nn.Dropout(dr) if dr > 0.0 else nn.Identity()
        self.proj = nn.Linear(hs, int(self.hp['vocab_size']), bias=False)

    def init_state(self, latent: Optional[torch.Tensor], B: int, device: torch.device) -> torch.Tensor:
        nl = int(self.hp['decoder_layers'])
        hs = int(self.hp['hidden_size'])
        if self.init_proj is not None and latent is not None:
            h0_single = self.init_proj(latent)
            h0 = h0_single.unsqueeze(0).expand(nl, -1, -1).contiguous()
        else:
            h0 = torch.zeros(nl, B, hs, device=device)
        return h0

    def forward_train_eval(self, latent: Optional[torch.Tensor], inputs_tokens: torch.Tensor, mode: str) -> torch.Tensor:
        B, T = inputs_tokens.shape
        x = self.emb(inputs_tokens)
        if self.word_dropout_p > 0.0 and mode == 'train':
            keep_mask = (torch.rand(B, T, 1, device=x.device) >= self.word_dropout_p).float()
            x = x * keep_mask
        if self.concat_latent and (latent is not None) and (self.hp.get('encoder_type') is not None):
            x = torch.cat([x, latent.unsqueeze(1).expand(-1, T, -1)], dim=-1)
        h0 = self.init_state(latent, B, x.device)
        x = self.dropout(x)
        out, _ = self.rnn(x, h0)
        out = self.dropout(out)
        logits = self.proj(out)
        return logits

    @staticmethod
    def _top_p_sample_from_logits(
        logits: torch.Tensor,
        top_p: float,
        temperature: float,
    ) -> torch.Tensor:
        """
        logits: [B, V]
        returns: next token ids [B]
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1].")
        # temperature scaling
        logits = logits / float(temperature)
        # convert to probs
        probs = torch.softmax(logits, dim=-1)  # [B,V]
        # sort probs descending
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)  # [B,V]
        cumsum = torch.cumsum(sorted_probs, dim=-1)  # [B,V]
        # keep smallest set whose cumulative prob >= top_p
        # mask tokens AFTER the cutoff
        cutoff = cumsum > top_p
        # ensure we keep at least 1 token
        cutoff[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        # renormalize
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        # sample in sorted space then map back to original vocab ids
        sampled_in_sorted = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)  # [B]
        next_tok = sorted_idx.gather(dim=-1, index=sampled_in_sorted.unsqueeze(-1)).squeeze(-1)  # [B]
        return next_tok

    @torch.no_grad()
    def forward_decode(
        self,
        latent: torch.Tensor,
        decode_strategy: str,
        top_p: float,
        temperature: float,
        ) -> torch.Tensor:
        B = latent.size(0)
        max_len = int(self.hp['max_output_length'])
        unk_idx = 2; eos_idx = 1
        device = latent.device

        decode_strategy = str(decode_strategy).lower()
        if decode_strategy not in ("greedy", "top_p"):
            raise ValueError(f"Unknown decode_strategy='{decode_strategy}'. Use 'greedy' or 'top_p'.")

        inputs = torch.full((B, 1), 0, dtype=torch.long, device=device)
        h = self.init_state(latent, B, device)
        outputs: List[torch.Tensor] = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            x = self.emb(inputs[:, -1]).unsqueeze(1)
            if self.concat_latent and (latent is not None) and (self.hp.get('encoder_type') is not None):
                x = torch.cat([x, latent.unsqueeze(1)], dim=-1)
            out, h = self.rnn(x, h)
            logits = self.proj(out.squeeze(1))
            logits[:, unk_idx] = -1e9

            if decode_strategy == "greedy":
                next_tok = torch.argmax(logits, dim=-1)
            else:
                next_tok = self._top_p_sample_from_logits(logits, top_p=top_p, temperature=temperature)

            outputs.append(next_tok)
            finished = finished | (next_tok == eos_idx)
            inputs = torch.cat([inputs, next_tok.unsqueeze(1)], dim=1)
            if finished.all():
                break
        out_tensor = torch.stack(outputs, dim=1) if outputs else torch.empty(B, 0, dtype=torch.long, device=device)
        return out_tensor


class SeqModel(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.hp = hparams
        self.use_precomputed = bool(self.hp.get('input_is_precomputed', True))
        self.plm_extractor: Optional[nn.Module] = None

        lora_enable = bool(self.hp.get('lora_enable', False))
        full_ft_enable = bool(self.hp.get('full_finetune_enable', False))

        if self.use_precomputed and lora_enable:
            raise ValueError("Invalid hparams: input_is_precomputed=True cannot be used with lora_enable=True. Set input_is_precomputed=False to train LoRA, or set lora_enable=False.")
        if self.use_precomputed and full_ft_enable:
            raise ValueError("Invalid hparams: input_is_precomputed=True cannot be used with full_finetune_enable=True. Set input_is_precomputed=False to fine-tune PLM, or set full_finetune_enable=False.")

        if not self.use_precomputed:
            if self.hp.get("plm_backend", None) is None:
                raise ValueError("plm_backend is required but not set. Choose one of: esm2, prott5, ankh.")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.plm_extractor = HFProteinFeatureExtractor(self.hp, device=device)
            input_dim = self.plm_extractor.hidden_size
        else:
            input_dim = int(self.hp.get('embedding_dim', self.hp.get('hidden_size', 512)))

        if self.hp.get('encoder_type') is not None:
            self.encoder_stack = EncoderStack(self.hp, input_dim=input_dim)
        else:
            self.encoder_stack = None

        et = self.hp.get('encoder_type')
        hs = int(self.hp['hidden_size'])
        nl = int(self.hp['encoder_layers'])
        if self.encoder_stack is not None:
            if et == 'rnn':
                enc_state_dim = hs * nl
            elif et == 'rnn_bidi_onlyfirst':
                enc_state_dim = (hs * 2) + hs * (nl - 1) if nl >= 1 else 0
            elif et == 'rnn_bidi':
                enc_state_dim = hs * 2 * nl
            elif et == 'transformer':
                enc_state_dim = self.encoder_stack.out_dim
            else:
                raise NotImplementedError
        else:
            enc_state_dim = int(self.hp.get('latent_size', 0))

        lt_use = self.hp.get('latent_type', None)
        self.latent_layer = LatentLayer(self.hp, enc_state_dim) if lt_use is not None else None

        vocab_size = int(self.hp['vocab_size'])
        dec_dim = int(self.hp.get('decoder_embedding_dim', self.hp['hidden_size']))
        self.decoder_embeddings = nn.Embedding(vocab_size, dec_dim)
        self.decoder = Decoder(self.hp, self.decoder_embeddings)

    def forward(
        self,
        inputs: Union[torch.Tensor, List[str]],
        mode: str,
        labels: Optional[torch.Tensor] = None,
        invert: Optional[torch.Tensor] = None,
        select_indices_batch: Optional[List[List[int]]] = None  # 追加: サンプルごとの選択インデックス
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:

        if mode != 'decode':
            if self.use_precomputed:
                if not isinstance(inputs, torch.Tensor):
                    raise ValueError("Precomputed path expects torch.Tensor [B,T,D]")
                embedded = inputs
                if embedded.dim() != 3:
                    raise ValueError("Embedded input must be rank 3 [B,T,D].")
                if mode == 'train':
                    noise_std = float(self.hp.get('input_noise_std', 0.0))
                    if noise_std > 0.0:
                        embedded = embedded + torch.randn_like(embedded) * noise_std
                    drop_rate = float(self.hp.get('input_dropout_rate', 0.0))
                    if drop_rate > 0.0:
                        keep = 1.0 - drop_rate
                        B, T, _ = embedded.shape
                        mask = (torch.rand(B, T, device=embedded.device) < keep).float().unsqueeze(-1)
                        embedded = embedded * mask / max(keep, 1e-6)
                if labels is not None:
                    mask = step_function(labels.float()).to(dtype=torch.int64)
                    lengths_out = mask.sum(dim=1)
                    lengths = (lengths_out - 1).clamp(min=0).to(torch.long)
                else:
                    nonzero = (embedded != 0.0).any(dim=2)
                    lengths = nonzero.sum(dim=1).to(torch.long)

                # precomputed mode:
                # selection must be applied in featurizer.py to keep embeddings and labels aligned.
                # If select_indices_batch is passed here, it likely causes double-selection bugs.
                if select_indices_batch is not None:
                    raise ValueError(
                        "select_indices_batch is not supported in precomputed mode. "
                        "Apply selection in featurizer.batch_generator (slicing emb/labels + re-adding EOS) "
                        "and pass select_indices_batch only in on-the-fly text mode."
                    )

            else:
                if not isinstance(inputs, list):
                    raise ValueError("On-the-fly ESM2/ProtT5/Ankh path expects inputs as list[str] sequences.")
                seq_lengths = None
                if labels is not None:
                    m = step_function(labels.float()).to(dtype=torch.int64)
                    lengths_out = m.sum(dim=1)
                    seq_lengths = (lengths_out - 1).clamp(min=0).to(torch.long)
                embedded, lengths = self.plm_extractor(inputs, seq_lengths=seq_lengths, mode=mode)

                has_any_selection = (
                    select_indices_batch is not None
                    and any((sel is not None) and (len(sel) > 0) for sel in select_indices_batch)
                )

                # 追加: サンプルごとのインデックス選択
                if has_any_selection:
                    B, T, D = embedded.shape
                    arrays = []
                    new_lengths = []
                    for b in range(B):
                        sel = select_indices_batch[b] if b < len(select_indices_batch) else None
                        if sel is not None and len(sel) > 0:
                            sel_t = torch.tensor([i for i in sel if 0 <= i < T], dtype=torch.long, device=embedded.device)
                            arr = embedded[b].index_select(0, sel_t)
                            arrays.append(arr)
                            new_lengths.append(arr.size(0))
                        else:
                            arrays.append(embedded[b, :lengths[b], :])
                            new_lengths.append(int(lengths[b].item()))
                    maxL = max(new_lengths) if new_lengths else 0
                    padded = torch.zeros(B, maxL, embedded.size(-1), device=embedded.device, dtype=embedded.dtype)
                    for b, a in enumerate(arrays):
                        Lb = a.size(0)
                        if Lb > 0:
                            padded[b, :Lb, :] = a
                    embedded = padded
                    lengths = torch.tensor(new_lengths, dtype=torch.long, device=embedded.device)

                    # ここからが修正点: labels も select に合わせて再構築する（[x_sel1, ..., x_selK, EOS]）
                    if labels is not None:
                        # 1) 元 labels から各サンプルの実長（EOSを除いた長さ）を算出
                        m_lab = step_function(labels.float()).to(dtype=torch.int64)
                        lengths_out_lab = m_lab.sum(dim=1)  # = 実長 + 1(EOS)

                        # 固定ID（featurizerと整合）：PAD=0, EOS=1
                        eos_id = 1
                        pad_id = 0

                        # 選択後の最長長に合わせて新しい labels を [B, max_sel+1] で用意（+1 は EOS）
                        max_sel = int(lengths.max().item()) if lengths.numel() > 0 else 0
                        new_labels = torch.full(
                            (labels.size(0), max_sel + 1),
                            fill_value=pad_id,
                            dtype=labels.dtype,
                            device=labels.device,
                        )

                        for b in range(labels.size(0)):
                            # 元の実長（EOSを除いた長さ）
                            Lb_real = int(max(lengths_out_lab[b].item() - 1, 0))
                            # このサンプルの選択インデックス
                            sel = select_indices_batch[b] if b < len(select_indices_batch) else None
                            if sel is not None and len(sel) > 0:
                                # 実長範囲内にクリップ
                                sel_valid = [i for i in sel if 0 <= i < Lb_real]
                                k = len(sel_valid)  # 選択後の長さ
                                if k > 0:
                                    idx_t = torch.tensor(sel_valid, dtype=torch.long, device=labels.device)
                                    picked = labels[b, :Lb_real].index_select(0, idx_t)  # [k]
                                    new_labels[b, :k] = picked
                                # EOS を末尾に付与
                                new_labels[b, k] = eos_id
                            else:
                                new_labels[b, :Lb_real] = labels[b, :Lb_real]
                                new_labels[b, Lb_real] = eos_id

                        # labels を置き換え
                        labels = new_labels

                if mode == 'train':
                    noise_std = float(self.hp.get('input_noise_std', 0.0))
                    if noise_std > 0.0:
                        embedded = embedded + torch.randn_like(embedded) * noise_std

                    drop_rate = float(self.hp.get('input_dropout_rate', 0.0))
                    if drop_rate > 0.0:
                        keep = 1.0 - drop_rate
                        B, T, _ = embedded.shape
                        mask = (torch.rand(B, T, device=embedded.device) < keep).float().unsqueeze(-1)
                        embedded = embedded * mask / max(keep, 1e-6)

        else:
            embedded = None
            lengths = None

        loss_latent: Dict[str, torch.Tensor] = {}
        if (self.hp.get('encoder_type') is not None) and (mode != 'decode'):
            context, state = self.encoder_stack(embedded, lengths, mode)
            concat_state = state

            if self.latent_layer is not None:
                latent, loss_latent = self.latent_layer(concat_state, mode, invert)
            else:
                latent = concat_state
        elif mode == 'decode':
            latent = inputs
        else:
            latent = None

        if mode == 'encode':
            return latent, {}

        if mode == 'decode':
            # 追加: hparams から decode 戦略を選べるようにする（デフォルト greedy）
            decode_strategy = str(self.hp.get("decode_strategy", "greedy"))
            top_p = float(self.hp.get("decode_top_p", 0.9))
            temperature = float(self.hp.get("decode_temperature", 1.0))

            probs = self.decoder.forward_decode(
                latent,
                decode_strategy=decode_strategy,
                top_p=top_p,
                temperature=temperature,
            )
            return probs, {}

        if labels is None:
            raise ValueError("labels are required in train/eval modes.")

        # 変更点: 先頭の1ステップ（x1）も学習に含めるため、PAD(=0) をBOS相当として前置し、
        # inputs_for_decoder と true を同じ長さに整列させる。
        # これにより、予測ペアは [PAD->x1, x1->x2, ..., xN->EOS] となる。
        B = labels.size(0)
        bos_col = torch.zeros(B, 1, dtype=torch.long, device=labels.device)  # PAD列（BOS等価）
        inputs_for_decoder = torch.cat([bos_col, labels[:, :-1]], dim=1).contiguous().to(torch.long)

        # Decoder は logits を返す
        logits = self.decoder.forward_train_eval(latent, inputs_for_decoder, mode)   # [B, T_pred, V]

        # ===== Losses aligned to prediction steps (float32 log_softmax for stability) =====
        losses: Dict[str, torch.Tensor] = {}
        vocab_size = logits.size(-1)
        T_pred = logits.size(1)

        # 変更点: true は labels の先頭から T_pred 分をそのまま使用
        # [x1, x2, ..., xN, EOS]（長さは inputs_for_decoder と同一）
        true = labels[:, :T_pred].contiguous()      # [B, T_pred]
        log_probs = F.log_softmax(logits.float(), dim=-1)  # [B, T_pred, V] in float32

        # 変更点: マスクは lengths + 1 に拡張（x1 と EOS を両方カバー）
        ce_mask = lengths_to_mask(lengths + 1, T_pred).to(dtype=log_probs.dtype)   # [B, T_pred]

        nll_tok = -(log_probs.gather(dim=2, index=true.unsqueeze(-1)).squeeze(-1)) # [B, T_pred]

        token_losses = nll_tok * ce_mask                                       # [B, T_pred]
        token_counts = ce_mask.sum(dim=1).clamp_min(1.0)                       # [B]

        # Token-average NLL (length normalized principal loss)
        nll_avg_seq = token_losses.sum(dim=1) / token_counts                   # [B]
        ppl_seq = torch.exp(nll_avg_seq)

        smooth = float(self.hp.get('label_smoothing', 0.0))
        apply_smoothing = (smooth > 0.0) and (mode == 'train')
        if apply_smoothing:
            u = 1.0 / float(vocab_size)
            one_hot = F.one_hot(true, num_classes=vocab_size).to(log_probs.dtype)
            targets = (1.0 - smooth) * one_hot + smooth * u

            nll_s_tok = -(targets * log_probs).sum(dim=-1)   # [B, T_pred]

            nll_s_sum_seq = (nll_s_tok * ce_mask).sum(dim=1)
            nll_s_avg_seq = nll_s_sum_seq / token_counts
            ppl_s_seq = torch.exp(nll_s_avg_seq)
            if mode == 'train':
                nll_final = nll_s_avg_seq.mean()
                ppl_final = ppl_s_seq.mean()
            else:
                nll_final = nll_avg_seq.mean()
                ppl_final = ppl_seq.mean()
        else:
            nll_final = nll_avg_seq.mean()
            ppl_final = ppl_seq.mean()

        losses['nll'] = nll_final
        losses['ppl'] = ppl_final
        losses.update(loss_latent)
        return logits, losses
