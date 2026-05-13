import numpy as np
import dimod
import os
import time
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from parrot import py_predictor as ppp

# TorchFM separated module (as discussed)
from fmqa import TorchFMBQM

# -----------------------------
# Config
# -----------------------------
OPTIMIZE_DIRECTION = "min"  # "min" or "max"

# FMQA
FM_RANK = 8
FM_LR = 0.01
FM_WEIGHT_DECAY = 0.01
FM_EPOCHS = 1000
FM_PATIENCE = 50
FM_AUTO_SCALE = False
FM_VAL_RATIO = 0.2
FM_BATCH_SIZE = 64



#CHAIN_STRENGTH = 1
ANNEALING_TIME = 20
NUM_READS = 10
MAX_RETRIES = 20
N_ITER = 200

DWAVE_ENDPOINT = "https://cloud.dwavesys.com/sapi"
DWAVE_SOLVER = "Advantage_system4.1"
DWAVE_TOKEN = "XXXX"  # set your token

# penalty for invalid/failed prediction (minimization)
PENALTY_SCORE = 1.0e6
MAX_SAME_AA_RUN = 5

# -----------------------------
# Predictor (single objective)
# -----------------------------
my_predictor_1 = ppp.Predictor(
    "./parrot_models/model_network.pt",
    dtype="sequence",
)


# -----------------------------
# QPU diagnostics helpers
# -----------------------------
def _get_max_chain_length(embedding):
    if not embedding:
        return 0
    return max(len(chain) for chain in embedding.values())


def _max_chain_length_from_anywhere(res, sampler):
    emb = getattr(getattr(sampler, "child", None), "embedding", None)
    if emb:
        return int(_get_max_chain_length(emb))

    info = getattr(res, "info", {}) or {}
    ctx = info.get("embedding_context")
    if isinstance(ctx, dict):
        emb2 = ctx.get("embedding")
        if emb2:
            return int(_get_max_chain_length(emb2))
    return 0


def _safe_get_chain_strength_from_info(res):
    info = getattr(res, "info", {}) or {}
    for key in ("chain_strength", "embedding_context", "chain_strengths"):
        if key in info:
            val = info[key]
            if isinstance(val, dict) and "chain_strength" in val:
                return val["chain_strength"]
            return val
    return None

def compute_energy_metrics(res, bqm):
    offset = float(getattr(bqm, "offset", 0.0))
    max_abs_h = float(max((abs(v) for v in bqm.linear.values()), default=0.0))
    max_abs_J = float(max((abs(v) for v in bqm.quadratic.values()), default=0.0))
    scale = max(max_abs_h, max_abs_J)
    num_vars = int(len(bqm.variables))
    denom = num_vars * scale

    E = res.record.energy.astype(float) # hamiltonian_reads
    cbf_reads = res.record["chain_break_fraction"].astype(float)

    E0 = E - offset
    E_norm_reads = np.full_like(E0, np.nan, dtype=float) if denom == 0.0 else (E0 / denom)

    best_i = int(np.argmin(E))
    best_energy = float(E[best_i])
    cbf_best = float(cbf_reads[best_i])
    E_best_norm = float(E_norm_reads[best_i]) if len(E_norm_reads) > 0 else float("nan")

    return {
        "offset": offset,
        "max_abs_h": max_abs_h,
        "max_abs_J": max_abs_J,
        "E": E,
        "cbf_reads": cbf_reads,
        "E_norm_reads": E_norm_reads,
        "best_i": best_i,
        "best_energy": best_energy,
        "cbf_best": cbf_best,
        "E_best_norm": E_best_norm,
    }


def append_qpu_log_csv(csv_path, iter_idx, res, sampler, bqm, requested_chain_strength=None):
    m = compute_energy_metrics(res, bqm)
    
    cbf_mean = float(m["cbf_reads"].mean())
    cbf_max = float(m["cbf_reads"].max())
    max_chain_length = _max_chain_length_from_anywhere(res, sampler)

    used_chain_strength = requested_chain_strength
    if used_chain_strength is None:
        used_chain_strength = _safe_get_chain_strength_from_info(res)
        
    need_header = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if need_header:
            f.write(
                "iter,best_energy,E_norm,offset,max_abs_h,max_abs_J,cbf_mean,cbf_max,cbf_best,max_chain_length,chain_strength\n"
            )
        f.write(
            f"{iter_idx},{m['best_energy']:.16g},{m['E_best_norm']:.16g},"
            f"{m['offset']:.16g},{m['max_abs_h']:.16g},{m['max_abs_J']:.16g},"
            f"{cbf_mean:.16g},{cbf_max:.16g},{m['cbf_best']:.16g},{max_chain_length},"
            f"{'' if used_chain_strength is None else used_chain_strength}\n"
        )


def sample_with_more_embedding_effort(
    bqm, token, num_reads, annealing_time, solver, endpoint, max_retries, #chain_strength,
):
    last = None
    for k in range(1, max_retries + 1):
        try:
            sampler = EmbeddingComposite(
                DWaveSampler(endpoint=endpoint, token=token, solver=solver),
            )
            return sampler.sample(
                bqm,
                num_reads=num_reads,
                annealing_time=annealing_time,
                #chain_strength=chain_strength,
                chain_break_fraction=True,
            )
        except ValueError as e:
            last = e
            print(f"[attempt {k}/{max_retries}] embedding failed: {e}")
            time.sleep(0.2)
    raise last


# -----------------------------
# Sequence utilities
# -----------------------------
def seq_fix():
    with open("./model_output/binary/decoded.txt", "r") as f:
        lines = f.readlines()

    seq_d = [i.replace(" ", "").rstrip() for i in lines]

    bad_pad = [s for s in seq_d if "<pad>" in s]
    bad_empty = [s for s in seq_d if len(s) == 0]
    bad_unk = [s for s in seq_d if "<unk>" in s]

    filler = "S" * 56

    for t in bad_pad:
        for i in range(len(seq_d)):
            if seq_d[i] == t:
                seq_d[i] = filler

    for t in bad_empty:
        for i in range(len(seq_d)):
            if seq_d[i] == t:
                seq_d[i] = filler

    for t in bad_unk:
        for i in range(len(seq_d)):
            if seq_d[i] == t:
                seq_d[i] = filler

    return seq_d

def has_run_of_same_char(s: str, n: int) -> bool:
    if n <= 1:
        return len(s) > 0
    run = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            run += 1
            if run >= n:
                return True
        else:
            run = 1
    return False

def to_minimization_target(values: np.ndarray, direction: str) -> np.ndarray:
    if direction == "min":
        return values.astype(np.float32)
    if direction == "max":
        return (-values).astype(np.float32)
    raise ValueError("direction must be 'min' or 'max'")

def black_box_function(seq_list):
    vals = []
    for s in seq_list:
        s = (s or "").strip()
        if len(s) == 0 or "<pad>" in s or "<unk>" in s:
            vals.append(PENALTY_SCORE)
            continue
        
        # 同一文字が n 連続以上ならペナルティ
        if has_run_of_same_char(s, MAX_SAME_AA_RUN):
            vals.append(PENALTY_SCORE)
            continue
        
        try:
            v = float(my_predictor_1.predict(s)[0])  # full sequence directly
            if np.isnan(v) or np.isinf(v):
                v = PENALTY_SCORE
            vals.append(v)
        except Exception:
            vals.append(PENALTY_SCORE)
    return np.asarray(vals, dtype=np.float32)

def filter_valid_for_fm(x: np.ndarray, y_raw: np.ndarray, penalty_score: float):
    mask = np.isfinite(y_raw) & (y_raw < penalty_score)
    return x[mask], y_raw[mask], mask

def standardize_targets(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    y = np.asarray(y, dtype=np.float32)
    mu = float(np.mean(y))
    sigma = float(np.std(y))
    if not np.isfinite(sigma) or sigma < 1e-12:
        return (y - mu).astype(np.float32), mu, 1.0
    return ((y - mu) / sigma).astype(np.float32), mu, sigma

def destandardize_targets(y_std: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (np.asarray(y_std, dtype=np.float32) * np.float32(sigma) + np.float32(mu)).astype(np.float32)

# -----------------------------
# Main
# -----------------------------
def main():
    # log directory
    os.makedirs("./model_output/binary/fm_log", exist_ok=True)
    # Decode initial binary vector set
    os.system(
        'CUDA_VISIBLE_DEVICES="" python decode_file.py --hparams_path experiment_configs/binary.yml '
        "--checkpoint_kind best "
        "--decode_strategy greedy"
    )

    c0 = [x.split(" ")[:] for x in open("./model_output/binary/vectors.txt").readlines()]

    vectors_all = []
    for k in range(len(c0)):
        vectors = []
        for i in c0[k]:
            vectors.append(int(float(i)))
        vectors_all.append(vectors)
    vectors_all = np.array(vectors_all, dtype=np.float32)

    # Initial sequence eval (single objective)
    seq_d_all = seq_fix()
    scores_raw = black_box_function(seq_d_all) # for reporting
    scores = to_minimization_target(scores_raw, OPTIMIZE_DIRECTION)  # FM学習用ターゲット: max の場合符号反転

    # FM学習用: penaltyを除外
    x_train, y_raw_train, _ = filter_valid_for_fm(vectors_all, scores_raw, PENALTY_SCORE)
    y_train = to_minimization_target(y_raw_train, OPTIMIZE_DIRECTION)
    if len(y_train) == 0:
        raise RuntimeError("No valid (non-penalty) samples for initial FM training.")

    # standardize
    y_train_std, y_mu, y_sigma = standardize_targets(y_train)
    current_y_mu, current_y_sigma = y_mu, y_sigma
    print(f"[iter 0] y_train standardize: mean={y_mu:.16g}, std={y_sigma:.16g}", flush=True)

    # log initial best (minimize)
    init_best_bb = np.min(scores_raw) if OPTIMIZE_DIRECTION == "min" else np.max(scores_raw)
    with open("./model_output/binary/all_points_best.txt", "w") as oo:
        oo.write(f"{0} {init_best_bb:.16g}\n")
    # log
    print(
        f"[iter 0] best_bb={init_best_bb:.16g} "
        f"(direction={OPTIMIZE_DIRECTION})",
        flush=True,
    )
    
    # initialize logs/files
    os.system("rm -f ./model_output/binary/fm_log/fm_train_log_iter_*.csv")
    # FM training (requested style)
    fmbqm = TorchFMBQM.from_data(
        x_train,
        y_train_std,
        k=FM_RANK,
        lr=FM_LR,
        weight_decay=FM_WEIGHT_DECAY,
        epochs=FM_EPOCHS,
        patience=FM_PATIENCE,
        auto_scale=FM_AUTO_SCALE,
        val_ratio=FM_VAL_RATIO,
        batch_size=FM_BATCH_SIZE,
        split_seed=42,
        log_path="./model_output/binary/fm_log/fm_train_log_iter_0000.csv",
    )
    bqm = fmbqm.to_bqm()
    
    # Simulated annealing for sampling
    #sampler = dimod.samplers.SimulatedAnnealingSampler()

    # For performance assessment use, replace "sampler" with the following code.
    #sampler = dimod.samplers.RandomSampler()

    # D-wave
    sampler = EmbeddingComposite(
        DWaveSampler(
            endpoint=DWAVE_ENDPOINT,
            token=DWAVE_TOKEN,
            solver=DWAVE_SOLVER,
        )
    )

    # initialize logs/files
    os.system("rm -f ./model_output/binary/vectors_list.txt")
    os.system("rm -f ./model_output/binary/decoded_list.txt")
    os.system("rm -f ./model_output/binary/all_points_samples.txt")
    os.system("rm -f ./model_output/binary/qpu_diag_log.csv")

    # accumulated targets for FM (always minimization target)
    scores_all = scores.copy()
    # accumulated raw black-box values (for reporting)
    scores_raw_all = scores_raw.copy()

    for iter_idx in range(N_ITER):
        
        # Simulated annealing or Random sampling
        #res = sampler.sample(bqm, num_reads=NUM_READS)
        
        # D-wave
        try:
            res = sampler.sample(
                bqm,
                annealing_time=ANNEALING_TIME,
                num_reads=NUM_READS,
                #chain_strength=CHAIN_STRENGTH,
                chain_break_fraction=True,
            )
        except ValueError as e:
            print("[DWave embedding] primary sampler failed, retrying with more effort:", e, flush=True)
            res = sample_with_more_embedding_effort(
                bqm,
                endpoint=DWAVE_ENDPOINT,
                token=DWAVE_TOKEN,
                solver=DWAVE_SOLVER,
                annealing_time=ANNEALING_TIME,
                num_reads=NUM_READS,
                #chain_strength=CHAIN_STRENGTH,
                max_retries=MAX_RETRIES,
            )

        append_qpu_log_csv(
            "./model_output/binary/qpu_diag_log.csv",
            iter_idx + 1,
            res,
            sampler,
            bqm,
            #CHAIN_STRENGTH,
        )

        # append sampled binary vectors
        vectors_all = np.r_[vectors_all, res.record["sample"]].astype(np.float32)
        vectors_sample = np.r_[res.record["sample"]]

        vectors_sample_out = [" ".join(str(int(v)) for v in row) for row in vectors_sample]

        with open("./model_output/binary/vectors.txt", "w") as f:
            for s in vectors_sample_out:
                f.write(s + "\n")

        with open("./model_output/binary/vectors_list.txt", "a") as f:
            for s in vectors_sample_out:
                f.write(s + "\n")

        # decode sampled vectors
        with open("./model_output/binary/decoded.txt", "w"):
            os.system(
                'CUDA_VISIBLE_DEVICES="" python decode_file.py --hparams_path experiment_configs/binary.yml '
                "--checkpoint_kind best "
                "--decode_strategy greedy"
            )

        with open("./model_output/binary/decoded_list.txt", "a") as f_out:
            for line in open("./model_output/binary/decoded.txt").readlines():
                f_out.write(line)

        seq_d = seq_fix()

        # evaluate sampled sequences (single objective)
        scores_sample_raw = black_box_function(seq_d)
        scores_sample = to_minimization_target(scores_sample_raw, OPTIMIZE_DIRECTION) # FM学習用ターゲット: max の場合符号反転
        
        scores_all = np.r_[scores_all, scores_sample].astype(np.float32)
        scores_raw_all = np.r_[scores_raw_all, scores_sample_raw].astype(np.float32)

        # energy diagnostics (shared)
        metrics = compute_energy_metrics(res, bqm)

        E = metrics["E"]  # hamiltonian_reads
        cbf_reads = metrics["cbf_reads"]
        E_norm_reads = metrics["E_norm_reads"]
        E_best_norm = metrics["E_best_norm"]
        cbf_best = metrics["cbf_best"]

        fm_preds_sample_std = fmbqm.predict(vectors_sample).astype(np.float32) # FM_pred
        fm_preds_sample = destandardize_targets(fm_preds_sample_std, current_y_mu, current_y_sigma)  # destandardize

        # best-so-far (minimize objective)
        with open("./model_output/binary/all_points_best.txt", "a") as oo:
            best_bb = np.min(scores_raw_all) if OPTIMIZE_DIRECTION == "min" else np.max(scores_raw_all)
            oo.write(f"{iter_idx+1} {best_bb:.16g} {E_best_norm:.16g} {cbf_best:.16g}\n")
        # log
        print(
            f"[iter {iter_idx+1}] best_bb={best_bb:.16g} "
            f"E_best_norm={E_best_norm:.16g} cbf_best={cbf_best:.16g}",
            flush=True,
        )

        with open("./model_output/binary/all_points_samples.txt", "a") as oo:
            for i in range(len(scores_sample_raw)):
                oo.write(
                    f"{iter_idx+1} "
                    f"{scores_sample_raw[i]:.16g} "
                    f"{E_norm_reads[i]:.16g} "
                    f"{cbf_reads[i]:.16g} "
                    f"{E[i]:.16g} "
                    f"{float(fm_preds_sample[i]):.16g}\n"
                )

        # FM再学習用: 累積データから penalty を除外
        x_train_all, y_raw_train_all, _ = filter_valid_for_fm(vectors_all, scores_raw_all, PENALTY_SCORE)
        y_train_all = to_minimization_target(y_raw_train_all, OPTIMIZE_DIRECTION)

        if len(y_train_all) == 0:
            print(f"[iter {iter_idx+1}] skip FM training: no valid non-penalty samples", flush=True)
        else:
            # stamdardize
            y_train_all_std, y_all_mu, y_all_sigma = standardize_targets(y_train_all)
            current_y_mu, current_y_sigma = y_all_mu, y_all_sigma
            print(f"[iter {iter_idx+1}] y_train standardize: mean={y_all_mu:.16g}, std={y_all_sigma:.16g}", flush=True)

            # FM re-training (requested style)
            iter_log_path = f"./model_output/binary/fm_log/fm_train_log_iter_{iter_idx+1:04d}.csv"
            fmbqm.train(
                x_train_all,
                y_train_all_std,
                lr=FM_LR,
                weight_decay=FM_WEIGHT_DECAY,
                epochs=FM_EPOCHS,
                patience=FM_PATIENCE,
                val_ratio=FM_VAL_RATIO,
                batch_size=FM_BATCH_SIZE,
                split_seed=42 + iter_idx, # 反復ごとに再現可能な変化をつける
                log_path=iter_log_path,
            )
            # D-wave
            bqm = fmbqm.to_bqm()


if __name__ == "__main__":
    main()
