import numpy as np
import fmqa
import dimod
import os
import time
import subprocess
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from itertools import groupby
from functools import reduce
from parrot import py_predictor as ppp
import pandas as pd


# Loading regression models
my_predictor_1 = ppp.Predictor('./parrot_models/full_log_deldup_weight1_opt_network.pt', dtype='sequence')
my_predictor_2 = ppp.Predictor('./parrot_models/full_log_deldup_weight1_opt_2nd_network.pt', dtype='sequence')
my_predictor_3 = ppp.Predictor('./parrot_models/full_log_deldup_weight1_opt_3rd_network.pt', dtype='sequence')
#my_predictor_4 = ppp.Predictor('./parrot_models/network_final_pred_model4.pt', dtype='sequence')
#my_predictor_5 = ppp.Predictor('./parrot_models/network_final_pred_model5.pt', dtype='sequence')

# mean/max の cbf, max_chain_length, chain_strength（指定していればその値、指定していなければ実際にサンプルに使った値）, best energy（= res.first.energy）, best energy のサンプルの cbf をログ
def _get_max_chain_length(embedding):
    # embedding: dict[logical] -> list[physical]
    if not embedding:
        return 0
    return max(len(chain) for chain in embedding.values())

def _max_chain_length_from_anywhere(res, sampler):
    # 1) sampler.child.embedding を試す
    emb = getattr(getattr(sampler, "child", None), "embedding", None)
    if emb:
        return int(_get_max_chain_length(emb))

    # 2) res.info["embedding_context"]["embedding"] を試す（よくある）
    info = getattr(res, "info", {}) or {}
    ctx = info.get("embedding_context")
    if isinstance(ctx, dict):
        emb2 = ctx.get("embedding")
        if emb2:
            return int(_get_max_chain_length(emb2))

    # 取れない
    return 0

def _safe_get_chain_strength_from_info(res):
    """
    chain_strength を明示指定しない場合に、実際に使われた値が res.info に入ることがあります。
    dwave-system のバージョン差があるので安全に取得。
    """
    info = getattr(res, "info", {}) or {}
    # よくある候補キー（環境で異なる可能性あり）
    for key in ("chain_strength", "embedding_context", "chain_strengths"):
        if key in info:
            val = info[key]
            # embedding_context の中に入っているケース
            if isinstance(val, dict) and "chain_strength" in val:
                return val["chain_strength"]
            return val
    return None

def append_qpu_log_csv(csv_path, iter_idx, res, sampler, bqm, requested_chain_strength=None):
    """
    res: SampleSet (EmbeddingComposite の戻り値)
    sampler: EmbeddingComposite インスタンス
    bqm: dimod.BinaryQuadraticModel（このiterでサンプルした論理BQM）
    requested_chain_strength: sample() に渡した chain_strength（渡していなければ None）
    """
    
    # ---- normalized energy (E_norm) + coefficient scales ----
    offset = float(getattr(bqm, "offset", 0.0))
    max_abs_h = float(max((abs(v) for v in bqm.linear.values()), default=0.0))
    max_abs_J = float(max((abs(v) for v in bqm.quadratic.values()), default=0.0))
    scale = max(max_abs_h, max_abs_J)
    num_vars = int(len(bqm.variables))
    denom = num_vars * scale
    
    # per-read energy (includes offset)
    E = res.record.energy.astype(float)
    # cbf per read
    cbf_reads = res.record["chain_break_fraction"].astype(float)
    
    # best (robust)
    best_i = int(np.argmin(res.record.energy))
    best_energy = float(E[best_i])
    cbf_best = float(cbf_reads[best_i])

    # normalized energy (E_norm)
    E0 = best_energy - offset
    E_norm = float("nan") if denom == 0.0 else (E0 / denom)

    # cbf stats
    cbf_mean = float(cbf_reads.mean())
    cbf_max = float(cbf_reads.max())

    # embedding / max chain length
    max_chain_length = _max_chain_length_from_anywhere(res, sampler)

    # chain_strength
    used_chain_strength = requested_chain_strength
    if used_chain_strength is None:
        used_chain_strength = _safe_get_chain_strength_from_info(res)

    # 1行追記（ヘッダなければ作る）
    need_header = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if need_header:
            f.write("iter,best_energy,E_norm,offset,max_abs_h,max_abs_J,cbf_mean,cbf_max,cbf_best,max_chain_length,chain_strength\n")
        f.write(f"{iter_idx},{best_energy:.16g},{E_norm:.16g},{offset:.16g},{max_abs_h:.16g},{max_abs_J:.16g},{cbf_mean:.16g},{cbf_max:.16g},{cbf_best:.16g},{max_chain_length},{'' if used_chain_strength is None else used_chain_strength}\n")

# Retry minor embedding
def sample_with_more_embedding_effort(bqm, token, num_reads, annealing_time, chain_strength, solver, endpoint, max_retries):
    last = None
    for k in range(1, max_retries + 1):
        try:
            sampler = EmbeddingComposite(
                DWaveSampler(endpoint=endpoint, token=token, solver=solver),
                #embedding_parameters={
                    # minorminer側の探索努力を増やす（環境により効く/無視されることがあります）
                #   "tries": 200,
                #   "timeout": 5000,
                #},
            )
            return sampler.sample(bqm, num_reads=num_reads, annealing_time=annealing_time, chain_strength=chain_strength, chain_break_fraction=True)
        except ValueError as e:
            # 典型: "no embedding found"
            last = e
            print(f"[attempt {k}/{max_retries}] embedding failed: {e}")
            time.sleep(0.2)
    raise last

# Pareto front identification
def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
            is_efficient[i] = True
    return is_efficient

# Elimination of sequences with non-AA characters
def seq_fix():
    lines_d = open('./model_output/binary/decoded.txt','r')
    Lines_d = lines_d.readlines()
    seq_d=[]
    for i in Lines_d:
        seq_d.append(i.replace(" ", "").rstrip())

    matching_d1 = [s for s in seq_d if "<pad>" in s]
    matching_d2 = [s for s in seq_d if len(s) == 0]
    matching_d3 = [s for s in seq_d if "<unk>" in s]

    for ii in range(len(matching_d1)):
        for i in range(len(seq_d)):
            if seq_d[i] == matching_d1[ii]:
                seq_d[i] = "SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS"

    for ii in range(len(matching_d2)):
        for i in range(len(seq_d)):
            if seq_d[i] == matching_d2[ii]:
                seq_d[i] = "SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS"

    for ii in range(len(matching_d3)):
        for i in range(len(seq_d)):
            if seq_d[i] == matching_d3[ii]:
                seq_d[i] = "SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS"

    return seq_d

# Objective function evaluation
def objectives(seq_d):

    # VHH sequence containing repetitive amino acics identification
    rep_AA_cutoff = 7

    counts = []
    for i in range(len(seq_d)):
        grouped_L = []
        grouped_L = [(k, sum(1 for i in g)) for k, g in groupby(seq_d[i])]
        counts.append(grouped_L)

    score_AA = []
    for i in counts:
        rep_count = []
        for j in range(len(i)):
            rep_count.append(i[j][1])

        rep_count_binary = []
        for i in rep_count:
            if i >= rep_AA_cutoff:
                score = 0
            else:
                score = 1
            rep_count_binary.append(score)

        score_AA.append(reduce(lambda x, y: x*y, rep_count_binary))

    # CDRs Extraction via ANARCI
    with open("./model_output/binary/seq_VHH.fasta", "w") as oo:
        for i in range(len(seq_d)):
            oo.write(">" + str(i) + "\n")
            oo.write(str(seq_d[i]) + "\n")

    # --- ANARCI (subprocess) ---
    out_prefix = "./model_output/binary/HER2_naive_3rd_cdr_out"
    csv_path = out_prefix + "_H.csv"

    # Avoid reading stale output from previous runs
    if os.path.exists(csv_path):
        os.remove(csv_path)

    anarci_cmd = [
        "ANARCI",
        "--sequence", "./model_output/binary/seq_VHH.fasta",
        "--outfile", out_prefix,
        "--scheme", "kabat",
        "--restrict", "heavy",
        "--csv",
    ]
    proc = subprocess.run(anarci_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("[ANARCI] Command failed.")
        print("[ANARCI] returncode:", proc.returncode)
        print("[ANARCI] stdout:\n", proc.stdout)
        print("[ANARCI] stderr:\n", proc.stderr)
        raise RuntimeError("ANARCI failed; see logs above.")

    if not os.path.exists(csv_path):
        # ANARCI returned 0 but did not produce the expected file -> print diagnostics
        print("[ANARCI] Completed but expected CSV not found:", csv_path)
        print("[ANARCI] stdout:\n", proc.stdout)
        print("[ANARCI] stderr:\n", proc.stderr)
        try:
            print("[ANARCI] Listing ./model_output/binary for related files:")
            for fn in sorted(os.listdir("./model_output/binary")):
                if "HER2_naive_3rd_cdr_out" in fn:
                    print("  -", fn)
        except Exception as e:
            print("[ANARCI] Could not list output directory:", e)
        raise FileNotFoundError(csv_path)

    df_num = pd.read_csv(csv_path)

    # Ensure Id is int so we can align with fasta headers (>0, >1, ...)
    if "Id" not in df_num.columns:
        raise RuntimeError(f"[ANARCI] 'Id' column not found in {csv_path}. Columns={list(df_num.columns)}")

    df_num["Id"] = df_num["Id"].astype(int)

    # Build score_del per Id (default 0 for missing Ids)
    # Original intent:
    #   if (110-113 all '-') OR (113 == '-') => score_del = 0 else 1
    # We'll compute it only for rows that exist in df_num, then align by Id.
    score_del_by_id = {}
    for _, row in df_num.iterrows():
        i_id = int(row["Id"])
        v110 = row.get("110", "-")
        v111 = row.get("111", "-")
        v112 = row.get("112", "-")
        v113 = row.get("113", "-")
        if ((v110 == "-") and (v111 == "-") and (v112 == "-") and (v113 == "-")) or (v113 == "-"):
            score_del_by_id[i_id] = 0
        else:
            score_del_by_id[i_id] = 1

    # df_num の列のうち、文字が"26"-"35"から始まる列名を抽出
    cols_26_35 = [col for col in df_num.columns if str(col).startswith(("26", "27", "28", "29", "30", "31", "32", "33", "34", "35"))]
    # df_num の列のうち、文字が"50"-"58"から始まる列名を抽出
    cols_50_58 = [col for col in df_num.columns if str(col).startswith(("50", "51", "52", "53", "54", "55", "56", "57", "58"))]
    # df_num の列のうち、文字が"95"-"102"から始まる列名を抽出
    cols_95_102 = [col for col in df_num.columns if str(col).startswith(("95", "96", "97", "98", "99", "100", "101", "102"))]
    # CDR の抽出
    cdr1_df = df_num[cols_26_35].agg(''.join, axis=1).str.replace("-", "")
    cdr2_df = df_num[cols_50_58].agg(''.join, axis=1).str.replace("-", "")
    cdr3_df = df_num[cols_95_102].agg(''.join, axis=1).str.replace("-", "")
    # cdr 列を連結
    cdr_series = cdr1_df + cdr2_df + cdr3_df

    # Map: Id -> cdr string (only for rows that exist)
    cdr_by_id = dict(zip(df_num["Id"].tolist(), cdr_series.tolist()))

    # Align to full seq_d length by Id=0..len(seq_d)-1
    all_ids = list(range(len(seq_d)))
    missing_ids = sorted(set(all_ids) - set(cdr_by_id.keys()))
    if missing_ids:
        print(f"[ANARCI] {len(missing_ids)} sequences did not pass ANARCI heavy-chain numbering. Missing Ids: {missing_ids}", flush=True)

    # Create aligned arrays:
    # - cdr_df_aligned: list[str|None] length = len(seq_d)
    # - score_del_aligned: list[int] length = len(seq_d), default 0 if missing
    cdr_df_aligned = []
    score_del = []
    for i_id in all_ids:
        cdr_df_aligned.append(cdr_by_id.get(i_id, None))
        score_del.append(int(score_del_by_id.get(i_id, 0)))


    # Sequences with above constraints violated will be penalized by assigning them lowest possible values
    pred_score_ave_ref = -2.683010178959062 # Phage display の下位0.1%
    solubility_ref = 0.

    # Sequence evaluation via prediction model ensemble
    pred_model_1 = [pred_score_ave_ref] * len(seq_d)
    pred_model_2 = [pred_score_ave_ref] * len(seq_d)
    pred_model_3 = [pred_score_ave_ref] * len(seq_d)

    # Only run predictors for sequences that have a CDR string (i.e., passed ANARCI)
    for i in range(len(seq_d)):
        cdr = cdr_df_aligned[i]
        if (cdr is None) or (len(cdr) == 0):
            continue
        pred_model_1[i] = my_predictor_1.predict(cdr)[0]
        pred_model_2[i] = my_predictor_2.predict(cdr)[0]
        pred_model_3[i] = my_predictor_3.predict(cdr)[0]


    pred_score_ave = []
    for i in range(len(seq_d)):
        # penalize if repetitive AA constraint violated OR ANARCI deletion constraint failed OR ANARCI missing
        if score_AA[i] == 0 or score_del[i] == 0 or cdr_df_aligned[i] is None or len(cdr_df_aligned[i]) == 0:
            pred_score_ave.append(pred_score_ave_ref)
        else:
            pred_score_ave.append(np.mean([pred_model_1[i], pred_model_2[i], pred_model_3[i]]))

    os.system('rm -f ./model_output/binary/HER2_naive_3rd_cdr_out_H.csv')
    # Solubility via NetSolP
    os.system('python ../netsolp-1.0.ALL/predict.py --MODELS_PATH ../netsolp-1.0.ALL/models --FASTA_PATH ./model_output/binary/seq_VHH.fasta --OUTPUT_PATH ./model_output/binary/seq_VHH_sol_preds.csv --MODEL_TYPE Distilled --PREDICTION_TYPE S')
    solubility = pd.read_csv('./model_output/binary/seq_VHH_sol_preds.csv', usecols = ['predicted_solubility'], low_memory = True)

    solubility_temp = []
    for i in solubility.to_numpy():
        solubility_temp.append(i[0])

    solubility = []
    for i in range(len(seq_d)):
        if score_AA[i] == 0 or score_del[i] == 0:
            solubility_c = solubility_ref
        else:
            solubility_c = solubility_temp[i]
        solubility.append(solubility_c)

    os.system('rm -f ./model_output/binary/seq_VHH_sol_preds.csv')
    os.system('rm -f ./model_output/binary/seq_VHH.fasta')

    return np.array(pred_score_ave), np.array(solubility)

# Hypervolume calculation
def ParetoV(objective_values):

    front_objective_values_0 = is_pareto_efficient(objective_values)

    front_objective_values = []
    front_index_objective_values_original = []
    for i in range(len(front_objective_values_0)):
        if front_objective_values_0[i] == True:
            front_objective_values.append(objective_values[i])
            front_index_objective_values_original.append(i)

    temp0 = []
    for i in front_objective_values:
        temp0.append(i)

    x0 = torch.from_numpy(np.array(temp0))

    pred_score_ave_ref = -2.683010178959062 # Phage display の下位0.1%
    solubility_ref = 0.

    xref = torch.from_numpy(np.array([pred_score_ave_ref, solubility_ref]))
    hv = Hypervolume(xref)
    pv0 = hv.compute(x0)

    with open('./model_output/binary/hv_score.txt', 'a') as f:
        f.write(str(pv0) + "\n")

# Non-dominated sorting procedure with pre-defined number of layers
def NSPareto(objective_values, objective_values_original, population_size, NLayers):

    SLayers = np.ones(population_size) * (-1) * 10

    for ii in range(NLayers):

        front_objective_values_0 = is_pareto_efficient(objective_values)

        front_objective_values = []
        front_index_objective_values = []
        for i in range(len(front_objective_values_0)):
            if front_objective_values_0[i] == True:
                front_objective_values.append(objective_values[i])
                front_index_objective_values.append(i)

        index_out = []
        for i in front_objective_values:
            index = np.where(i == objective_values_original)
            index_out.append(index[0][0])

        for i in index_out:
            SLayers[i] = 1. / (ii + 1.)

        objective_values = np.delete(objective_values, front_index_objective_values, axis=0)

    return SLayers


# Decode initial binary vector set
os.system('CUDA_VISIBLE_DEVICES="" python decode_file.py --hparams_path experiment_configs/binary.yml \
        --checkpoint_kind last \
        --decode_strategy greedy')

c0 = [x.split(' ')[:] for x in open('./model_output/binary/vectors.txt').readlines()]

vectors_all = []
for k in range(len(c0)):
    vectors = []
    for i in c0[k]:
        vectors.append(int(float(i)))
    vectors_all.append(vectors)

vectors_all = np.array(vectors_all)

# Get rid of wrong sequences
seq_d_all = seq_fix()

# Calculate objective functions
scores_all = objectives(seq_d_all)
scores_all_pred_score_ave = scores_all[0]
scores_all_solubility = scores_all[1]

# Output best metrics in the initial set
with open('./model_output/binary/all_points_best.txt', 'w') as oo:
    oo.write(f"{0} {np.max(scores_all_pred_score_ave)} {np.max(scores_all_solubility)}\n")

objective_values = []
for i in range(len(scores_all_pred_score_ave)):
    objective_values.append([scores_all_pred_score_ave[i], scores_all_solubility[i]])

objective_values_original = np.array(objective_values)
objective_values = objective_values_original

NLayers = 20  # number of non-dominated layers to consider

population_size = len(scores_all_pred_score_ave)

# Calculate initial Pareto hypervolume
os.system('rm -f ./model_output/binary/hv_score.txt')
ParetoV(objective_values)

# Non-dominated sorting for the initial population
SLayers = NSPareto(objective_values, objective_values_original, population_size, NLayers)
scores = (-1.)*SLayers

# FM training
model = fmqa.FMBQM.from_data(vectors_all, scores)

## Simulated annealing for sampling
#sampler = dimod.samplers.SimulatedAnnealingSampler()

# if you use D-wave, replace above "sampler" with the following code.
bqm = dimod.BinaryQuadraticModel(model)
sampler = EmbeddingComposite(DWaveSampler(endpoint='https://cloud.dwavesys.com/sapi', token='XXXX', solver='Advantage_system4.1'))
CHAIN_STRENGTH = 1

## For performance assessment use, replace above "sampler" with the following code.
# sampler = dimod.samplers.RandomSampler() # Fast

# decoded_list and vectors_list files initialization
os.system('rm -f ./model_output/binary/vectors_list.txt')
os.system('rm -f ./model_output/binary/decoded_list.txt')
os.system('rm -f ./model_output/binary/all_points_samples.txt')
os.system('rm -f ./model_output/binary/qpu_diag_log.csv')
# Sampling/Evaluation/Training loop
for iter in range(400):

    ## Simulated Annealing
    #res = sampler.sample(model, num_reads=10)

    # if you use D-wave, replace above "res" with the following code.
    try:
        res = sampler.sample(bqm, annealing_time=20, num_reads=25, chain_strength=CHAIN_STRENGTH, chain_break_fraction=True)
    except ValueError as e:
        print("[DWave embedding] primary sampler failed, retrying with more effort:", e, flush=True)
        res = sample_with_more_embedding_effort(bqm, endpoint='https://cloud.dwavesys.com/sapi', token='XXXX', solver='Advantage_system4.1',
                annealing_time=20, num_reads=25, chain_strength=CHAIN_STRENGTH, max_retries=20)
    # log
    append_qpu_log_csv("./model_output/binary/qpu_diag_log.csv", iter+1, res, sampler, bqm, CHAIN_STRENGTH)

    # SQ / QA
    vectors_all = np.r_[vectors_all, res.record['sample']]
    vectors_sample = np.r_[res.record['sample']]

    vectors_sample_out = []
    for row in vectors_sample:
        vectors_sample_out.append(" ".join(str(int(v)) for v in row))

    # write vectors.txt for decode_file.py
    with open("./model_output/binary/vectors.txt", "w") as f:
        for s in vectors_sample_out:
            f.write(s + "\n")

    # recode sampled binary vectors in text format
    with open("./model_output/binary/vectors_list.txt", "a") as f:
        for s in vectors_sample_out:
            f.write(s + "\n")

    # decode sampled vectors
    with open("./model_output/binary/decoded.txt", "w"):
        os.system('CUDA_VISIBLE_DEVICES="" python decode_file.py --hparams_path experiment_configs/binary.yml \
        --checkpoint_kind last \
        --decode_strategy top_p \
        --top_p 0.95 \
        --temperature 1.8')

    # recode sampled sequences in text format
    with open("./model_output/binary/decoded_list.txt", "a") as f_out:
        for line in open("./model_output/binary/decoded.txt").readlines():
            f_out.write(line)

    seq_d = seq_fix()

    seq_d_all = np.r_[seq_d_all, seq_d]

    # Calculate objective functions for sampled sequences
    scores_sample = objectives(seq_d)

    scores_sample_pred_score_ave = scores_sample[0]
    scores_all_pred_score_ave = np.r_[scores_all_pred_score_ave, scores_sample_pred_score_ave]

    scores_sample_solubility = scores_sample[1]
    scores_all_solubility = np.r_[scores_all_solubility, scores_sample_solubility]

    objective_values = []
    for i in range(len(scores_all_pred_score_ave)):
        objective_values.append([scores_all_pred_score_ave[i], scores_all_solubility[i]])

    objective_values_original = np.array(objective_values)
    objective_values = objective_values_original

    # Calculate Pareto hypervolume for the updated population
    ParetoV(objective_values)

    # Non-dominated sorting procedure with gradual layer number reduction
    STEP_GEOM = 0.982
    ii = iter
    if ii == 0:
        NLayerss = 20.
    else:
        NLayerss = STEP_GEOM * NLayerss

    NLayers = round(NLayerss)
    population_size = len(scores_all_pred_score_ave)

    SLayers = NSPareto(objective_values, objective_values_original, population_size, NLayers)

    scores_all = (-1.)*SLayers

    # Output metrics of samples for the updated population
    offset = float(getattr(bqm, "offset", 0.0))
    max_abs_h = float(max((abs(v) for v in bqm.linear.values()), default=0.0))
    max_abs_J = float(max((abs(v) for v in bqm.quadratic.values()), default=0.0))
    scale = max(max_abs_h, max_abs_J)
    num_vars = int(len(bqm.variables))
    denom = num_vars * scale

    # per-read energy (includes offset)
    E = res.record.energy.astype(float)
    # cbf per read
    cbf_reads = res.record["chain_break_fraction"].astype(float)

    # pick best by minimum record energy (robust)
    best_i = int(np.argmin(E))
    best_energy = float(E[best_i])
    cbf_best = float(cbf_reads[best_i])
    
    E0_best = best_energy - offset
    E_best_norm = float("nan") if denom == 0.0 else (E0_best / denom)

    # Output best metrics for the updated population
    with open('./model_output/binary/all_points_best.txt', 'a') as oo:
        oo.write(f"{iter+1} {np.max(scores_all_pred_score_ave)} {np.max(scores_all_solubility)} {E_best_norm:.16g} {cbf_best:.16g}\n")
    
    # per-read normalized energy)
    E0 = E - offset
    E_norm_reads = np.full_like(E0, np.nan, dtype=float) if denom == 0.0 else (E0 / denom)
        
    # Output metrics of samples for the updated population
    with open('./model_output/binary/all_points_samples.txt', 'a') as oo:
        for i in range(len(scores_sample_pred_score_ave)):
            # columns: pred_score_ave solubility E_norm cbf
            oo.write(f"{scores_sample_pred_score_ave[i]} {scores_sample_solubility[i]} {E_norm_reads[i]:.16g} {cbf_reads[i]:.16g}\n")

    # FM re-training
    model.train(vectors_all, scores_all)

    # if you use D-wave, add the following code
    bqm = dimod.BinaryQuadraticModel(model)

# reproduce initial vectors file
#os.system('cp vectors.txt_BK model_output/binary/vectors.txt')
