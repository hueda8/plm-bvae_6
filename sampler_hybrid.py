import numpy as np
import fmqa
import dimod
import os
import subprocess
from dwave.system import LeapHybridSampler
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from itertools import groupby
from functools import reduce
from parrot import py_predictor as ppp
import pandas as pd
import random

# Loading regression models
my_predictor_1 = ppp.Predictor('./parrot_models/full_log_deldup_weight1_opt_network.pt', dtype='sequence')
my_predictor_2 = ppp.Predictor('./parrot_models/full_log_deldup_weight1_opt_2nd_network.pt', dtype='sequence')
my_predictor_3 = ppp.Predictor('./parrot_models/full_log_deldup_weight1_opt_3rd_network.pt', dtype='sequence')
#my_predictor_4 = ppp.Predictor('./parrot_models/network_final_pred_model4.pt', dtype='sequence')
#my_predictor_5 = ppp.Predictor('./parrot_models/network_final_pred_model5.pt', dtype='sequence')

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

def sampleset_to_fixed_k_numpy(sampleset, bqm, k: int, pad: str, alpha: float, candidate_pool: int,
) -> np.ndarray:
    """
    Convert dimod.SampleSet -> (k, n_vars) numpy array with variables ordered as bqm.variables.

    Select k samples using an energy-diversity tradeoff (Hamming distance).
    - Start with the lowest-energy unique sample.
    - Iteratively add samples that maximize:
        score = alpha * energy_score + (1-alpha) * diversity_score
      where:
        energy_score is higher for lower energy (normalized),
        diversity_score is higher for larger minimum Hamming distance to selected set (normalized).

    pad:
      - "repeat_best": if <k samples are available, repeat the best sample to reach k
      - "error": raise if <k samples

    alpha:
      - 1.0 => energy only
      - 0.0 => diversity only

    candidate_pool:
      - consider only the best (lowest-energy) N unique samples as candidates for diversity selection
        to control runtime.
    """
    vars_order = list(bqm.variables)

    # Collect and sort by energy
    rows = list(sampleset.data(fields=["sample", "energy"]))
    rows.sort(key=lambda r: float(r.energy))

    # Deduplicate by bit pattern in fixed variable order
    cand_bits = []
    cand_energy = []
    seen = set()
    for r in rows:
        s = r.sample
        bits = tuple(int(s[v]) for v in vars_order)
        if bits in seen:
            continue
        seen.add(bits)
        cand_bits.append(bits)
        cand_energy.append(float(r.energy))
        if len(cand_bits) >= candidate_pool:
            break

    n_cand = len(cand_bits)
    n_vars = len(vars_order)

    if n_cand == 0:
        raise RuntimeError("Sampler returned 0 samples; cannot select.")

    # If we already have <=k candidates, just take by energy and pad if needed
    if n_cand <= k:
        out = np.zeros((k, n_vars), dtype=int)
        take = min(n_cand, k)
        for i in range(take):
            out[i, :] = np.asarray(cand_bits[i], dtype=int)
        if take < k:
            if pad == "error":
                raise RuntimeError(f"Only {take} unique samples available; need {k}.")
            best = np.asarray(cand_bits[0], dtype=int)
            for i in range(take, k):
                out[i, :] = best
        return out

    # Normalize energies so that lower energy => higher score
    e_min = min(cand_energy)
    e_max = max(cand_energy)
    if e_max == e_min:
        energy_score = [1.0] * n_cand
    else:
        # map energy to [0,1], where 1 is best (lowest energy)
        energy_score = [(e_max - e) / (e_max - e_min) for e in cand_energy]

    # Helper: Hamming distance between two bit tuples
    def hamming(a, b) -> int:
        # assumes same length
        return sum(x != y for x, y in zip(a, b))

    # Greedy selection with energy+diversity tradeoff
    selected_idx = [0]  # start from best energy
    remaining = set(range(1, n_cand))

    # Precompute distances on the fly (cheap enough for candidate_pool~200, n_vars~448)
    while len(selected_idx) < k and remaining:
        best_j = None
        best_score = None

        # Compute diversity score as min Hamming distance to already selected
        for j in list(remaining):
            min_d = min(hamming(cand_bits[j], cand_bits[i]) for i in selected_idx)
            # normalize min_d to [0,1]
            div = min_d / float(n_vars) if n_vars > 0 else 0.0
            score = alpha * energy_score[j] + (1.0 - alpha) * div

            if (best_score is None) or (score > best_score):
                best_score = score
                best_j = j

        if best_j is None:
            break
        selected_idx.append(best_j)
        remaining.remove(best_j)

    # Build output
    out = np.zeros((k, n_vars), dtype=int)
    take = min(len(selected_idx), k)
    for i in range(take):
        out[i, :] = np.asarray(cand_bits[selected_idx[i]], dtype=int)

    # Pad if needed
    if take < k:
        if pad == "error":
            raise RuntimeError(f"Could select only {take} samples; need {k}. Increase candidate_pool/time_limit.")
        best = np.asarray(cand_bits[selected_idx[0]], dtype=int)
        for i in range(take, k):
            out[i, :] = best
    return out

def dropout_weak_quadratic_edges(bqm: dimod.BinaryQuadraticModel, drop_prob: float, weak_quantile: float, seed: int = None,
) -> dimod.BinaryQuadraticModel:
    """
    Dropout only 'weak' quadratic terms (small |J|).
    - Define weak edges as those with |J| <= quantile(|J|, weak_quantile).
    - For weak edges, drop with probability drop_prob.
    - Strong edges are always kept.
    """
    rng = random.Random(seed)

    # compute threshold
    absJ = np.array([abs(w) for w in bqm.quadratic.values()], dtype=float)
    if absJ.size == 0:
        return bqm.copy()
    thr = float(np.quantile(absJ, weak_quantile))

    sbqm = dimod.BinaryQuadraticModel({}, {}, bqm.offset, vartype=bqm.vartype)
    sbqm.add_linear_from(bqm.linear)

    for (u, v), w in bqm.quadratic.items():
        if abs(w) <= thr:
            # weak edge: drop with probability drop_prob
            if rng.random() >= drop_prob:
                sbqm.add_interaction(u, v, w)
        else:
            # strong edge: always keep
            sbqm.add_interaction(u, v, w)
    return sbqm

def rescore_sampleset_to_reference_bqm(
    ss: dimod.SampleSet,
    ref_bqm: dimod.BinaryQuadraticModel
) -> dimod.SampleSet:
    """
    Recompute energies of samples in ss using ref_bqm, replacing the energy field.
    Compatible with older dimod (no dimod.energies()).
    """
    # ss.record.sample is a numpy structured array field: shape (n_samples, n_vars)
    # But its variable order is ss.variables
    vars_order = list(ss.variables)

    new_energy = np.empty(ss.record.shape[0], dtype=float)
    for i, sample_row in enumerate(ss.record.sample):
        # sample_row is aligned to ss.variables
        sample = {v: int(sample_row[j]) for j, v in enumerate(vars_order)}
        new_energy[i] = float(ref_bqm.energy(sample))

    return dimod.SampleSet.from_samples(
        ss,  # keep samples/vartype from existing sampleset
        vartype=ss.vartype,
        energy=new_energy,
        info=ss.info
    )

def sample_diverse_by_restarts_dropout_rescored(sampler, ref_bqm: dimod.BinaryQuadraticModel, total_time: int, restarts: int, drop_prob: float, weak_quantile: float, base_seed: int = 0):
    per_time = max(3, total_time // restarts)
    all_samples = []
    for r in range(restarts):
        # 1) make dropped BQM
        bqm_dp = dropout_weak_quadratic_edges(ref_bqm, drop_prob=drop_prob, weak_quantile=weak_quantile, seed=base_seed + r)

        # 2) solve dropped BQM
        ss = sampler.sample(bqm_dp, time_limit=per_time)

        # 3) re-score on the original (full) BQM
        ss_ref = rescore_sampleset_to_reference_bqm(ss, ref_bqm)

        all_samples.append(ss_ref)

    return dimod.concatenate(all_samples)


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
    oo = open('./model_output/binary/seq_VHH.fasta','w')
    for i in range(len(seq_d)):
        oo.write('>' + str(i) + '\n')
        oo.write(str(seq_d[i]) + '\n')
    oo.close()

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
        print(f"[ANARCI] {len(missing_ids)} sequences did not pass ANARCI heavy-chain numbering. Missing Ids: {missing_ids}")

    # Create aligned arrays:
    # - cdr_df_aligned: list[str|None] length = len(seq_d)
    # - score_del_aligned: list[int] length = len(seq_d), default 0 if missing
    cdr_df_aligned = []
    score_del = []
    for i_id in all_ids:
        cdr_df_aligned.append(cdr_by_id.get(i_id, None))
        score_del.append(int(score_del_by_id.get(i_id, 0)))


    # Sequences with above constraints violated will be penalized by assigning them lowest possible values
    pred_score_1_ref = 0.
    pred_score_2_ref = 0.
    pred_score_3_ref = 0.
    solubility_ref = 0.

    # Sequence evaluation via prediction model ensemble
    pred_model_1 = [pred_score_1_ref] * len(seq_d)
    pred_model_2 = [pred_score_2_ref] * len(seq_d)
    pred_model_3 = [pred_score_3_ref] * len(seq_d)

    # Only run predictors for sequences that have a CDR string (i.e., passed ANARCI)
    for i in range(len(seq_d)):
        cdr = cdr_df_aligned[i]
        if (cdr is None) or (len(cdr) == 0):
            continue
        pred_model_1[i] = np.exp(my_predictor_1.predict(cdr)[0])
        pred_model_2[i] = np.exp(my_predictor_2.predict(cdr)[0])
        pred_model_3[i] = np.exp(my_predictor_3.predict(cdr)[0])


    pred_score_1 = []
    pred_score_2 = []
    pred_score_3 = []
    for i in range(len(seq_d)):
        # penalize if repetitive AA constraint violated OR ANARCI deletion constraint failed OR ANARCI missing
        if score_AA[i] == 0 or score_del[i] == 0 or cdr_df_aligned[i] is None or len(cdr_df_aligned[i]) == 0:
            pred_score_1.append(pred_score_1_ref)
            pred_score_2.append(pred_score_2_ref)
            pred_score_3.append(pred_score_3_ref)
        else:
            pred_score_1.append(pred_model_1[i])#, pred_model_3[i], pred_model_4[i], pred_model_5[i]]))
            pred_score_2.append(pred_model_2[i])#, pred_model_3[i], pred_model_4[i], pred_model_5[i]]))
            pred_score_3.append(pred_model_3[i])#, pred_model_3[i], pred_model_4[i], pred_model_5[i]]))

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

    return np.array(pred_score_1), np.array(pred_score_2), np.array(pred_score_3), np.array(solubility)

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

    pred_score_1_ref = 0.
    pred_score_2_ref = 0.
    pred_score_3_ref = 0.
    solubility_ref = 0.

    xref = torch.from_numpy(np.array([pred_score_1_ref, pred_score_2_ref, pred_score_3_ref, solubility_ref]))
    hv = Hypervolume(xref)
    pv0 = hv.compute(x0)

    oo = open('./model_output/binary/hv_score.txt', 'a')
    oo.write(str(pv0) + '\n')
    oo.close()

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
os.system('CUDA_VISIBLE_DEVICES="" python decode_file.py --hparams_path experiment_configs/binary.yml --checkpoint_kind last')

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
scores_all_pred_score_1 = scores_all[0]
scores_all_pred_score_2 = scores_all[1]
scores_all_pred_score_3 = scores_all[2]
scores_all_solubility = scores_all[3]

# Output best metrics in the initial set
oo = open('./model_output/binary/all_points_best.txt','w')
oo.write(str(np.max(scores_all_pred_score_1)) + ' ' + str(np.max(scores_all_pred_score_2)) + ' ' + str(np.max(scores_all_pred_score_3)) + ' ' + str(np.max(scores_all_solubility)) + '\n')
oo.close()

objective_values = []
for i in range(len(scores_all_pred_score_1)):
    objective_values.append([scores_all_pred_score_1[i], scores_all_pred_score_2[i], scores_all_pred_score_3[i], scores_all_solubility[i]])

objective_values_original = np.array(objective_values)
objective_values = objective_values_original

NLayers = 20  # number of non-dominated layers to consider

population_size = len(scores_all_pred_score_1)

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
sampler = LeapHybridSampler(endpoint='https://cloud.dwavesys.com/sapi', token='XXXX', solver='hybrid_binary_quadratic_model_version2p')

## For performance assessment use, replace above "sampler" with the following code.
# sampler = dimod.samplers.RandomSampler() # Fast

# Sampling/Evaluation/Training loop
# decoded_list and vectors_list files initialization
os.system('rm -f ./model_output/binary/vectors_list.txt')
os.system('rm -f ./model_output/binary/decoded_list.txt')
os.system('rm -f ./model_output/binary/all_points_samples.txt')
for iter in range(200):

    ## Simulated Annealing
    #res = sampler.sample(model, num_reads=10)

    # hybrid_binary_quadratic_model_version2p
    res = sample_diverse_by_restarts_dropout_rescored(sampler, bqm, total_time=300, restarts=100, drop_prob=0.1, weak_quantile = 0.3, base_seed=iter*1000)
    vectors_sample = sampleset_to_fixed_k_numpy(res, bqm, k=10, alpha=0.8, pad="repeat_best", candidate_pool=100)

    vectors_all = np.r_[vectors_all, vectors_sample]

    vectors_sample_out = []
    for row in vectors_sample:
        vectors_sample_out.append(" ".join(str(int(v)) for v in row))

    os.system('rm -f ./model_output/binary/vectors.txt')
    oo = open('./model_output/binary/vectors.txt','a')
    for i in vectors_sample_out:
        oo.write(str(i) + '\n')
    oo.close()

    # recode sampled binary vectors in text format
    vectors_list = open('./model_output/binary/vectors_list.txt','a')
    for i in vectors_sample_out:
        vectors_list.write(str(i) + '\n')
    vectors_list.close()

    os.system('rm -f ./model_output/binary/decoded.txt')
    os.system('CUDA_VISIBLE_DEVICES="" python ./decode_file.py --hparams_path experiment_configs/binary.yml --checkpoint_kind last')

    # recode sampled sequences in text format
    decoded_list = open('./model_output/binary/decoded_list.txt','a')
    for i in open('./model_output/binary/decoded.txt').readlines():
        decoded_list.write(i)
    decoded_list.close()

    seq_d = seq_fix()

    seq_d_all = np.r_[seq_d_all, seq_d]

    # Calculate objective functions for sampled sequences
    scores_sample = objectives(seq_d)

    scores_sample_pred_score_1 = scores_sample[0]
    scores_all_pred_score_1 = np.r_[scores_all_pred_score_1, scores_sample_pred_score_1]

    scores_sample_pred_score_2 = scores_sample[1]
    scores_all_pred_score_2 = np.r_[scores_all_pred_score_2, scores_sample_pred_score_2]

    scores_sample_pred_score_3 = scores_sample[2]
    scores_all_pred_score_3 = np.r_[scores_all_pred_score_3, scores_sample_pred_score_3]

    scores_sample_solubility = scores_sample[3]
    scores_all_solubility = np.r_[scores_all_solubility, scores_sample_solubility]

    objective_values = []
    for i in range(len(scores_all_pred_score_1)):
        objective_values.append([scores_all_pred_score_1[i], scores_all_pred_score_2[i], scores_all_pred_score_3[i], scores_all_solubility[i]])

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
    population_size = len(scores_all_pred_score_1)

    SLayers = NSPareto(objective_values, objective_values_original, population_size, NLayers)

    scores_all = (-1.)*SLayers

    # Output best metrics for the updated population
    oo = open('./model_output/binary/all_points_best.txt','a')
    oo.write(str(iter+1) + ' ' + str(np.max(scores_all_pred_score_1)) + ' ' + str(np.max(scores_all_pred_score_2)) + ' ' + str(np.max(scores_all_pred_score_3)) + ' ' + str(np.max(scores_all_solubility)) + '\n')
    oo.close()

    # Output metrics of samples for the updated population
    oo = open('./model_output/binary/all_points_samples.txt','a')
    for i in range(len(scores_sample_pred_score_1)):
        oo.write(str(scores_sample_pred_score_1[i]) + ' ' + str(scores_sample_pred_score_2[i]) + ' ' + str(scores_sample_pred_score_3[i]) + ' ' + str(scores_sample_solubility[i]) + '\n')
    oo.close()

    # FM re-training
    model.train(vectors_all, scores_all)

    # if you use D-wave, add the following code
    bqm = dimod.BinaryQuadraticModel(model)

# reproduce initial vectors file
#os.system('cp vectors.txt_BK model_output/binary/vectors.txt')
