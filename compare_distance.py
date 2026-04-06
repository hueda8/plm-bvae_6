import argparse
import csv
import os
import sys
import yaml

from load_hparams import _flatten_hparams

import pandas as pd
import numpy as np
from tqdm import tqdm
import Levenshtein
from Bio.Align import PairwiseAligner, substitution_matrices
from collections import Counter
from scipy.stats import pearsonr, spearmanr


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare per-line sequences between two files: compute Levenshtein distance and BLOSUM62 alignment score."
        )
    p.add_argument("--test_seqs", type=str,
                   help="Path to test sequence file (e.g., data_wsMOQA/cdr_seqs/test.txt)")

    p.add_argument("--decoded_seqs1", type=str,
                   help="Path to previous decoded sequence file (e.g., model_output/binary/decoded.txt)")

    p.add_argument("--decoded_seqs2", type=str, default="./model_output/binary/decoded.txt",
                   help="Path to modified decoded sequence file (e.g., model_output/binary/decoded.txt)")

    p.add_argument("--binary_seqs1", type=str,
                   help="Path to previous binary vector sequence file (e.g., model_output/binary/vector.txt)")

    p.add_argument("--binary_seqs2", type=str, default="./model_output/binary/vector.txt",
                   help="Path to previous binary vector sequence file (e.g., model_output/binary/vector.txt)")

    p.add_argument("--output", type=str, default="./model_output/binary/compare_results.txt",
                   help="Output file path.")

    p.add_argument("--hparams_path", type=str, default="./experiment_configs/binary.yml",
                   help="Path to hparams YAML used for default index settings.")

    p.add_argument("--select_indices", type=str, default=None,
                   help="1-based indices, comma-separated. e.g. '1,2,5,9'")

    p.add_argument("--allow_sidecar_indices", default=False,
                   help="Use per-line index files ./embeddings/test/indices/{i}.idx.npy (1-based). Ignored if --select_indices is set.")

    p.add_argument("--indices_dir", type=str, default="./embeddings/test/indices",
                   help="Directory containing sidecar index files like {i}.idx.npy (1-based).")

    return p.parse_args()

def _load_hp(path: str) -> dict:
    with open(path, "r") as f:
        return _flatten_hparams(
            yaml.safe_load(f),
            preserve_keys={"lr_schedule", "early_stopping"}
        )

def _parse_select_indices(s: str):
    if s is None or str(s).strip() == "":
        return None
    vals = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    return vals if vals else None

def _apply_index_selection(df, select_indices=None, allow_sidecar_indices=False, indices_dir=None):
    # df["seqs"] を行ごとに抽出して返す（1-based指定）
    out = df.copy()
    selected = []

    for i, seq in enumerate(out["seqs"].tolist()):
        chars = list(seq)

        if select_indices is not None:
            sel0 = [k - 1 for k in select_indices if 1 <= k <= len(chars)]
        elif allow_sidecar_indices and indices_dir:
            idx_path = os.path.join(indices_dir, f"{i}.idx.npy")
            if os.path.isfile(idx_path):
                arr = np.load(idx_path).astype(np.int64).reshape(-1)
                sel0 = [int(k - 1) for k in arr if 1 <= int(k) <= len(chars)]
            else:
                sel0 = list(range(len(chars)))  # sidecar無ければ全残基
        else:
            sel0 = list(range(len(chars)))      # 指定なしなら全残基

        selected.append("".join(chars[j] for j in sel0))

    out["seqs"] = selected
    return out

def load_file(path: str):
    comp_seqs = pd.read_csv(path, sep="\t", header=None, names=["seqs"])
    comp_seqs["seqs"] = comp_seqs["seqs"].str.replace(" ", "")
    return comp_seqs


def cal_levenshtein_dist(test_seqs, decoded_seqs):
    lev_distances = []
    for original, decoded in zip(test_seqs["seqs"], decoded_seqs["seqs"]):
        dist = Levenshtein.distance(original, decoded)
        lev_distances.append(dist)
    return lev_distances


def make_distance_count_df(distances, model_name, colname="Levenshtein Distance"):
    counts = Counter(distances)
    counts = dict(sorted(counts.items()))

    df = pd.DataFrame(list(counts.items()), columns=[colname, "Count"])
    df["Model"] = model_name
    return df


def cal_blosum62_score(test_seqs, decoded_seqs):
    aligner = PairwiseAligner(mode="global")
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5

    scores = []
    for original, decoded in zip(test_seqs["seqs"], decoded_seqs["seqs"]):
        score = aligner.score(original, decoded)
        scores.append(score)
    return scores


def make_blosum62_score_df(scores, model_name, binwidth=5):
    # スコアをビンに集計
    counts = Counter((int(score) // binwidth) * binwidth for score in scores)

    # スコアが空なら空DFを返す
    if not counts:
        return pd.DataFrame(columns=["BLOSUM62 Score", "Count", "Model"])

    # Count > 0 が存在するビンの最小〜最大のみを範囲として採用
    present_bins = sorted(counts.keys())
    min_bin, max_bin = present_bins[0], present_bins[-1]
    bins = list(range(min_bin, max_bin + binwidth, binwidth))

    df = (
        pd.DataFrame(list(counts.items()), columns=["BLOSUM62 Score", "Count"])
        .set_index("BLOSUM62 Score")
        .reindex(bins, fill_value=0)
        .reset_index()
    )
    df["Model"] = model_name
    return df


def calculate_levenshtein_distance_matrix(seqs):
    n = seqs.shape[0]
    dist_matrix_Lev = np.zeros([n, n])
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            dist = Levenshtein.distance(seqs.iloc[i]['seqs'], seqs.iloc[j]['seqs'])
            dist_matrix_Lev[i][j] = dist
            dist_matrix_Lev[j][i] = dist
    return dist_matrix_Lev


from Levenshtein import hamming
def calculate_hamming_distance_matrix(vecs):
    n = vecs.shape[0]
    dist_matrix_ham = np.zeros([n, n])
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            dist = hamming(vecs.iloc[i]['seqs'], vecs.iloc[j]['seqs'])
            dist_matrix_ham[i][j] = dist
            dist_matrix_ham[j][i] = dist
    return dist_matrix_ham


def distance_matrix_correlation(mat1, mat2, method="pearson"):
    vec1 = mat1[np.triu_indices_from(mat1, k=1)]
    vec2 = mat2[np.triu_indices_from(mat2, k=1)]

    if method == "pearson":
        r, _ = pearsonr(vec1, vec2)
    elif method == "spearman":
        r, _ = spearmanr(vec1, vec2)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")
    return r


def calculate_blosum62_alignment_matrix(vecs):
    n = vecs.shape[0]
    mat = np.zeros([n, n])

    aligner = PairwiseAligner(mode="global")
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5

    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            score = aligner.score(vecs.iloc[i]['seqs'], vecs.iloc[j]['seqs'])
            mat[i][j] = score
            mat[j][i] = score
    return mat


def main():
    args = parse_args()

    for path in [args.test_seqs, args.decoded_seqs1, args.decoded_seqs2, args.binary_seqs1, args.binary_seqs2]:
        if not os.path.exists(path):
            print(f"Error: not found: {path}", file=sys.stderr)
            sys.exit(1)

    # output ファイルオープン
    with open(args.output, "w") as f:

        hp = _load_hp(args.hparams_path)

        # YAML既定
        yaml_select = hp.get("select_indices", None)
        yaml_allow_sidecar = bool(hp.get("allow_sidecar_indices", False))

        # CLI上書き
        if args.select_indices:
            sel = _parse_select_indices(args.select_indices)
        else:
            sel = yaml_select if (yaml_select is not None and len(yaml_select) > 0) else None

        allow_sidecar = bool(args.allow_sidecar_indices) or (sel is None and yaml_allow_sidecar)
        indices_dir = args.indices_dir

        # sequences file の読み込み
        test_seqs = load_file(args.test_seqs)

        if (sel is not None) or allow_sidecar:
            test_seqs = _apply_index_selection(
                    test_seqs,
                    select_indices=sel,
                    allow_sidecar_indices=allow_sidecar,
                    indices_dir=indices_dir)

        decoded_seqs_pre = load_file(args.decoded_seqs1)
        decoded_seqs_mod = load_file(args.decoded_seqs2)

        # Levenshtein距離
        dist_pre = cal_levenshtein_dist(test_seqs, decoded_seqs_pre)
        dist_mod = cal_levenshtein_dist(test_seqs, decoded_seqs_mod)

        count_pre_df = make_distance_count_df(dist_pre, "Previous")
        count_mod_df = make_distance_count_df(dist_mod, "Modified")

        # 行ずれ防止: 距離で揃えて横持ちに整形し、列順を固定
        count_df = (
            pd.concat([count_pre_df, count_mod_df], ignore_index=True)
            .pivot_table(index="Levenshtein Distance", columns="Model", values="Count", fill_value=0)
            .sort_index()
            .reset_index()
        )
        desired_order = ["Levenshtein Distance", "Previous", "Modified"]
        count_df = count_df[[c for c in desired_order if c in count_df.columns]]
        for col in ["Previous", "Modified"]:
            if col in count_df.columns:
                count_df[col] = count_df[col].astype(int)

        f.write("Levenshtein distances: Previous vs Modified\n")
        f.write(count_df.to_string(index=False) + "\n\n")

        # BLOSUM62スコア
        scores_pre = cal_blosum62_score(test_seqs, decoded_seqs_pre)
        scores_mod = cal_blosum62_score(test_seqs, decoded_seqs_mod)

        count_scores_pre_df = make_blosum62_score_df(scores_pre, "Previous")
        count_scores_mod_df = make_blosum62_score_df(scores_mod, "Modified")

        # 行ずれ防止: スコアで揃えて横持ちに整形し、列順を固定
        count_scores_df = (
            pd.concat([count_scores_pre_df, count_scores_mod_df], ignore_index=True)
            .pivot_table(index="BLOSUM62 Score", columns="Model", values="Count", fill_value=0)
            .sort_index()
            .reset_index()
        )
        desired_order_scores = ["BLOSUM62 Score", "Previous", "Modified"]
        count_scores_df = count_scores_df[[c for c in desired_order_scores if c in count_scores_df.columns]]
        for col in ["Previous", "Modified"]:
            if col in count_scores_df.columns:
                count_scores_df[col] = count_scores_df[col].astype(int)

        f.write("BLOSUM62 scores: Previous vs Modified\n")
        f.write(count_scores_df.to_string(index=False) + "\n\n")

        # 距離行列相関
        vecs_pre = load_file(args.binary_seqs1)
        vecs_mod = load_file(args.binary_seqs2)

        dist_matrix_Lev_test_seqs = calculate_levenshtein_distance_matrix(test_seqs)
        dist_matrix_Ham_vecs_pre = calculate_hamming_distance_matrix(vecs_pre)
        dist_matrix_Ham_vecs_mod = calculate_hamming_distance_matrix(vecs_mod)

        # Pearson
        r1 = distance_matrix_correlation(dist_matrix_Lev_test_seqs, dist_matrix_Ham_vecs_pre, method="pearson")
        r2 = distance_matrix_correlation(dist_matrix_Lev_test_seqs, dist_matrix_Ham_vecs_mod, method="pearson")

        f.write("Levenshtein distance of test_seqs vs Hamming distance of binary vectors:\n")
        f.write(f"Previous model Pearson相関: {r1}\n")
        f.write(f"Modified model Pearson相関: {r2}\n\n")

        # Spearman
        r1 = distance_matrix_correlation(dist_matrix_Lev_test_seqs, dist_matrix_Ham_vecs_pre, method="spearman")
        r2 = distance_matrix_correlation(dist_matrix_Lev_test_seqs, dist_matrix_Ham_vecs_mod, method="spearman")

        f.write(f"Previous model Spearman相関: {r1}\n")
        f.write(f"Modified model Spearman相関: {r2}\n\n")

        # BLOSUM62行列と相関
        blosum62_matrix_test_seqs = calculate_blosum62_alignment_matrix(test_seqs)

        r1 = distance_matrix_correlation(blosum62_matrix_test_seqs, dist_matrix_Ham_vecs_pre, method="pearson")
        r2 = distance_matrix_correlation(blosum62_matrix_test_seqs, dist_matrix_Ham_vecs_mod, method="pearson")

        f.write("BLOSUM62 scores of test_seqs vs Hamming distance of binary vectors:\n")
        f.write(f"Previous model Pearson相関: {r1}\n")
        f.write(f"Modified model Pearson相関: {r2}\n\n")

        r1 = distance_matrix_correlation(blosum62_matrix_test_seqs, dist_matrix_Ham_vecs_pre, method="spearman")
        r2 = distance_matrix_correlation(blosum62_matrix_test_seqs, dist_matrix_Ham_vecs_mod, method="spearman")

        f.write(f"Previous model Spearman相関: {r1}\n")
        f.write(f"Modified model Spearman相関: {r2}\n\n")

        # 長さ分布
        f.write("The length of decoded_seqs by Previous model:\n")
        f.write(decoded_seqs_pre["seqs"].apply(len).value_counts().to_string() + "\n\n")

        f.write("The length of decoded_seqs by Modified model:\n")
        f.write(decoded_seqs_mod["seqs"].apply(len).value_counts().to_string() + "\n\n")


if __name__ == "__main__":
    main()
