import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

# z-scale like 3D descriptor (compact physicochemical representation)
AA_ZSCALE = {
    "A": [0.24, -2.32, 0.60],
    "C": [0.84, -1.67, 3.71],
    "D": [3.98, 0.93, 1.93],
    "E": [3.11, 0.26, -0.11],
    "F": [-4.22, 1.94, 1.06],
    "G": [2.05, -4.06, 0.36],
    "H": [2.47, 1.95, 0.26],
    "I": [-3.89, -1.73, -1.71],
    "K": [2.29, 0.89, -2.49],
    "L": [-4.28, -1.30, -1.49],
    "M": [-2.85, -0.22, 0.47],
    "N": [3.05, 1.62, 1.04],
    "P": [-1.66, 0.27, 1.84],
    "Q": [1.75, 0.50, -1.44],
    "R": [3.52, 2.50, -3.50],
    "S": [2.39, -1.07, 1.15],
    "T": [0.75, -2.18, -1.12],
    "V": [-2.59, -2.64, -1.54],
    "W": [-4.36, 3.94, 0.59],
    "Y": [-2.54, 2.44, 0.43],
}

def _is_numeric_series(s: pd.Series) -> bool:
    coerced = pd.to_numeric(s, errors="coerce")
    return bool(coerced.notna().all())


def encode_sequence(seq: str, descriptor: str, max_len: int) -> List[float]:
    seq = str(seq)

    if descriptor == "onehot":
        dim = len(AA_ORDER)
        vec = np.zeros((max_len, dim), dtype=np.float32)
        aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}
        for i, aa in enumerate(seq[:max_len]):
            idx = aa_to_idx.get(aa)
            if idx is not None:
                vec[i, idx] = 1.0
        return vec.reshape(-1).tolist()

        # zscale3 (default)
    dim = 3
    vec = np.zeros((max_len, dim), dtype=np.float32)
    for i, aa in enumerate(seq[:max_len]):
        if aa in AA_ZSCALE:
            vec[i, :] = np.array(AA_ZSCALE[aa], dtype=np.float32)
    return vec.reshape(-1).tolist()


def build_sequence_features(seq_series: pd.Series, descriptor: str, max_len: int) -> pd.DataFrame:
    encoded = [encode_sequence(seq, descriptor, max_len) for seq in seq_series]
    arr = np.asarray(encoded, dtype=np.float32)
    col_prefix = f"seq_{descriptor}"
    columns = [f"{col_prefix}_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=columns, index=seq_series.index)


def load_train_dev(path: str, aa_descriptor: str, max_seq_len: int = 0) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Rule for first column:
    - numeric: drop it
    - amino acid sequence (string): keep and convert by descriptor

    Target is always the last column.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] < 2:
        raise ValueError("Input data must have at least 2 columns.")

    y = pd.to_numeric(df.iloc[:, -1], errors="coerce")
    if y.isna().any():
        raise ValueError("Last column must be numeric target.")

    X_raw = df.iloc[:, :-1].copy()
    first_col = X_raw.iloc[:, 0]

    if _is_numeric_series(first_col):
        X_raw = X_raw.iloc[:, 1:].copy()
    else:
        seqs = first_col.astype(str)
        seq_len = int(seqs.map(len).max())
        if max_seq_len > 0:
            seq_len = min(seq_len, max_seq_len)
        seq_feat = build_sequence_features(seqs, aa_descriptor, seq_len)

        other = X_raw.iloc[:, 1:].copy()
        if other.shape[1] > 0:
            for c in other.columns:
                if other[c].dtype == object:
                    other[c] = pd.to_numeric(other[c], errors="coerce")
            other = other.fillna(0.0)
            X_raw = pd.concat([seq_feat, other.reset_index(drop=True)], axis=1)
        else:
            X_raw = seq_feat

    X = X_raw.copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category")

    return X, y.astype(float)


def suggest_params(trial: optuna.trial.Trial, seed: int) -> Dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": seed,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        "min_sum_hessian_in_leaf": trial.suggest_float(
            "min_sum_hessian_in_leaf", 1e-6, 1e-2, log=True
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna + LightGBM hyperparameter tuning with CV RMSE"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="C:/Users/hueda8/Desktop/Code/VScode_Data/plm-bvae/data/parrot/peptide/peptide_train_dev.txt",
        help="Path to dataset txt file.",
    )
    parser.add_argument(
        "--train_dev_path",
        type=str,
        default=None,
        help="Deprecated alias of --dataset_path.",
    )
    parser.add_argument(
        "--aa_descriptor",
        type=str,
        default="zscale",
        choices=["zscale", "onehot"],
        help="Amino-acid descriptor for sequence column.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=0,
        help="Max sequence length to encode (0: use dataset max).",
    )
    parser.add_argument(
        "--study_name", type=str, default="peptide_lgb_rmse", help="Optuna study name."
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_peptide_lgb.db",
        help="Optuna storage URL.",
    )
    parser.add_argument("--n_trials", type=int, default=100, help="Number of Optuna trials.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--num_boost_round", type=int, default=5000, help="Max boosting rounds."
    )
    parser.add_argument(
        "--early_stopping_rounds", type=int, default=100, help="Early stopping rounds."
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path if args.train_dev_path is None else args.train_dev_path
    X, y = load_train_dev(dataset_path, args.aa_descriptor, args.max_seq_len)

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_params(trial, args.seed)

        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        fold_rmses = []

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
            tr_x, va_x = X.iloc[tr_idx], X.iloc[va_idx]
            tr_y, va_y = y.iloc[tr_idx], y.iloc[va_idx]

            lgb_train = lgb.Dataset(tr_x, tr_y)
            lgb_valid = lgb.Dataset(va_x, va_y)

            model = lgb.train(
                params=params,
                train_set=lgb_train,
                num_boost_round=args.num_boost_round,
                valid_sets=[lgb_valid],
                valid_names=["valid"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False)
                ],
            )

            va_pred = model.predict(va_x, num_iteration=model.best_iteration)
            rmse = mean_squared_error(va_y, va_pred, squared=False)
            fold_rmses.append(rmse)

            trial.report(rmse, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_rmse = float(np.mean(fold_rmses))
        trial.set_user_attr("fold_rmse", [float(v) for v in fold_rmses])
        return mean_rmse

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print("Number of finished trials:", len(study.trials))
    print("Best trial:")
    print("  Value (CV RMSE):", study.best_trial.value)
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    '{key}': {value}")

    out_dir = Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    study_df_path = out_dir / f"{args.study_name}_trials.csv"
    study.trials_dataframe().to_csv(study_df_path, index=False)
    print(f"Saved trials dataframe: {study_df_path}")

    best_json_path = out_dir / f"{args.study_name}_best_params.json"
    with best_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_value_cv_rmse": float(study.best_value),
                "best_params": study.best_trial.params,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved best params: {best_json_path}")


if __name__ == "__main__":
    main()



