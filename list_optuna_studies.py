#!/usr/bin/env python3
"""
Optuna の過去 Study 一覧を表示・削除するユーティリティ。

使い方例:
  - 一覧表示（テキスト）
    python list_optuna_studies.py --storage-url sqlite:///optuna_plm_bvae.db

  - 一覧表示（JSON）
    python list_optuna_studies.py --storage-url sqlite:///optuna_plm_bvae.db --json

  - Study を削除
    python list_optuna_studies.py --storage-url sqlite:///optuna_plm_bvae.db --delete-study plm_bvae_opt_old
"""

import argparse
import optuna
import json
from typing import Any, Dict


def summarize_study(study: optuna.Study) -> Dict[str, Any]:
    """Study オブジェクトから要約情報を辞書化."""
    info: Dict[str, Any] = {
        "study_name": study.study_name,
        "direction": str(study.direction),
        "n_trials": len(study.trials),
    }
    try:
        if study.best_trial is not None and study.best_trial.value is not None:
            info["best_value"] = float(study.best_value)
            info["best_trial_number"] = int(study.best_trial.number)
            info["best_params"] = study.best_trial.params
        else:
            info["best_value"] = None
            info["best_trial_number"] = None
            info["best_params"] = None
    except Exception:
        info["best_value"] = None
        info["best_trial_number"] = None
        info["best_params"] = None
    return info


def main():
    parser = argparse.ArgumentParser(
        description="Optuna storage から過去 Study 一覧を取得して表示・削除するスクリプト。"
    )
    parser.add_argument(
        "--storage-url",
        required=True,
        type=str,
        help="Optuna のストレージ URL (例: sqlite:///optuna_plm_bvae.db)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON 形式で詳細一覧を出力。",
    )
    parser.add_argument(
        "--delete-study",
        type=str,
        default=None,
        help="削除したい Study 名。指定された場合は一覧表示後に削除を実行。",
    )
    args = parser.parse_args()

    storage_url = args.storage_url

    # Study 名一覧の取得
    try:
        study_names = optuna.get_all_study_names(storage=storage_url)
    except Exception as e:
        print(f"[ERROR] Study 名一覧の取得に失敗しました: {e}")
        print("ヒント: --storage-url は SQLAlchemy 形式の URL を指定してください。例: sqlite:///optuna_plm_bvae.db")
        return

    if not study_names:
        print(f"[INFO] ストレージ '{storage_url}' に Study は存在しません。")
    else:
        if args.json:
            # JSON モード: すべての Study 要約を JSON 配列で出力
            summaries = []
            for name in study_names:
                try:
                    st = optuna.load_study(study_name=name, storage=storage_url)
                    summaries.append(summarize_study(st))
                except Exception as e:
                    summaries.append(
                        {
                            "study_name": name,
                            "error": f"Failed to load: {e}",
                        }
                    )
            print(json.dumps(summaries, ensure_ascii=False, indent=2))
        else:
            # テキストモード: 人間可読な形式
            print(f"[INFO] Storage: {storage_url}")
            print(f"[INFO] Study count: {len(study_names)}\n")
            for name in study_names:
                print(f"Study: {name}")
                try:
                    st = optuna.load_study(study_name=name, storage=storage_url)
                    summary = summarize_study(st)
                    print(f"  direction          : {summary['direction']}")
                    print(f"  n_trials           : {summary['n_trials']}")
                    if summary["best_value"] is not None:
                        print(f"  best_value         : {summary['best_value']:.6f}")
                        print(f"  best_trial_number  : {summary['best_trial_number']}")
                        params_str = ", ".join(
                            f"{k}={v}" for k, v in (summary["best_params"] or {}).items()
                        )
                        print(f"  best_params        : {params_str}")
                    else:
                        print("  (no completed trials)")
                except Exception as e:
                    print(f"  Failed to load study details: {e}")
                print("-" * 60)

    # 削除処理（オプション）
    target = args.deleteStudy if hasattr(args, "deleteStudy") else args.delete_study
    if target:
        try:
            names = optuna.get_all_study_names(storage=storage_url)
            if target in names:
                optuna.delete_study(study_name=target, storage=storage_url)
                print(f"Deleted: {target}")
            else:
                print(f"Study '{target}' not found. Existing: {names}")
        except Exception as e:
            print(f"[ERROR] Study 削除に失敗しました: {e}")


if __name__ == "__main__":
    main()
