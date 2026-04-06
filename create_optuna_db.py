#!/usr/bin/env python3
"""
create_optuna_db.py

Supercomputer / HPC friendly initialization script for an Optuna SQLite storage.
This script:
  - Decides a safe path for the SQLite file (optionally using scratch/TMPDIR).
  - Creates the SQLite database file if it does not exist.
  - Applies recommended PRAGMA settings for multi-process robustness on shared filesystems.
  - Optionally bootstraps an Optuna study and inserts a dummy trial so tables are created.
  - Verifies integrity and prints basic info.

USAGE EXAMPLES:
  python create_optuna_db.py                          # uses default ./optuna_plm_bvae.db
  python create_optuna_db.py --db-path /path/to/db/optuna.db
  python create_optuna_db.py --study-name plm_bvae_opt --bootstrap-optuna
  python create_optuna_db.py --use-tmpdir --bootstrap-optuna
  srun -n1 python create_optuna_db.py --db-path $SCRATCH/optuna/optuna_plm_bvae.db --bootstrap-optuna

NOTES FOR HPC:
  1. Prefer node-local SSD or per-job scratch ($TMPDIR, $LOCAL_SCRATCH, $SCRATCH) over shared NFS to reduce locking contention.
  2. WAL journal mode may improve concurrency but on some parallel filesystems (e.g. older Lustre) WAL is not fully supported.
     If you see 'database is locked' errors, consider:
       - Increasing --busy-timeout
       - Reducing concurrent writers (Optuna uses short transactions)
       - Falling back to journal_mode=DELETE
  3. If running many simultaneous trials via Optuna distributed optimization, consider switching to MySQL/PostgreSQL for higher concurrency.

"""

import os
import sys
import sqlite3
import argparse
import textwrap
import datetime
from typing import Optional

# Optional: initialize Optuna structures
try:
    import optuna
    _HAS_OPTUNA = True
except Exception:
    _HAS_OPTUNA = False


def resolve_db_path(args: argparse.Namespace) -> str:
    if args.db_path:
        return args.db_path
    # Fallback to local directory
    return "optuna_plm_bvae.db"


def ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def apply_pragma(conn: sqlite3.Connection, args: argparse.Namespace):
    """
    Apply PRAGMA settings to improve robustness. Adjust if your filesystem
    does not support WAL or if locking issues arise.
    """
    cursor = conn.cursor()

    # Journal mode: WAL (better concurrency) or fallback to DELETE
    journal_mode = "WAL" if args.journal_wal else "DELETE"
    cursor.execute(f"PRAGMA journal_mode={journal_mode};")

    # Synchronous modes: FULL (safest), NORMAL (fast), OFF (risky)
    cursor.execute(f"PRAGMA synchronous={args.synchronous};")

    # Busy timeout to reduce 'database is locked' errors in concurrent use
    cursor.execute(f"PRAGMA busy_timeout={args.busy_timeout};")

    # Foreign keys for Optuna integrity
    cursor.execute("PRAGMA foreign_keys=ON;")

    # Cache size (negative => KB)
    if args.cache_kb > 0:
        cursor.execute(f"PRAGMA cache_size=-{int(args.cache_kb)};")

    conn.commit()


def bootstrap_optuna(storage_path: str, study_name: str):
    if not _HAS_OPTUNA:
        print("[WARN] Optuna is not installed; skipping bootstrap.")
        return

    print(f"[OPTUNA] Creating/loading study '{study_name}' in storage '{storage_path}' ...")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        direction="minimize",
        load_if_exists=True,
    )

    # Insert a dummy trial if no trials exist (ensures tables exist)
    if len(study.trials) == 0:
        def dummy_objective(trial: optuna.Trial):
            # Return a very large finite value to avoid impacting statistics with inf/NaN.
            return 1e12

        study.optimize(dummy_objective, n_trials=1)
        print("[OPTUNA] Dummy trial inserted to initialize schema (value=1e12).")
    else:
        print(f"[OPTUNA] Study already has {len(study.trials)} trial(s). No dummy insertion needed.")


def verify_db(conn: sqlite3.Connection):
    cursor = conn.cursor()
    # Basic integrity check
    try:
        cursor.execute("PRAGMA integrity_check;")
        res = cursor.fetchall()
        ok = all(r[0] == "ok" for r in res)
        print(f"[CHECK] integrity_check: {res} (OK={ok})")
    except Exception as e:
        print(f"[WARN] integrity_check failed: {e}")

    # List tables if Optuna already created them
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cursor.fetchall()]
        print(f"[INFO] Tables present: {tables}")
    except Exception as e:
        print(f"[WARN] Could not list tables: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Initialize an Optuna SQLite database on an HPC system.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--db_path", type=str, default=None,
                        help="Path to SQLite file. If omitted, tries ./optuna_plm_bvae.db.")
    parser.add_argument("--bootstrap_optuna", action="store_true",
                        help="Create/load an Optuna study and insert a dummy trial to initialize schema.")
    parser.add_argument("--study_name", type=str, default="plm_bvae_opt",
                        help="Study name used when bootstrapping Optuna.")

    # PRAGMA related arguments
    parser.add_argument("--journal_wal", action="store_true",
                        help="Use WAL journal mode (recommended if filesystem supports it).")
    parser.add_argument("--synchronous", type=str, default="NORMAL", choices=["FULL", "NORMAL", "OFF"],
                        help="PRAGMA synchronous level.")
    parser.add_argument("--busy_timeout", type=int, default=8000,
                        help="PRAGMA busy_timeout in milliseconds.")
    parser.add_argument("--cache_kb", type=int, default=0,
                        help="PRAGMA cache_size in KB (0 = leave default).")

    args = parser.parse_args()

    db_path = resolve_db_path(args)
    ensure_parent_dir(db_path)
    new_file = not os.path.isfile(db_path)

    print(textwrap.dedent(f"""
        =====================================
        Optuna SQLite DB Initialization
        =====================================
        Target file : {db_path}
        New file?    : {new_file}
        Journal WAL  : {args.journal_wal}
        Synchronous  : {args.synchronous}
        Busy timeout : {args.busy_timeout} ms
        Cache size   : {args.cache_kb if args.cache_kb > 0 else 'default'}
        Bootstrap    : {args.bootstrap_optuna}
        Study name   : {args.study_name}
        Timestamp    : {datetime.datetime.utcnow().isoformat()}Z
        =====================================
    """).strip())

    # Connect (creates file if it does not exist)
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        print(f"[ERROR] Failed to open/create SQLite file: {e}")
        sys.exit(1)

    try:
        apply_pragma(conn, args)
        verify_db(conn)
    finally:
        conn.close()

    if args.bootstrap_optuna:
        storage_path = f"sqlite:///{os.path.abspath(db_path)}"
        bootstrap_optuna(storage_path, args.study_name)

    print("[DONE] Initialization complete.")


if __name__ == "__main__":
    main()
