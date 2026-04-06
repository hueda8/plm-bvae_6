#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
import datetime
from typing import List

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run multiple action scripts over multiple hparams YAMLs sequentially."
    )
    ap.add_argument(
        "--actions", nargs="+", required=True,
        help="List of Python scripts to run (e.g. train.py evaluate.py)"
    )
    ap.add_argument(
        "--hparams", nargs="+", required=True,
        help="List of hparams YAML files (e.g. experiment_configs/binary.yml other.yml)"
    )
    ap.add_argument(
        "--stop_on_error", action="store_true",
        help="Stop execution on first failure (non-zero return code)."
    )
    ap.add_argument(
        "--log_file", type=str, default=None,
        help="Optional path to save a log summary."
    )
    return ap.parse_args()

def main():
    args = parse_args()
    python_exec = sys.executable

    commands: List[List[str]] = []
    for action in args.actions:
        for hp in args.hparams:
            commands.append([python_exec, action, hp])

    print("[INFO] Python:", python_exec)
    print("[INFO] Will run commands ({} total):".format(len(commands)))
    for c in commands:
        print("  ", " ".join(c))

    timings = {}
    failures = []

    for c in commands:
        cmd_str = " ".join(c)
        print("\n[RUN] Starting:", cmd_str)
        start = time.time()
        try:
            result = subprocess.run(c, stdout=sys.stdout, stderr=sys.stderr)
            rc = result.returncode
        except Exception as e:
            rc = -999
            print(f"[ERROR] Exception while running {cmd_str}: {e}")

        elapsed = time.time() - start
        timings[cmd_str] = elapsed
        if rc != 0:
            print(f"[WARN] Command failed (rc={rc}): {cmd_str}")
            failures.append((cmd_str, rc))
            if args.stop_on_error:
                print("[INFO] stop_on_error enabled; aborting remaining commands.")
                break
        else:
            print(f"[DONE] {cmd_str} finished in {elapsed:.2f}s")

    # Summary
    print("\n[SUMMARY] Timings:")
    for cmd, sec in timings.items():
        td = datetime.timedelta(seconds=sec)
        print(f"  {td}  {cmd}")

    if failures:
        print("\n[SUMMARY] Failures:")
        for cmd, rc in failures:
            print(f"  rc={rc}  {cmd}")
    else:
        print("\n[SUMMARY] All commands succeeded.")

    if args.log_file:
        try:
            with open(args.log_file, "w") as f:
                f.write("Command,Seconds,ReturnCode\n")
                for cmd, sec in timings.items():
                    rc = 0
                    for fc, fr in failures:
                        if fc == cmd:
                            rc = fr
                            break
                    f.write(f"{cmd},{sec:.4f},{rc}\n")
            print(f"[INFO] Log written to {args.log_file}")
        except Exception as e:
            print(f"[WARN] Failed to write log file {args.log_file}: {e}")

if __name__ == "__main__":
    main()
