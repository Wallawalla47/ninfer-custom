#!/usr/bin/env python3
"""Arm the ngram C>1 verification watcher as a DETACHED process so it keeps running
after this shell/session ends. It fires when the production server on port 8080
closes and the GPU is free.

Re-arm anytime with:  python arm_ngram_c2_watcher.py
Stop it with:         taskkill /F /PID <pid in ngram_c2_watcher.pid>
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WATCHER = HERE / "ngram_c2_watcher.ps1"
PIDFILE = HERE / "ngram_c2_watcher.pid"
LAUNCHLOG = HERE / "ngram_c2_watcher_launch.log"

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000020
CREATE_NO_WINDOW = 0x08000000
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def main() -> int:
    args_tail = sys.argv[1:]
    # Prefer the powershell on PATH (Windows PowerShell 5.1); the script is 5.1-compatible.
    creationflags = 0
    if os.name == "nt":
        creationflags = (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
                         | CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB)
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
           "-File", str(WATCHER), *args_tail]
    with LAUNCHLOG.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd, creationflags=creationflags, stdout=fh, stderr=subprocess.STDOUT,
            cwd=str(HERE), close_fds=True,
        )
        fh.write(f"watcher pid={proc.pid}\n")
    PIDFILE.write_text(str(proc.pid), encoding="utf-8")
    print(f"ngram C>1 watcher armed (detached) pid={proc.pid}; log: ngram_c2_result.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
