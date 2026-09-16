"""Run the OLD ngram-mod C>1 coherence tests (E:\\NInfer-Claude) against the NEW build.

The tests that demonstrated the correctness issue are unchanged here:
  * test_single_coherence.py  - C=1 path sanity: 2 x 2000-token free-form generations,
    per-400-token unique ratio must stay >= 0.15 (gibberish/repetition otherwise).
  * test_c3_stridefix.py      - the C=3 concurrent reproduction: 3 free-form requests
    per round, temperature 0, per-lane unique ratio must stay >= 0.20. This is the test
    that showed one lane per round collapsing to token repetition (uniq ~0.03).

Both speak the OpenAI wire protocol to http://127.0.0.1:1987, so they run unmodified
against any NInfer server launched on that port. This harness launches this tree's
build with the same server configuration the old rig (arm_stridefix3.py) used, then
runs both old tests and reports.

Never touches production port 8080.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

EXE = r"E:\NInfer-V3\build-windows\apps\Release\ninfer-serve.exe"
ART = r"E:\NInfer-Deploy-V3\qwen3_8_27b_nvfp4-official.ninfer"
OLD = r"E:\NInfer-Claude"
PORT = 1987
BASE = f"http://127.0.0.1:{PORT}"
ALIAS = "qwen3.8-27b"          # the model id the old tests send
LOG = r"E:\NInfer-V3\old_c3_tests_serve.log"
SUM = r"E:\NInfer-V3\old_c3_tests_SUMMARY.txt"
PY = sys.executable


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(SUM, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def listen_pids(port: int) -> set[str]:
    try:
        out = subprocess.check_output(["netstat", "-ano"], timeout=10, text=True)
    except Exception:
        return set()
    pids = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and f":{port}" in parts[1] and "LISTENING" in line:
            pids.add(parts[4])
    return pids


def free_mib() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
                                       "--format=csv,noheader"], timeout=10, text=True)
        return int(out.strip().split(",")[0].split()[0])
    except Exception:
        return 0


def kill_port(port: int) -> None:
    if port == 8080:
        raise RuntimeError("refusing to kill production 8080")
    for pid in listen_pids(port):
        subprocess.run(["taskkill", "/f", "/pid", pid], capture_output=True)


def run_test(name: str, script: str, args: list[str], result: dict) -> None:
    log(f"--- {name}: {script} {' '.join(args)} ---")
    t0 = time.time()
    proc = subprocess.run([PY, os.path.join(OLD, script), *args],
                          capture_output=True, text=True, timeout=1800)
    elapsed = int(time.time() - t0)
    out = (proc.stdout or "").strip()
    for line in out.splitlines():
        log("    " + line)
    if proc.returncode:
        err = (proc.stderr or "").strip()
        if err:
            log("    stderr: " + err.replace("\n", "\n    "))
    log(f"    {name} exit={proc.returncode} elapsed={elapsed}s")
    result[name] = (proc.returncode, elapsed, out)


def main() -> int:
    open(SUM, "w", encoding="utf-8").close()
    for path, label in ((EXE, "build"), (ART, "artifact"),
                        (os.path.join(OLD, "test_c3_stridefix.py"), "old C=3 test"),
                        (os.path.join(OLD, "test_single_coherence.py"), "old C=1 test")):
        if not os.path.exists(path):
            log(f"MISSING {label}: {path}")
            return 2
    free, prod = free_mib(), bool(listen_pids(8080))
    log(f"GPU free={free} MiB  prod8080={prod}")
    if prod or free < 25000:
        log("GPU not free (production up or <25 GiB free); aborting without running.")
        return 3

    # Same server configuration arm_stridefix3.py used, plus a matching public alias.
    args = [EXE, ART, "--host", "127.0.0.1", "--port", str(PORT), "--max-context", "8192",
            "--max-concurrency", "3", "--spec", "mtp", "--draft-tokens", "4",
            "--lm-head-draft", "--ngram-draft-tokens", "15", "--ngram-min-match", "12",
            "--kv-dtype", "int8", "--model-id", ALIAS, "--log-colours", "off"]
    log("server args: " + " ".join(args))
    srv_out = open(LOG, "w", encoding="utf-8")
    srv = subprocess.Popen(args, stdout=srv_out, stderr=subprocess.STDOUT)

    results: dict[str, tuple] = {}
    try:
        ready = False
        for _ in range(300):
            if srv.poll() is not None:
                log(f"server exited early rc={srv.poll()}")
                break
            try:
                urllib.request.urlopen(BASE + "/health", timeout=3)
                ready = True
                break
            except Exception:
                time.sleep(1)
        if not ready:
            with open(LOG, encoding="utf-8", errors="replace") as fh:
                log("SERVER NOT READY; log tail:\n" + "".join(fh.readlines()[-40:]))
            return 1
        log("server ready")

        run_test("single_coherence_C1", "test_single_coherence.py", [], results)
        run_test("c3_concurrent_gibberish", "test_c3_stridefix.py", ["5", "500"], results)
    finally:
        try:
            srv.kill()
            srv.wait(timeout=60)
        except Exception:
            pass
        time.sleep(2)
        kill_port(PORT)
        srv_out.close()

    log("==================== VERDICT ====================")
    failed = 0
    for name, (code, elapsed, _) in results.items():
        log(f"  {name:24s} exit={code} ({elapsed}s)")
        if code != 0:
            failed += 1
    log(f"  OVERALL: {'ALL COHERENT' if failed == 0 else f'{failed} test(s) failed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())