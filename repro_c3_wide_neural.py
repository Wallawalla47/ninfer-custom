"""Reproduce the prior C=3 "gibberish" scenario against THIS build.

Config mirrors E:\\NInfer-Claude\\run_mtpdiag_now.py (MTP + ngram + CUDA graphs,
int8 KV, 3 concurrent free-form requests, temperature 0) but points at the V3
build and the V3 quasar artifact. Never touches port 8080.

Usage: python repro_c3_wide_neural.py [rounds] [tokens]
Exit 0 = all lanes coherent; 1 = gibberish detected.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

EXE = r"E:\NInfer-V3\build-windows\apps\Release\ninfer-serve.exe"
ART = r"E:\NInfer-Deploy-V3\qwen3_8_27b_nvfp4-quasar.ninfer"
PORT = 1987
BASE = f"http://127.0.0.1:{PORT}"
LOG = r"E:\NInfer-V3\c3_wide_neural_serve.log"
SUM = r"E:\NInfer-V3\c3_wide_neural_SUMMARY.txt"

PROMPTS = [
    "Write a 200 word story about a lighthouse keeper during a storm.",
    "Explain how a coffee percolator works in detail.",
    "Describe the process of making sourdough bread from scratch.",
    "Write a polite email declining a meeting rescheduled to next week.",
    "Explain the difference between a compiler and an interpreter.",
]


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


def kill_port(port: int) -> None:
    if port == 8080:
        raise RuntimeError("refusing to kill production 8080")
    for pid in listen_pids(port):
        subprocess.run(["taskkill", "/f", "/pid", pid], capture_output=True)


def gen(prompt: str, max_tokens: int) -> str:
    body = json.dumps({"model": "qwen3.8-27b",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0,
                       "stream": False}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.loads(r.read().decode())
        m = d["choices"][0]["message"]
        return (m.get("reasoning_content") or "") + (m.get("content") or "")
    except Exception as e:  # noqa: BLE001
        return f"__ERR__{type(e).__name__}:{e}"


def coherence(text: str) -> float:
    toks = text.split() if text and not text.startswith("__ERR__") else []
    if len(toks) < 20:
        return 0.0
    return len(set(toks)) / len(toks)


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    artifact = sys.argv[3] if len(sys.argv) > 3 else ART
    archive = len(sys.argv) > 4 and sys.argv[4] == "archive"
    open(SUM, "w", encoding="utf-8").close()
    log("=== C=3 wide-neural gibberish repro (this build) ===")
    if listen_pids(8080):
        log("WARNING: port 8080 is in use; running anyway, but GPU may be contended")
    args = [EXE, artifact, "--host", "127.0.0.1", "--port", str(PORT),
            "--max-context", "4096", "--max-concurrency", "3", "--spec", "mtp",
            "--draft-tokens", "4", "--lm-head-draft", "--ngram-draft-tokens", "15",
            "--ngram-min-match", "12", "--kv-dtype", "int8", "--log-colours", "off"]
    if archive:
        args += ["--ngram-archive-mib", "256", "--ngram-session-mib", "64",
                 "--ngram-native-sessions"]
    log("args: " + " ".join(args))
    srv_out = open(LOG, "w", encoding="utf-8")
    srv = subprocess.Popen(args, stdout=srv_out, stderr=subprocess.STDOUT)
    log(f"serve pid={srv.pid}")
    gibber = -1
    try:
        ready = False
        waited = 0
        while waited < 600:
            if srv.poll() is not None:
                log(f"serve EXITED rc={srv.poll()} before ready")
                break
            try:
                urllib.request.urlopen(BASE + "/health", timeout=3)
                ready = True
                break
            except Exception:
                waited += 5
                if waited % 30 == 0:
                    log(f"... waiting for ready ({waited}s)")
                time.sleep(5)
        if not ready:
            try:
                with open(LOG, encoding="utf-8", errors="replace") as fh:
                    tail = fh.readlines()[-40:]
            except Exception:
                tail = []
            log("SERVER NOT READY. tail:\n" + "".join(tail))
            return 1
        log(f"server ready after ~{waited}s")
        log(f"TEST: {rounds} rounds x 3 concurrent free-form req, {tokens} tokens")
        gibber = 0
        for rnd in range(rounds):
            with ThreadPoolExecutor(max_workers=3) as ex:
                futs = {ex.submit(gen, PROMPTS[(rnd + i) % len(PROMPTS)], tokens): i
                        for i in range(3)}
                results = {futs[f]: f.result() for f in as_completed(futs)}
            print(f"round {rnd}:", flush=True)
            for i in range(3):
                txt = results.get(i, "__MISS__")
                coh = coherence(txt)
                toks = txt.split()
                top = Counter(toks).most_common(1)[0] if toks else ("none", 0)
                bad = coh < 0.20 or txt.startswith("__ERR__")
                gibber += 1 if bad else 0
                print(f"  lane {i}: {'GIBBERISH' if bad else 'OK':9s} uniq={coh:.2f} "
                      f"top={top} len={len(txt)}", flush=True)
        log(f"RESULT: {'GIBBERISH DETECTED' if gibber else 'ALL COHERENT'} "
            f"({gibber} bad lanes / {rounds * 3})")
    finally:
        try:
            srv.kill()
            srv.wait(timeout=60)
        except Exception:
            pass
        time.sleep(2)
        kill_port(PORT)
        log("=== cleanup: test server on 1987 killed (prod 8080 untouched) ===")
    return 1 if gibber else 0


if __name__ == "__main__":
    sys.exit(main())