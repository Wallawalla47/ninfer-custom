"""Probe /v1/models and /v1/models/{id} on a live ninfer-serve and print the metadata.

Launches a short-lived server on port 1987 with a distinctive configured context,
fetches the model objects, then kills only port 1987 (never prod 8080).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request

EXE = r"E:\NInfer-V3\build-windows\apps\Release\ninfer-serve.exe"
ART = r"E:\NInfer-Deploy-V3\qwen3_8_27b_nvfp4-official.ninfer"
PORT = 1987
BASE = f"http://127.0.0.1:{PORT}"
LOG = r"E:\NInfer-V3\models_meta_serve.log"
ALIAS = "qwen3.8-27b-test"
MAX_CONTEXT = "8192"


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


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode())


def main() -> int:
    if listen_pids(8080):
        print("WARNING: prod 8080 is up; GPU may be contended")
    args = [EXE, ART, "--host", "127.0.0.1", "--port", str(PORT),
            "--max-context", MAX_CONTEXT, "--model-id", ALIAS, "--log-colours", "off"]
    print("args: " + " ".join(args))
    with open(LOG, "w", encoding="utf-8") as out:
        srv = subprocess.Popen(args, stdout=out, stderr=subprocess.STDOUT)
    print(f"serve pid={srv.pid}")
    try:
        ready = False
        for _ in range(120):
            if srv.poll() is not None:
                print(f"serve EXITED rc={srv.poll()}")
                break
            try:
                urllib.request.urlopen(BASE + "/health", timeout=3)
                ready = True
                break
            except Exception:
                time.sleep(5)
        if not ready:
            print("SERVER NOT READY")
            return 1
        listing = get("/v1/models")
        single = get(f"/v1/models/{ALIAS}")
        print("---- GET /v1/models ----")
        print(json.dumps(listing, indent=2))
        print("---- GET /v1/models/{id} ----")
        print(json.dumps(single, indent=2))
        meta = listing["data"][0]["meta"]
        ok = (listing["data"][0]["id"] == ALIAS
              and single["id"] == ALIAS
              and listing["data"][0]["max_model_len"] == int(MAX_CONTEXT)
              and meta["n_ctx"] == int(MAX_CONTEXT)
              and meta["n_ctx_train"] > 0
              and meta["n_vocab"] > 0 and meta["n_embd"] > 0
              and meta["n_params"] > 0 and meta["size"] > 0
              and len(meta["ftype"]) > 0)
        print(f"CHECK n_ctx={meta['n_ctx']} (configured) n_ctx_train={meta['n_ctx_train']} "
              f"(native) ftype={meta['ftype']!r} -> {'OK' if ok else 'FAIL'}")
        return 0 if ok else 1
    finally:
        try:
            srv.kill()
            srv.wait(timeout=60)
        except Exception:
            pass
        time.sleep(2)
        kill_port(PORT)


if __name__ == "__main__":
    sys.exit(main())