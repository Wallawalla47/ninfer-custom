#!/usr/bin/env python3
"""Analyze an agentic A/B run directory and write report.md + summary.json into it.

Usage: python analyze.py <run_dir>

Each arm directory holds the client's own record of every request (client.jsonl, with
the per-request seed and the workload class) and the serve's request log
(request_log.jsonl). They are joined on the request seed, which is identical in both arms,
so every metric compares the same logical requests. All timings and token counts come from
the serve's request log.
"""
import json
import os
import re
import statistics
import sys

ARMS = ("control", "treatment")
LABEL = {"control": "Upstream + Windows port", "treatment": "This fork"}
CONTINUING = {"loop", "after_idle", "history_edit", "retry", "abort_retry", "subagent_loop"}
NEW_LONG = {"cold_resume", "compaction", "check"}
CLASS_DOC = [
    ("cold_resume", "a saved session resumed after a restart (whole prompt is new)"),
    ("loop", "an agent tool-loop turn continuing its own conversation"),
    ("subagent_first", "a subagent's first turn (shares its system+tools prefix with siblings)"),
    ("subagent_loop", "a subagent tool-loop turn"),
    ("compaction", "the whole conversation sent for summarisation (new prompt)"),
    ("restart", "the fresh context after compaction (system prompt + summary)"),
    ("check", "a loop-detection side call over the whole history (new prompt)"),
    ("after_idle", "a session resumed after sitting idle while other sessions ran"),
    ("history_edit", "the first turn after the client cleared old tool results"),
    ("retry", "the user retrying the previous turn with the identical prompt"),
    ("abort_retry", "the retry after the client aborted a long request"),
]


def load_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        rows.append(json.loads(ln))
                    except ValueError:
                        pass
    return rows


def load_arm(run_dir, arm):
    d = os.path.join(run_dir, arm)
    client = load_jsonl(os.path.join(d, "client.jsonl"))
    server = load_jsonl(os.path.join(d, "request_log.jsonl"))
    meta = {}
    if os.path.exists(os.path.join(d, "arm.json")):
        with open(os.path.join(d, "arm.json"), encoding="utf-8") as f:
            meta = json.load(f)
    done, errs, start, thr = {}, {}, None, []
    for o in server:
        ev = o.get("event")
        seed = ((o.get("request") or {}).get("sampling") or {}).get("seed")
        if ev == "server_start":
            start = o
        elif ev == "request_done" and seed is not None:
            done[seed] = o
        elif ev in ("request_error", "request_rejected") and seed is not None:
            errs[seed] = o
        elif ev == "throughput":
            thr.append(o)
    reqs, events = [], []
    for c in client:
        if "event" in c:
            events.append(c)
            continue
        s = done.get(c["seed"])
        r = dict(c)
        r["server"] = s
        if s:
            res, t = s.get("result") or {}, s.get("timings_seconds") or {}
            spec = s.get("speculative") or {}
            r.update({
                "prompt": res.get("prompt_tokens") or 0,
                "hit": res.get("prefix_cache_hit_tokens") or 0,
                "computed": res.get("computed_prefill_tokens") or 0,
                "completion": res.get("completion_tokens") or 0,
                "thinking": res.get("model_thinking_tokens") or 0,
                "path": res.get("prefix_reuse_path") or "unknown",
                "finish_server": res.get("finish_reason"),
                "ttft": t.get("ttft"), "prefill_s": t.get("prefill") or 0.0,
                "decode_s": t.get("decode") or 0.0, "total_s": t.get("total") or 0.0,
                "queue_s": (s.get("engine_timing") or {}).get("queue_wait_seconds") or 0.0,
                "end": s["timestamp_unix_ms"] / 1000.0,
                "spec_accepted": spec.get("accepted_tokens") or 0,
                "spec_drafted": spec.get("drafted_tokens") or 0,
                "ngram_accepted": spec.get("ngram_accepted_tokens") or 0,
                "ngram_drafted": spec.get("ngram_drafted_tokens") or 0,
                "rounds": spec.get("rounds") or 0,
            })
        reqs.append(r)
    build = "?"
    serve_log = os.path.join(d, "serve.log")
    if os.path.exists(serve_log):
        with open(serve_log, encoding="utf-8", errors="replace") as f:
            for ln in f:
                m = re.search(r"build ([0-9a-f]{7,40})", ln)
                if m:
                    build = m.group(1)
                    break
    return {"arm": arm, "requests": reqs, "events": events, "start": start,
            "throughput": thr, "meta": meta, "errors": errs, "build": build}


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def median(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def pct(v, p):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def decode_overlaps(rows):
    """Mark requests whose decode window overlapped another request's decode window."""
    win = [(r["end"] - r["decode_s"], r["end"], id(r)) for r in rows if r.get("decode_s")]
    alone = set()
    for a0, a1, ia in win:
        if not any(ib != ia and b0 < a1 and a0 < b1 for b0, b1, ib in win):
            alone.add(ia)
    return alone


def solo_seeds(A):
    """Seeds of requests (>=128 output tokens) whose decode overlapped no other request."""
    rows = [r for r in A["requests"] if r.get("server") and r["cls"] != "aborted"
            and r["decode_s"] > 0]
    alone = decode_overlaps(rows)
    return {r["seed"] for r in rows if id(r) in alone and r["completion"] >= 128}


def metrics(A, cold_seeds, matched_solo=frozenset()):
    rows = [r for r in A["requests"] if r.get("server") and r["cls"] != "aborted"]
    cont = [r for r in rows if r["cls"] in CONTINUING]
    newl = [r for r in rows if r["cls"] in NEW_LONG]
    cold = [r for r in rows if r["seed"] in cold_seeds]
    dec = [r for r in rows if r["decode_s"] > 0 and r["completion"] > 1]
    copy = [r for r in dec if r["copy"]]
    alone = decode_overlaps(rows)
    solo = [r["completion"] / r["decode_s"] for r in dec
            if id(r) in alone and r["completion"] >= 64]
    big = [r for r in cold if r["computed"] >= 32768]
    tok = sum((o.get("tokens") or {}).get("committed_decode", 0) for o in A["throughput"])
    active = sum(o.get("interval_seconds") or 0 for o in A["throughput"]
                 if (o.get("tokens") or {}).get("committed_decode", 0) > 0)
    batch = [o["decode_batch"]["average_size"] for o in A["throughput"]
             if (o.get("decode_batch") or {}).get("average_size")]
    solo_rows = [r for r in dec if id(r) in alone and r["completion"] >= 128]
    solo_ng = [r for r in solo_rows if r["ngram_accepted"] >= 0.2 * r["completion"]]
    solo_plain = [r for r in solo_rows if r["ngram_accepted"] < 0.2 * r["completion"]]
    budget = ((A["start"] or {}).get("server") or {}).get("default_thinking_budget")
    msolo = [r["completion"] / r["decode_s"] for r in dec if r["seed"] in matched_solo]
    tp = sum(r["prompt"] for r in rows)
    th = sum(r["hit"] for r in rows)
    m = {
        "n": len(rows),
        "n_failed": sum(1 for r in A["requests"] if r["cls"] != "aborted" and not r.get("server")),
        "ttft_mean": mean([r["ttft"] for r in rows]),
        "ttft_median": median([r["ttft"] for r in rows]),
        "ttft_p90": pct([r["ttft"] for r in rows], 0.9),
        "service_ttft_mean": mean([r["ttft"] - r["queue_s"] for r in rows]),
        "queue_mean": mean([r["queue_s"] for r in rows]),
        "ttft_cont_mean": mean([r["ttft"] for r in cont]),
        "ttft_cont_median": median([r["ttft"] for r in cont]),
        "ttft_new_mean": mean([r["ttft"] for r in newl]),
        "n_cont": len(cont), "n_new": len(newl),
        "prompt_tokens": tp, "hit_tokens": th, "hit_rate": th / tp if tp else None,
        "computed_tokens": sum(r["computed"] for r in rows),
        "cont_full_prefill": sum(1 for r in cont if r["hit"] == 0),
        "cont_full_prefill_main": sum(1 for r in cont if r["hit"] == 0 and r["cls"] != "subagent_loop"),
        "cont_full_prefill_sub": sum(1 for r in cont if r["hit"] == 0 and r["cls"] == "subagent_loop"),
        "n_cont_main": sum(1 for r in cont if r["cls"] != "subagent_loop"),
        "n_cont_sub": sum(1 for r in cont if r["cls"] == "subagent_loop"),
        "cold_prefill_big": (sum(r["computed"] for r in big) /
                             sum(r["prefill_s"] for r in big)) if big else None,
        "n_cold_big": len(big),
        "server_decode_tps": tok / active if active else None,
        "mean_decode_batch": mean(batch),
        "solo_plain_tps": (sum(r["completion"] for r in solo_plain) /
                           sum(r["decode_s"] for r in solo_plain)) if solo_plain else None,
        "solo_ngram_tps": (sum(r["completion"] for r in solo_ng) /
                           sum(r["decode_s"] for r in solo_ng)) if solo_ng else None,
        "n_solo_plain": len(solo_plain), "n_solo_ngram": len(solo_ng),
        "thinking_budget_hits": sum(1 for r in rows if budget and r["thinking"] >= budget),
        "budget_hit_ngram": sum(r["ngram_accepted"] for r in rows
                                if budget and r["thinking"] >= budget),
        "budget_hit_tokens": sum(r["completion"] for r in rows
                                 if budget and r["thinking"] >= budget),
        "matched_solo_median": median(msolo), "n_matched_solo": len(msolo),
        "cont_hit_rate": (sum(r["hit"] for r in cont) / max(1, sum(r["prompt"] for r in cont))),
        "cold_prefill_median": median([r["computed"] / r["prefill_s"] for r in cold if r["prefill_s"]]),
        "cold_prefill_aggregate": (sum(r["computed"] for r in cold) /
                                   max(1e-9, sum(r["prefill_s"] for r in cold))) if cold else None,
        "n_cold": len(cold),
        "output_tps": (sum(r["completion"] for r in dec) / sum(r["decode_s"] for r in dec)) if dec else None,
        "output_tps_copy": (sum(r["completion"] for r in copy) / sum(r["decode_s"] for r in copy)) if copy else None,
        "n_copy": len(copy),
        "solo_decode_median": median(solo), "n_solo": len(solo),
        "completion_tokens": sum(r["completion"] for r in rows),
        "thinking_tokens": sum(r["thinking"] for r in rows),
        "decode_seconds": sum(r["decode_s"] for r in rows),
        "spec_accept": (sum(r["spec_accepted"] for r in rows) /
                        max(1, sum(r["spec_drafted"] for r in rows))),
        "ngram_accepted": sum(r["ngram_accepted"] for r in rows),
        "ngram_drafted": sum(r["ngram_drafted"] for r in rows),
        "wall_seconds": A["meta"].get("wall_seconds"),
        "context_guards": sum(1 for e in A["events"] if e.get("event") == "context_guard"),
    }
    paths = {}
    for r in rows:
        d = paths.setdefault(r["path"], [0, 0])
        d[0] += 1
        d[1] += r["hit"]
    m["paths"] = paths
    pressure = {}
    for o in A["throughput"]:
        cc = o.get("context_cache") or {}
        for k, v in (cc.get("pressure") or {}).items():
            if isinstance(v, (int, float)):
                pressure[k] = pressure.get(k, 0) + v
        for kind in ("main_kv_transfers",):
            for d, t in ((cc.get(kind) or {}).items()):
                pressure["%s_%s_bytes" % (kind, d)] = pressure.get("%s_%s_bytes" % (kind, d), 0) + \
                    (t.get("bytes") or 0)
    m["pressure"] = pressure
    return m


# ---------------------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------------------

def f1(v):
    return "n/a" if v is None else "%.1f" % v


def f2(v):
    return "n/a" if v is None else "%.2f" % v


def ntok(v):
    return "n/a" if v is None else format(int(round(v)), ",d")


def ppct(v):
    return "n/a" if v is None else "%.1f %%" % (v * 100)


def chg(c, t):
    if c in (None, 0) or t is None:
        return "n/a"
    d = (t - c) / c * 100
    return "%s%.1f %%" % ("+" if d >= 0 else "−", abs(d))


def chg_pp(c, t):
    if c is None or t is None:
        return "n/a"
    d = (t - c) * 100
    return "%s%.1f points" % ("+" if d >= 0 else "−", abs(d))


def chg_abs(c, t):
    if c is None or t is None:
        return "n/a"
    d = t - c
    return "%s%d" % ("+" if d >= 0 else "−", abs(d))


def headline(mc, mt):
    L = ["| Metric | %s | %s | Change |" % (LABEL["control"], LABEL["treatment"]),
         "|---|---|---|---|"]

    def row(name, c, t, fmt, change):
        L.append("| %s | %s | %s | %s |" % (name, fmt(c), fmt(t), change(c, t)))

    row("Average time to first token (s, lower is better)", mc["ttft_mean"], mt["ttft_mean"], f1, chg)
    row("Median time to first token (s, lower is better)", mc["ttft_median"], mt["ttft_median"], f2, chg)
    row("90th-percentile time to first token (s, lower is better)", mc["ttft_p90"], mt["ttft_p90"], f1, chg)
    row("Average TTFT, continuing-session turns (s)", mc["ttft_cont_mean"], mt["ttft_cont_mean"], f2, chg)
    row("Average TTFT, new long prompts (s)", mc["ttft_new_mean"], mt["ttft_new_mean"], f1, chg)
    row("Prompt tokens served from cache", mc["hit_rate"], mt["hit_rate"], ppct, chg_pp)
    L.append("| Cached tokens (of %s / %s prompt tokens) | %s | %s | %s |"
             % (ntok(mc["prompt_tokens"]), ntok(mt["prompt_tokens"]), ntok(mc["hit_tokens"]),
                ntok(mt["hit_tokens"]), chg(mc["hit_tokens"], mt["hit_tokens"])))
    row("Prompt tokens prefilled (lower is better)", mc["computed_tokens"], mt["computed_tokens"], ntok, chg)
    L.append("| Main-session turns that re-prefilled the whole prompt (of %d) | %d | %d | %s |"
             % (mc["n_cont_main"], mc["cont_full_prefill_main"], mt["cont_full_prefill_main"],
                chg_abs(mc["cont_full_prefill_main"], mt["cont_full_prefill_main"])))
    L.append("| Subagent turns that re-prefilled the whole prompt (of %d) | %d | %d | %s |"
             % (mc["n_cont_sub"], mc["cont_full_prefill_sub"], mt["cont_full_prefill_sub"],
                chg_abs(mc["cont_full_prefill_sub"], mt["cont_full_prefill_sub"])))
    row("Prefill tok/s, requests with no cache hit (all sizes)", mc["cold_prefill_aggregate"],
        mt["cold_prefill_aggregate"], ntok, chg)
    row("Prefill tok/s, requests with no cache hit, 32K+ tokens", mc["cold_prefill_big"],
        mt["cold_prefill_big"], ntok, chg)
    row("Output tok/s while decoding (server, all lanes)", mc["server_decode_tps"],
        mt["server_decode_tps"], ntok, chg)
    row("Single-request decode tok/s, same requests decoding alone (median)",
        mc["matched_solo_median"], mt["matched_solo_median"], ntok, chg)
    row("Workload wall time (min, lower is better)",
        mc["wall_seconds"] and mc["wall_seconds"] / 60, mt["wall_seconds"] and mt["wall_seconds"] / 60,
        f1, chg)
    return "\n".join(L)


def cold_table(C, T, cold_seeds):
    by = {r["seed"]: r for r in C["requests"] if r.get("server")}
    bt = {r["seed"]: r for r in T["requests"] if r.get("server")}
    L = ["| Request | Class | Prompt tokens | Upstream prefill tok/s | Fork prefill tok/s | Change |",
         "|---|---|---|---|---|---|"]
    rows = sorted(((by[s], bt[s]) for s in cold_seeds), key=lambda p: p[0]["computed"])
    for c, t in rows:
        rc = c["computed"] / c["prefill_s"] if c["prefill_s"] else None
        rt = t["computed"] / t["prefill_s"] if t["prefill_s"] else None
        L.append("| %s | %s | %s | %s | %s | %s |" % (c["tag"], c["cls"], ntok(c["computed"]),
                                                     ntok(rc), ntok(rt), chg(rc, rt)))
    return "\n".join(L)


def class_table(C, T):
    L = ["| Class | n | Upstream avg TTFT (s) | Fork avg TTFT (s) | Upstream cached | Fork cached |",
         "|---|---|---|---|---|---|"]
    for cls, _ in CLASS_DOC:
        rc = [r for r in C["requests"] if r["cls"] == cls and r.get("server")]
        rt = [r for r in T["requests"] if r["cls"] == cls and r.get("server")]
        if not rc and not rt:
            continue
        hc = sum(r["hit"] for r in rc) / max(1, sum(r["prompt"] for r in rc))
        ht = sum(r["hit"] for r in rt) / max(1, sum(r["prompt"] for r in rt))
        L.append("| %s | %d | %s | %s | %s | %s |" % (cls, max(len(rc), len(rt)),
                                                     f2(mean([r["ttft"] for r in rc])),
                                                     f2(mean([r["ttft"] for r in rt])),
                                                     ppct(hc), ppct(ht)))
    return "\n".join(L)


def path_table(mc, mt):
    L = ["| Reuse path | Upstream (requests / cached tokens) | Fork (requests / cached tokens) |",
         "|---|---|---|"]
    for p in sorted(set(mc["paths"]) | set(mt["paths"])):
        c = mc["paths"].get(p, [0, 0])
        t = mt["paths"].get(p, [0, 0])
        L.append("| %s | %d / %s | %d / %s |" % (p, c[0], ntok(c[1]), t[0], ntok(t[1])))
    return "\n".join(L)


def pressure_table(mc, mt):
    keys = ["private_owners_evicted", "private_owners_degraded", "shared_owners_evicted",
            "shared_owners_degraded", "checkpoints_dropped", "maximal_fallback_selections",
            "search_budget_exhaustions", "spill_pages", "main_kv_transfers_d2h_bytes",
            "main_kv_transfers_h2d_bytes"]
    L = ["| Engine counter (summed over the run) | Upstream | Fork |", "|---|---|---|"]
    for k in keys:
        c, t = mc["pressure"].get(k), mt["pressure"].get(k)
        if c is None and t is None:
            continue
        fmt = (lambda v: "n/a" if v is None else "%.1f GiB" % (v / 2 ** 30)) if k.endswith("bytes") \
            else (lambda v: "n/a" if v is None else format(int(v), ",d"))
        L.append("| %s | %s | %s |" % (k, fmt(c), fmt(t)))
    return "\n".join(L)


def per_request(A):
    L = ["| Tag | Class | Prompt | Cached | Prefilled | Path | Queue s | TTFT s | Out | Decode tok/s | Finish |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(A["requests"], key=lambda r: r.get("t_send") or 0):
        if not r.get("server"):
            L.append("| %s | %s | | | | | | | | | %s |" % (r["tag"], r["cls"], r["status"]))
            continue
        L.append("| %s | %s%s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["tag"], r["cls"], " (copy)" if r["copy"] else "", ntok(r["prompt"]), ntok(r["hit"]),
            ntok(r["computed"]), r["path"], f1(r["queue_s"]), f2(r["ttft"]), ntok(r["completion"]),
            ntok(r["completion"] / r["decode_s"]) if r["decode_s"] else "n/a", r["finish_server"]))
    return "\n".join(L)


def server_row(A):
    s = A["start"] or {}
    e, mem = s.get("engine") or {}, s.get("memory") or {}
    cc = e.get("context_cache") or {}
    return ("KV capacity %s tokens; host state slots %s; host KV %.1f GiB; private "
            "continuations %s; long anchors per continuation %s; shared prefixes %s; build %s"
            % (ntok(e.get("kv_capacity")), cc.get("host_state_slots"),
               (mem.get("host_kv_capacity_bytes") or 0) / 2 ** 30, cc.get("max_private_continuations"),
               cc.get("max_long_anchors_per_continuation"), cc.get("max_shared_prefixes"),
               A["build"]))


def flag_str(flags):
    return " ".join(n + ("" if v is None else " " + (('"%s"' % v) if " " in v else v))
                    for n, v in flags)


def main(argv):
    run_dir = argv[0]
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    C, T = load_arm(run_dir, "control"), load_arm(run_dir, "treatment")
    # Matched cold set: requests with no cache hit in BOTH arms and a real prefill.
    dc = {r["seed"]: r for r in C["requests"] if r.get("server")}
    dt = {r["seed"]: r for r in T["requests"] if r.get("server")}
    cold = {s for s in dc if s in dt and dc[s]["hit"] == 0 and dt[s]["hit"] == 0
            and dc[s]["computed"] >= 4096 and dt[s]["computed"] >= 4096}
    msolo = solo_seeds(C) & solo_seeds(T)
    mc, mt = metrics(C, cold, msolo), metrics(T, cold, msolo)
    summary = {"config": cfg, "control": mc, "treatment": mt,
               "cold_seeds": sorted(cold)}
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    ctx = cfg.get("max_context")
    gpu = ((T["start"] or C["start"] or {}).get("environment") or {}).get("gpu_name", "GPU")
    L = []
    L.append("# Agentic A/B: upstream + Windows port vs this fork\n")
    L.append("Run directory: `%s`\n" % run_dir)
    L.append("- Model: `%s` on an %s, `--max-context %s` for both arms (launch bat: %s)."
             % (os.path.basename(cfg["model"]), gpu, ntok(ctx), ntok(cfg.get("bat_max_context"))))
    L.append("- Workload: %d requests per arm (seed %d, scale %.2f, corpus commit `%s`), "
             "`max_tokens: %d` on agent turns, thinking on."
             % (mc["n"], cfg["seed"], cfg["scale"], cfg["corpus_commit"], cfg["agent_max_tokens"]))
    L.append("- Control: `%s` - %s" % (cfg["control_exe"], server_row(C)))
    L.append("- Treatment: `%s` - %s" % (cfg["treatment_exe"], server_row(T)))
    L.append("- Total benchmark time including model loads: %.1f min.\n" % (cfg.get("total_seconds", 0) / 60))
    problems = []
    for name, m in (("control", mc), ("treatment", mt)):
        if m["n_failed"]:
            problems.append("%s: %d workload request(s) have no request_done record"
                            % (name, m["n_failed"]))
        if m["context_guards"]:
            problems.append("%s: the client context guard fired %d time(s)" % (name, m["context_guards"]))
    if problems:
        L.append("**Validity warnings:** " + "; ".join(problems) + "\n")

    L.append("## Headline\n")
    L.append(headline(mc, mt) + "\n")
    L.append("How to read it:\n")
    L.append("- Every row compares the same logical requests: the client tags each one with a "
             "seed that is identical in both arms. Observations (tool results, user messages, "
             "summaries) are byte-identical; assistant turns are each arm's own output fed back, "
             "as an agent client does, so prompt totals differ slightly (%s vs %s tokens)."
             % (ntok(mc["prompt_tokens"]), ntok(mt["prompt_tokens"])))
    L.append("- TTFT includes queueing behind other sessions (the serve runs 2 lanes and up to "
             "7 requests are in flight). Average queue wait: upstream %s s, fork %s s; average "
             "TTFT without queue wait: upstream %s s, fork %s s."
             % (f1(mc["queue_mean"]), f1(mt["queue_mean"]), f2(mc["service_ttft_mean"]),
                f2(mt["service_ttft_mean"])))
    L.append("- Continuing-session turns (%d) are tool-loop turns, retries, the post-idle and "
             "post-history-edit turns: cache retention decides their TTFT. New long prompts (%d) "
             "are resumed sessions, compaction and loop-check calls: prefill speed decides theirs."
             % (mc["n_cont"], mc["n_new"]))
    L.append("- \"No cache hit\" prefill rates are token-weighted (total prefilled tokens / "
             "total prefill time) over the %d requests that had no cache hit in *both* arms and "
             "prefilled at least 4,096 tokens (%d of them 32K+; per-request table below). The "
             "per-request median is %s vs %s tok/s: most of those requests are ~11-17K subagent "
             "prompts, where the prompt-attention kernel matters least."
             % (mt["n_cold"], mt["n_cold_big"], ntok(mc["cold_prefill_median"]),
                ntok(mt["cold_prefill_median"])))
    L.append("- Output while decoding is the tokens the server committed / the time any lane was "
             "decoding, from the serve's throughput records; it includes batching (mean decode "
             "batch: upstream %s, fork %s) and every ngram copy. Single-request decode is the "
             "median rate of the %d requests (>=128 output tokens) whose decode overlapped no "
             "other request in *both* arms, so it compares decode speed on the same turns "
             "without batching. Per-request completion / decode-time averages (upstream %s, fork "
             "%s tok/s; file-writing turns %s vs %s) are not in the table because batching "
             "lowers each request's own rate."
             % (f2(mc["mean_decode_batch"]), f2(mt["mean_decode_batch"]), mt["n_matched_solo"],
                ntok(mc["output_tps"]), ntok(mt["output_tps"]), ntok(mc["output_tps_copy"]),
                ntok(mt["output_tps_copy"])))
    L.append("- Output volume differs because the sampled text does: upstream %s completion "
             "tokens (%s thinking), fork %s (%s thinking); turns that used the whole thinking "
             "budget: upstream %d, fork %d (ngram copies supplied upstream %s of %s, fork %s of "
             "%s of those turns' tokens; a high share signals repetitive thinking). Longer "
             "outputs hold more KV and add batching, so they "
             "also shift cache pressure and queueing; compare repeated runs before attributing a "
             "thinking-length difference to a build."
             % (ntok(mc["completion_tokens"]), ntok(mc["thinking_tokens"]),
                ntok(mt["completion_tokens"]), ntok(mt["thinking_tokens"]),
                mc["thinking_budget_hits"], mt["thinking_budget_hits"],
                ntok(mc["budget_hit_ngram"]), ntok(mc["budget_hit_tokens"]),
                ntok(mt["budget_hit_ngram"]), ntok(mt["budget_hit_tokens"])))
    L.append("- Speculative acceptance: upstream %s, fork %s of drafted tokens; fork ngram "
             "drafting accepted %s of %s drafted tokens (%s of all fork output)."
             % (ppct(mc["spec_accept"]), ppct(mt["spec_accept"]), ntok(mt["ngram_accepted"]),
                ntok(mt["ngram_drafted"]),
                ppct(mt["ngram_accepted"] / max(1, mt["completion_tokens"]))))
    L.append("")
    L.append("## Prefill on requests with no cache hit\n")
    L.append(cold_table(C, T, cold) + "\n")
    L.append("## By request class\n")
    L.append(class_table(C, T) + "\n")
    L.append("Classes: " + "; ".join("`%s` %s" % c for c in CLASS_DOC) + ".\n")
    L.append("## Where cache hits came from\n")
    L.append(path_table(mc, mt) + "\n")
    L.append("## Cache pressure\n")
    L.append(pressure_table(mc, mt) + "\n")
    L.append("## Launch parameters\n")
    L.append("This fork:\n\n```text\n%s\n```\n" % flag_str(cfg.get("treatment_flags", [])))
    L.append("Upstream + Windows port (fork-only flags dropped: %s; the fork's `--host-cache-mib` "
             "is replaced by the explicit host-cache flags it resolved to):\n\n```text\n%s\n```\n"
             % (", ".join("`%s`" % d for d in cfg.get("dropped_for_control", [])),
                flag_str(cfg.get("control_flags", []))))
    L.append("## Per-request detail\n")
    L.append("### Upstream + Windows port\n\n" + per_request(C) + "\n")
    L.append("### This fork\n\n" + per_request(T) + "\n")
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(headline(mc, mt))
    print("report: %s" % os.path.join(run_dir, "report.md"))


if __name__ == "__main__":
    main(sys.argv[1:])
