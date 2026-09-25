#!/usr/bin/env python3
"""Agentic A/B benchmark runner: upstream + Windows port (control) vs this fork (treatment).

Serves each build in turn on the same GPU with the same model, replays the closed-loop
agentic workload from workload.py against it over the OpenAI chat-completions API, and
writes a comparison report (analyze.py). Metrics come from each serve's own request log
(--request-log-jsonl); every request carries a per-request `seed` that is identical in
both arms, which is how client requests and log records are joined.

Sequence:
  1. calibrate: the largest --max-context (starting at the launch bat's own value) at which
     the control serve starts; both arms run at it;
  2. treatment arm: the launch bat's flags plus AB_TREATMENT_EXTRA_FLAGS;
  3. control arm: the same flags minus those the control's --help does not advertise; the
     fork's single --host-cache-mib ceiling is translated into the control's explicit
     --host-state-slots / --host-kv-mib / catalog limits using the split the treatment
     resolved at startup, so both arms get the same pinned host RAM and catalog sizes;
  4. analyze.py writes report.md / summary.json into the run directory.

Usage: python runner.py [--arms treatment,control] [--ctx N] [--scale F] [--out DIR]
                        [--dry-run]
Paths come from AB_* environment variables (see README.md). The runner never starts or
stops a production server; stop it before running.
"""
import argparse
import copy
import http.client
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

import workload

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DEPLOY = os.environ.get("AB_DEPLOY", r"E:\NInfer-Deploy-V3")
LAUNCH_BAT = os.environ.get("AB_LAUNCH_BAT",
                            os.path.join(DEPLOY, "LaunchQwen3.8-27B-official-dflash2-ngram.bat"))
MODEL = os.environ.get("AB_MODEL")  # default: the model path in the launch bat
TREATMENT_EXE = os.environ.get("AB_TREATMENT_EXE",
                               os.path.join(REPO, "build-windows", "apps", "Release",
                                            "ninfer-serve.exe"))
CONTROL_EXE = os.environ.get("AB_CONTROL_EXE",
                             os.path.join(HERE, "control", "build", "apps", "Release",
                                          "ninfer-serve.exe"))
# Fork flags the launch bat does not already carry. --fast-prefill-kernel is in the
# production (nvidia) launcher and is part of what this A/B measures.
TREATMENT_EXTRA_FLAGS = os.environ.get("AB_TREATMENT_EXTRA_FLAGS", "--fast-prefill-kernel").split()
HOST = os.environ.get("AB_HOST", "127.0.0.1")
PORT = int(os.environ.get("AB_PORT", "8080"))
OUT_ROOT = os.environ.get("AB_OUT", os.path.join(REPO, "profiles", "bench", "agentic_ab"))
# Context candidates below the bat's own value, tried in order when the control cannot start.
CTX_FALLBACKS = [200000, 180000, 170000, 160000]
AGENT_MAX_TOKENS = 64000   # what the production clients request on every agent turn
REQUEST_TIMEOUT_S = 1200
CONTEXT_GUARD_TOKENS = 24000  # keep prompts this far below --max-context
CREATE_NEW_PROCESS_GROUP = 0x00000200

_log_lock = threading.Lock()
_log_path = None


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    with _log_lock:
        print(line, flush=True)
        if _log_path:
            with open(_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


# ---------------------------------------------------------------------------------------
# Launch flags
# ---------------------------------------------------------------------------------------

def parse_bat(path):
    """(model_path, [(flag, value|None), ...]) from the ninfer-serve line of a launch bat."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "ninfer-serve" in line and not line.lstrip().lower().startswith("rem"):
                toks = [a if a else b for a, b in re.findall(r'"([^"]*)"|(\S+)', line)]
                model = toks[1]
                flags, i = [], 2
                while i < len(toks):
                    name = toks[i]
                    val = None
                    if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                        val = toks[i + 1]
                        i += 1
                    flags.append((name, val))
                    i += 1
                return model, flags
    raise SystemExit("no ninfer-serve command line in %s" % path)


def help_flags(exe):
    if not os.path.exists(exe):
        raise SystemExit("missing serve executable: %s (set AB_CONTROL_EXE / AB_TREATMENT_EXE)"
                         % exe)
    out = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=60)
    return set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9-]*", (out.stdout or "") + (out.stderr or "")))


def set_flag(flags, name, value):
    out = [f for f in flags if f[0] != name]
    out.append((name, value))
    return out


def without(flags, names):
    return [f for f in flags if f[0] not in names]


def flag_str(flags):
    return " ".join(n + ("" if v is None else " " + (('"%s"' % v) if " " in v else v))
                    for n, v in flags)


def host_translation(server_start):
    """The control's explicit host-cache flags equal to the treatment's resolved split."""
    cc = server_start["engine"]["context_cache"]
    mem = server_start["memory"]
    return [("--host-state-slots", str(cc["host_state_slots"])),
            ("--host-kv-mib", str(mem["host_kv_capacity_bytes"] // (1 << 20))),
            ("--max-private-continuations", str(cc["max_private_continuations"])),
            ("--max-long-anchors-per-continuation", str(cc["max_long_anchors_per_continuation"])),
            ("--max-shared-prefixes", str(cc["max_shared_prefixes"]))]


# ---------------------------------------------------------------------------------------
# Serve lifecycle
# ---------------------------------------------------------------------------------------

def port_open():
    try:
        with socket.create_connection((HOST, PORT), timeout=1):
            return True
    except OSError:
        return False


def gpu_memory_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def wait_gpu_idle(timeout=240, threshold=6000):
    deadline = time.time() + timeout
    mib = None
    while time.time() < deadline:
        mib = gpu_memory_mib()
        if mib is not None and mib < threshold:
            return
        time.sleep(3)
    raise SystemExit("GPU still busy (%s MiB in use); stop the other server first" % mib)


def read_server_start(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            try:
                o = json.loads(ln)
            except ValueError:
                continue
            if o.get("event") == "server_start":
                return o
    return None


class Serve:
    def __init__(self, exe, model, flags, request_log, serve_log):
        self.exe, self.model, self.flags = exe, model, flags
        self.request_log, self.serve_log = request_log, serve_log
        self.proc = None

    def start(self, load_timeout=600):
        if port_open():
            raise SystemExit("port %d is already in use; stop the other server first" % PORT)
        wait_gpu_idle()
        if os.path.exists(self.request_log):
            os.remove(self.request_log)
        args = [self.exe, self.model]
        for n, v in self.flags:
            args += [n] if v is None else [n, v]
        args += ["--request-log-jsonl", self.request_log]
        log("launch: %s" % subprocess.list2cmdline(args))
        with open(self.serve_log, "w", encoding="utf-8") as sf:
            self.proc = subprocess.Popen(args, cwd=os.path.dirname(self.request_log), stdout=sf,
                                         stderr=subprocess.STDOUT,
                                         creationflags=CREATE_NEW_PROCESS_GROUP)
        deadline = time.time() + load_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("serve exited during startup (rc=%s); see %s"
                                   % (self.proc.returncode, self.serve_log))
            try:
                with urllib.request.urlopen("http://%s:%d/v1/models" % (HOST, PORT), timeout=5) as r:
                    data = json.loads(r.read().decode("utf-8"))
                if data.get("data"):
                    self.model_id = data["data"][0]["id"]
                    log("serve ready (pid %d, model %s)" % (self.proc.pid, self.model_id))
                    return self
            except Exception:
                pass
            time.sleep(2)
        self.stop()
        raise RuntimeError("serve not ready within %ds; see %s" % (load_timeout, self.serve_log))

    def stop(self):
        if self.proc and self.proc.poll() is None:
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                           capture_output=True)
            try:
                self.proc.wait(timeout=90)
            except Exception:
                self.proc.kill()
        wait_gpu_idle()


# ---------------------------------------------------------------------------------------
# Client: one closed-loop agent per actor, as an agent client drives the API.
# ---------------------------------------------------------------------------------------

def flatten(history):
    """The conversation as text, as compaction and loop-check side calls send it."""
    out = []
    for m in history[1:]:
        role = m["role"]
        if m.get("content"):
            out.append("[%s]\n%s" % (role, m["content"]))
        for tc in m.get("tool_calls") or []:
            out.append("[%s -> %s]\n%s" % (role, tc["function"]["name"], tc["function"]["arguments"]))
    return "\n\n".join(out)


COMPACT_INSTRUCTION = (
    "\n\nYour task is to create a detailed summary of the conversation above so the agent can "
    "continue in a fresh context. Cover: the user's requests and intent, key technical concepts, "
    "files and code sections examined or changed (with paths), errors and how they were fixed, "
    "problem solving so far, pending tasks, the current work, and the next step. Keep it under "
    "700 words and use headings.")

CHECK_INSTRUCTION = (
    "\n\nAnalyze the conversation above. Is the agent in an unproductive loop, and who should "
    "speak next? Respond with the JSON object only.")

RESUME_AFTER_COMPACT = (
    "This session is being continued from a previous conversation that ran out of context. "
    "The conversation is summarized below:\n\n%s\n\nPlease continue the conversation from where "
    "we left it off without asking the user any further questions. Continue with the last task "
    "that you were asked to work on.")


class Result:
    def __init__(self):
        self.status = "error"
        self.content, self.reasoning = "", ""
        self.tool_calls = []
        self.finish = None
        self.usage = None
        self.error = None
        self.t_send = self.t_first = self.t_done = None

    def assistant_message(self):
        msg = {"role": "assistant", "content": self.content}
        if self.reasoning:
            msg["reasoning_content"] = self.reasoning
        if self.tool_calls:
            msg["tool_calls"] = [{"id": tc["id"], "type": "function",
                                  "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                                 for tc in self.tool_calls]
        return msg


class ArmClient:
    def __init__(self, arm, plan, model_id, out_dir, max_context):
        self.arm, self.plan, self.model_id = arm, plan, model_id
        self.max_context = max_context
        self.personas = plan["personas"]
        self.actors = {a["name"]: a for a in plan["actors"]}
        self.signals = {}
        self.sig_lock = threading.Lock()
        self.client_log = os.path.join(out_dir, "client.jsonl")
        self.write_lock = threading.Lock()
        self.failures = []

    def signal(self, name):
        with self.sig_lock:
            return self.signals.setdefault(name, threading.Event())

    def record(self, row):
        with self.write_lock:
            with open(self.client_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    # -- HTTP --------------------------------------------------------------------------
    def post(self, messages, tools, max_tokens, seed, abort_after=None):
        body = {"model": self.model_id, "messages": messages, "max_tokens": max_tokens,
                "temperature": 1.0, "top_p": 0.95, "seed": seed, "stream": True,
                "stream_options": {"include_usage": True}}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        data = json.dumps(body).encode("utf-8")
        r = Result()
        conn = http.client.HTTPConnection(HOST, PORT, timeout=REQUEST_TIMEOUT_S)
        timer = None
        calls = {}
        try:
            r.t_send = time.time()
            conn.request("POST", "/v1/chat/completions", body=data,
                         headers={"Content-Type": "application/json"})
            if abort_after:
                def cut():
                    try:
                        conn.sock.shutdown(socket.SHUT_RDWR)
                    except Exception:
                        pass
                timer = threading.Timer(abort_after, cut)
                timer.start()
            resp = conn.getresponse()
            if resp.status != 200:
                r.error = "HTTP %d: %s" % (resp.status, resp.read()[:600].decode("utf-8", "replace"))
                return r
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                ev = json.loads(payload)
                if ev.get("usage"):
                    r.usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")) \
                            and r.t_first is None:
                        r.t_first = time.time()
                    r.content += d.get("content") or ""
                    r.reasoning += d.get("reasoning_content") or ""
                    for tc in d.get("tool_calls") or []:
                        e = calls.setdefault(tc.get("index", len(calls)),
                                             {"id": None, "name": "", "arguments": ""})
                        e["id"] = tc.get("id") or e["id"]
                        fn = tc.get("function") or {}
                        e["name"] += fn.get("name") or ""
                        e["arguments"] += fn.get("arguments") or ""
                    if ch.get("finish_reason"):
                        r.finish = ch["finish_reason"]
            r.tool_calls = [calls[k] for k in sorted(calls)]
            r.status = "ok"
        except Exception as e:  # noqa: BLE001 - every failure is recorded, never raised
            r.status = "aborted" if abort_after else "error"
            r.error = "%s: %s" % (type(e).__name__, e)
        finally:
            r.t_done = time.time()
            if timer:
                timer.cancel()
            conn.close()
        return r

    def request(self, actor, step, messages, tools, max_tokens, seed, cls, abort_after=None):
        res = self.post(messages, tools, max_tokens, seed, abort_after)
        row = {"arm": self.arm, "actor": actor["name"], "tag": step.get("tag"), "cls": cls,
               "copy": bool(step.get("copy")), "seed": seed, "status": res.status,
               "error": res.error, "finish": res.finish, "usage": res.usage,
               "t_send": res.t_send, "t_first": res.t_first, "t_done": res.t_done,
               "messages": len(messages), "tool_calls": len(res.tool_calls),
               "content_chars": len(res.content), "reasoning_chars": len(res.reasoning),
               "tool_argument_chars": sum(len(t["arguments"]) for t in res.tool_calls)}
        self.record(row)
        wall = (res.t_done - res.t_send) if res.t_send else 0
        if res.status == "ok":
            u = res.usage or {}
            log("  [%s] %-11s %-14s ok  %5.1fs prompt=%s out=%s calls=%d finish=%s"
                % (self.arm[0], step.get("tag"), cls, wall, u.get("prompt_tokens"),
                   u.get("completion_tokens"), len(res.tool_calls), res.finish))
        else:
            log("  [%s] %-11s %-14s %s %5.1fs %s" % (self.arm[0], step.get("tag"), cls,
                                                      res.status.upper(), wall, res.error))
            if res.status == "error":
                self.failures.append((step.get("tag"), res.error))
        return res

    # -- conversation handling -----------------------------------------------------------
    @staticmethod
    def deliver(hist, last_calls, obs, user):
        """Append the next observation the way an agent client does."""
        if last_calls:
            for i, tc in enumerate(last_calls):
                hist.append({"role": "tool", "tool_call_id": tc["id"] or "call_missing",
                             "content": (obs if obs is not None else "(no output)") if i == 0
                             else "(no additional output)"})
            if user:
                hist.append({"role": "user", "content": user})
        else:
            text = "\n\n".join(t for t in (obs, user) if t)
            if text:
                hist.append({"role": "user", "content": text})
            elif hist[-1]["role"] == "assistant":
                hist.append({"role": "user", "content": "Continue."})

    def guard(self, actor, hist, calib):
        """Clear old tool results if the next prompt would come too close to --max-context,
        as agent clients do on their own; logged so a run that needed it is visible."""
        limit = self.max_context - CONTEXT_GUARD_TOKENS
        est = int((workload.messages_tokens(hist) +
                   workload.tools_tokens(self.personas[actor["persona"]]["tools"])) * calib)
        if est > limit:
            n = clear_old_tool_results(hist, 6)
            log("  [%s] %s context guard: ~%d tokens > %d, cleared %d old tool results"
                % (self.arm[0], actor["name"], est, limit, n))
            self.record({"arm": self.arm, "actor": actor["name"], "event": "context_guard",
                         "estimate": est, "cleared": n})

    def run_actor(self, actor):
        try:
            self._run_actor(actor)
        except Exception:
            self.failures.append((actor["name"], traceback.format_exc()))
            log("actor %s crashed:\n%s" % (actor["name"], traceback.format_exc()))

    def _run_actor(self, actor):
        persona = self.personas[actor["persona"]]
        tools = persona["tools"]
        hist = copy.deepcopy(actor["initial"])
        last_calls = []
        last_sent = None
        calib = 1.0
        for step in actor["steps"]:
            op = step["op"]
            if op == "signal":
                self.signal(step["name"]).set()
            elif op == "wait":
                self.signal(step["signal"]).wait()
            elif op == "spawn":
                subs = [threading.Thread(target=self.run_actor, args=(self.actors[n],), daemon=True)
                        for n in step["actors"]]
                for t in subs:
                    t.start()
                for t in subs:
                    t.join()
            elif op == "request":
                self.deliver(hist, last_calls, step.get("obs"), step.get("user"))
                time.sleep(step.get("delay", 0) + step.get("user_delay", 0))
                self.guard(actor, hist, calib)
                cls = step["cls"]
                if step.get("abort_after"):
                    self.request(actor, step, hist, tools, AGENT_MAX_TOKENS, step["abort_seed"],
                                 "aborted", abort_after=step["abort_after"])
                    time.sleep(step.get("retry_delay", 3.0))
                    cls = "abort_retry"
                last_sent = copy.deepcopy(hist)
                res = self.request(actor, step, hist, tools, step.get("max_tokens", AGENT_MAX_TOKENS),
                                   step["seed"], cls)
                if res.status == "ok":
                    hist.append(res.assistant_message())
                    last_calls = res.tool_calls
                    if res.usage and res.usage.get("prompt_tokens"):
                        est = (workload.messages_tokens(last_sent) +
                               workload.tools_tokens(tools))
                        calib = max(0.5, min(2.0, res.usage["prompt_tokens"] / max(1, est)))
                else:
                    last_calls = []
            elif op == "retry":
                # The user retries the previous turn: the identical prompt is sent again and
                # the new answer replaces the old one.
                if last_sent is None:
                    continue
                time.sleep(step.get("delay", 0))
                res = self.request(actor, step, last_sent, tools, AGENT_MAX_TOKENS, step["seed"],
                                   step["cls"])
                if res.status == "ok":
                    hist = copy.deepcopy(last_sent)
                    hist.append(res.assistant_message())
                    last_calls = res.tool_calls
            elif op == "side_call":
                p = self.personas[step["persona"]]
                msgs = [{"role": "system", "content": p["system"]},
                        {"role": "user", "content": flatten(hist) + CHECK_INSTRUCTION}]
                self.request(actor, step, msgs, p["tools"], step["max_tokens"], step["seed"],
                             step["cls"])
            elif op == "compact":
                p = self.personas["compact"]
                msgs = [{"role": "system", "content": p["system"]},
                        {"role": "user", "content": flatten(hist) + COMPACT_INSTRUCTION}]
                self.request(actor, step, msgs, p["tools"], step["max_tokens"], step["seed"],
                             step["cls"])
                # The scripted summary (identical in both arms) seeds the fresh context.
                hist = [hist[0], {"role": "user", "content": RESUME_AFTER_COMPACT % step["summary"]}]
                last_calls = []
            elif op == "clear_old_tool_results":
                n = clear_old_tool_results(hist, step["keep_last"])
                self.record({"arm": self.arm, "actor": actor["name"], "event": "history_edit",
                             "cleared": n})
            else:
                raise ValueError("unknown op %r" % op)

    def run(self):
        tops = [a for a in self.plan["actors"] if a.get("top")]
        threads = [threading.Thread(target=self.run_actor, args=(a,), daemon=True) for a in tops]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return time.time() - t0


def clear_old_tool_results(hist, keep_last):
    idx = [i for i, m in enumerate(hist) if m["role"] == "tool"]
    n = 0
    for i in idx[:-keep_last] if keep_last else idx:
        if hist[i]["content"] != "[Old tool result content cleared]":
            hist[i] = dict(hist[i], content="[Old tool result content cleared]")
            n += 1
    return n


# ---------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------

def calibrate_ctx(model, ctrl_flags, bat_ctx, run_dir):
    """Largest context (bat value first) at which the control starts; both arms use it."""
    for ctx in [bat_ctx] + [c for c in CTX_FALLBACKS if c < bat_ctx]:
        probe = Serve(CONTROL_EXE, model, set_flag(ctrl_flags, "--max-context", str(ctx)),
                      os.path.join(run_dir, "calibration_%d.jsonl" % ctx),
                      os.path.join(run_dir, "calibration_%d_serve.log" % ctx))
        log("calibration: control at --max-context %d" % ctx)
        try:
            probe.start()
        except RuntimeError as e:
            log("  control does not start at %d: %s" % (ctx, e))
            continue
        probe.stop()
        log("  control starts at %d; both arms run at this context" % ctx)
        return ctx
    raise SystemExit("control failed to start at every candidate context")


def run_arm(name, exe, model, flags, plan, run_dir, ctx):
    arm_dir = os.path.join(run_dir, name)
    os.makedirs(arm_dir, exist_ok=True)
    log("=== arm %s: %s" % (name, exe))
    serve = Serve(exe, model, flags, os.path.join(arm_dir, "request_log.jsonl"),
                  os.path.join(arm_dir, "serve.log")).start()
    start = read_server_start(serve.request_log)
    wall, client = None, None
    try:
        client = ArmClient(name, plan, serve.model_id, arm_dir, ctx)
        # One trivial request primes the arm (CUDA graphs, allocator); it is excluded from
        # the metrics because it carries no workload seed.
        client.post([{"role": "user", "content": "Reply with the single word: ok"}], [], 8, 1)
        wall = client.run()
        log("=== arm %s: workload finished in %.0fs (%d client failures)"
            % (name, wall, len(client.failures)))
    finally:
        serve.stop()
    with open(os.path.join(arm_dir, "arm.json"), "w", encoding="utf-8") as f:
        json.dump({"arm": name, "exe": exe, "flags": flags, "wall_seconds": wall,
                   "failures": client.failures if client else []}, f, indent=1)
    return start, (client.failures if client else [("arm", "did not run")])


def main():
    global _log_path
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", default="treatment,control")
    ap.add_argument("--ctx", type=int, help="skip calibration and run both arms at this context")
    ap.add_argument("--scale", type=float, default=1.0, help="stretch the session loop lengths")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", help="run directory (default: AB_OUT/<timestamp>)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and flags only")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    run_dir = args.out or os.path.join(OUT_ROOT, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    _log_path = os.path.join(run_dir, "runner.log")

    bat_model, bat_flags = parse_bat(LAUNCH_BAT)
    model = MODEL or bat_model
    bat_ctx = int(dict(bat_flags).get("--max-context") or 220000)
    base = without(bat_flags, {"--request-log-jsonl"})
    base = set_flag(set_flag(base, "--host", HOST), "--port", str(PORT))
    treat_flags = list(base)
    for extra in TREATMENT_EXTRA_FLAGS:
        if extra not in dict(treat_flags):
            treat_flags.append((extra, None))
    ctrl_supported = help_flags(CONTROL_EXE)
    ctrl_flags = [f for f in base if f[0] in ctrl_supported]
    dropped = [f[0] for f in treat_flags if f[0] not in ctrl_supported]

    plan = workload.build_plan(seed=args.seed, scale=args.scale)
    log("plan: " + workload.summarize(plan).splitlines()[-1])
    log("model: %s" % model)
    log("treatment flags: %s" % flag_str(treat_flags))
    log("control drops (not in its --help): %s" % ", ".join(dropped))
    config = {"launch_bat": LAUNCH_BAT, "model": model, "treatment_exe": TREATMENT_EXE,
              "control_exe": CONTROL_EXE, "treatment_flags": treat_flags,
              "control_flags_base": ctrl_flags, "dropped_for_control": dropped,
              "seed": args.seed, "scale": args.scale, "corpus_commit": plan["corpus_commit"],
              "agent_max_tokens": AGENT_MAX_TOKENS}
    if args.dry_run:
        print(workload.summarize(plan))
        print("control flags:", flag_str(ctrl_flags))
        return
    for exe in ([TREATMENT_EXE] if "treatment" in arms else []) + \
               ([CONTROL_EXE] if "control" in arms else []):
        if not os.path.exists(exe):
            raise SystemExit("missing serve executable: %s" % exe)
    with open(os.path.join(run_dir, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f)

    t0 = time.time()
    ctx = args.ctx or (calibrate_ctx(model, without(ctrl_flags, {"--max-context"}), bat_ctx, run_dir)
                       if "control" in arms else bat_ctx)
    config["max_context"] = ctx
    config["bat_max_context"] = bat_ctx
    failures = []
    starts = {}
    if "treatment" in arms:
        flags = set_flag(treat_flags, "--max-context", str(ctx))
        config["treatment_flags"] = flags
        starts["treatment"], f = run_arm("treatment", TREATMENT_EXE, model, flags, plan, run_dir, ctx)
        failures += f
    if "control" in arms:
        flags = set_flag(ctrl_flags, "--max-context", str(ctx))
        if "--host-cache-mib" in dict(treat_flags) and "--host-cache-mib" not in ctrl_supported:
            if starts.get("treatment") is None:
                raise SystemExit("the control's host-cache translation needs the treatment's "
                                 "resolved split; run the treatment arm first")
            for n, v in host_translation(starts["treatment"]):
                flags = set_flag(flags, n, v)
            log("control host cache = treatment's resolved --host-cache-mib split: %s"
                % flag_str(host_translation(starts["treatment"])))
        config["control_flags"] = flags
        starts["control"], f = run_arm("control", CONTROL_EXE, model, flags, plan, run_dir, ctx)
        failures += f
    config["total_seconds"] = time.time() - t0
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=1)
    log("all arms finished in %.1f min" % (config["total_seconds"] / 60))

    import analyze
    analyze.main([run_dir])
    if failures:
        log("RUN INVALID: %d client request failure(s): %s" % (len(failures), failures[:5]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
