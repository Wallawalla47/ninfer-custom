# Agentic A/B benchmark: upstream + Windows port vs this fork

A black-box A/B benchmark of two `ninfer-serve` builds serving the same model on the same GPU,
driven by a closed-loop replay of real agentic coding traffic over the OpenAI chat-completions
API. It produces a README-style comparison table covering:

- **prefix-cache hits** (tokens served from cache, continuing turns that had to re-prefill);
- **time to first token**, split into continuing-session turns (cache retention) and new long
  prompts (prefill speed);
- **raw prefill tok/s** on requests that had no cache hit in either arm;
- **output tok/s**, overall, on file-writing turns (where ngram copy drafting applies) and for
  requests that decoded alone.

Metrics come from each serve's own request log (`--request-log-jsonl`); the client only supplies
the request classification.

## Files

| File | Purpose |
|---|---|
| `workload.py` | Deterministic scenario: personas, observations and session timeline (`python workload.py` prints a summary). |
| `runner.py` | Calibrates a common `--max-context`, runs the treatment then the control, drives the workload, calls the analyzer. |
| `analyze.py` | `python analyze.py <run_dir>`: joins client and serve logs, writes `report.md` and `summary.json`. |
| `build_control.bat` | Windows build of the control's `ninfer-serve` from a control checkout. |

Each run writes to `profiles/bench/agentic_ab/<timestamp>/` (gitignored): `runner.log`,
`config.json`, `plan.json`, `report.md`, `summary.json`, and per arm `client.jsonl`,
`request_log.jsonl`, `serve.log`, `arm.json`.

## What the workload models

The shape comes from the deploy folder's production request logs (~3,300 requests from Qwen Code
and Claude Code style clients): prompts of 30K-210K tokens (median ~100K) that grow by a few
hundred to a few thousand tokens per turn, short tool-calling answers (median ~380 tokens,
p90 ~4,000) with thinking on, `max_tokens: 64000` on every agent turn, periodic compaction,
subagent fan-outs, whole-history side calls, retries, aborted requests, and several sessions
sharing one serve with up to eight requests in flight.

One run replays three interleaved sessions plus eight subagents (115 requests at scale 1.0):

| Actor | Persona | What it does |
|---|---|---|
| A | Claude-Code-like (20 tools, ~13K-token system+tools) | Resumes a ~85K-token C++ session (cold), runs a tool loop, fans out 4 explore subagents, loops on to ~125K, compacts, restarts from the summary, fans out 3 more subagents, finishes. |
| B | Qwen-Code-like (17 tools, ~11K) | Resumes a ~60K Python session, loops (a user retry, a loop-detection side call), sits idle while A compacts, comes back, loops, sends a second side call, has its older tool results cleared by the client (all but the last 20), loops. |
| C | Claude-Code-like | Resumes a ~25K tests/docs session, loops, receives a pasted ~16K-token CI log that the client aborts after 1.5 s and re-sends, launches a review subagent, loops. |
| explore-1..7, review-1 | Subagent personas (6 tools, ~12-15K shared prefix) | 4-5 read-only research turns each, final report. |

The three resumes run one after another before the sessions start interleaving, so they are
clean, isolated cold prefills of ~25K, ~55K and ~80K tokens. Four agent turns ask the model to
rewrite or edit a file it has just read (copy-heavy output).

**Closed loop, identical observations.** Every observation the client sends (tool results, user
messages, subagent reports, the post-compaction summary) is generated before the run from real
files, diffs and history of this repository at a pinned commit (`AB_CORPUS_COMMIT`, default
`e48a0d28`), so both arms receive byte-identical observations. The assistant turns are each
arm's own streamed output (reasoning, content and tool calls), fed back as an agent client does;
that is what makes the engine's private-endpoint reuse behave as in production. Observations
answer the model's tool calls by id whatever it asked for, so every arm sees the same token
growth. Each request carries a `seed` that is identical in both arms; it also tags the request
in the serve log.

What differs between arms is only what should: the model's sampled text (the kernels differ
numerically, so outputs diverge after a few tokens), and therefore interleaving and queueing.
Rates, cache fractions and per-class averages are comparable; absolute completion totals are not.

## Fairness

- **Control = upstream at the commit the fork has merged, plus only the Windows port.** A
  control built from an older upstream would credit the fork with upstream's own newer work.
- **Same launch configuration.** The treatment runs the deploy folder's launch bat
  (`AB_LAUNCH_BAT`, default the official-artifact launcher) plus `AB_TREATMENT_EXTRA_FLAGS`
  (default `--fast-prefill-kernel`, from the production launcher). The control runs the same
  flags minus the ones its `--help` does not advertise.
- **Same host RAM.** Upstream has no `--host-cache-mib`. The runner reads the split the fork
  resolved at startup (host state slots, host KV bytes, private continuations, long anchors,
  shared prefixes) and passes the control those exact values as explicit flags.
- **Same context.** Upstream keeps a fixed 1 GiB of VRAM spare under `--kv-capacity auto`, so it
  may not start at the bat's context. The runner tries the bat's value first, then 200000,
  180000, 170000, 160000, and runs both arms at the first one the control starts with. The
  fork's larger device KV at that context (`--kv-headroom-mib 0`,
  `--cuda-graph-allowance-mib 500`) is part of what is being compared and is shown in the
  report header.

## Running it

1. Stop any server on the port and let the GPU go idle (the runner checks both and refuses
   otherwise; it never starts or stops a production server).
2. Build the control:

   ```bat
   git worktree add -b ab/upstream-windows-port <dir>\src origin/master
   git -C <dir>\src cherry-pick <windows-port-commit>
   set AB_CONTROL_SRC=<dir>\src
   set AB_CONTROL_BUILD=<dir>\build
   build_control.bat configure
   build_control.bat build
   ```

3. Run both arms (about 30 minutes on an RTX 5090, including model loads):

   ```bat
   set AB_CONTROL_EXE=<dir>\build\apps\Release\ninfer-serve.exe
   python runner.py
   ```

   `--arms treatment` runs one arm, `--ctx N` skips calibration, `--scale F` stretches or
   shrinks the session loops (0.3 is a quick smoke run), `--dry-run` prints the plan and flags.
   `python analyze.py <run_dir>` re-renders a report.

| Variable | Default |
|---|---|
| `AB_LAUNCH_BAT` | `<AB_DEPLOY>\LaunchQwen3.8-27B-official-dflash2-ngram.bat` |
| `AB_DEPLOY` | `E:\NInfer-Deploy-V3` |
| `AB_MODEL` | the model path in the launch bat |
| `AB_TREATMENT_EXE` | `build-windows\apps\Release\ninfer-serve.exe` in this checkout |
| `AB_CONTROL_EXE` | `bench\agentic_ab\control\build\apps\Release\ninfer-serve.exe` |
| `AB_TREATMENT_EXTRA_FLAGS` | `--fast-prefill-kernel` |
| `AB_HOST` / `AB_PORT` | `127.0.0.1` / `8080` |
| `AB_OUT` | `profiles\bench\agentic_ab` in this checkout |
| `AB_CORPUS_REPO` / `AB_CORPUS_COMMIT` | this checkout / `e48a0d28` |

A run is marked invalid (non-zero exit, warning in the report) if any workload request fails.
The report also flags when the client's context guard had to clear old tool results to keep a
prompt 24K tokens under `--max-context`.

## Reading the report

- **TTFT** includes queueing: two lanes serve up to seven requests in flight. The report also
  gives the average queue wait and TTFT without it.
- **Continuing-session turns** are tool-loop turns, retries, the post-idle turn, the
  post-history-edit turn and the abort retry; cache retention decides their TTFT. **New long
  prompts** are the resumes, compaction and side calls; prefill speed decides theirs.
- **Re-prefilled turns** are split into main-session and subagent turns: losing a 100K main
  session costs far more tokens than losing a 15K subagent, so the two tell different stories.
- **Prefill tok/s with no cache hit** is token-weighted (total prefilled tokens over total
  prefill time) over requests that had no hit in either arm and prefilled at least 4,096 tokens,
  with a separate 32K+ row; the per-request table shows them by size.
- **Output tok/s while decoding** is what the server committed over the time any lane was
  decoding (from the throughput records), so it includes batching and ngram copies.
  **Single-request decode** is the median rate over requests that decoded alone in *both* arms:
  the same turns, without batching. Per-request completion over decode time is reported in the
  notes only, because batching lowers each request's own rate.
- **Output volume** is sampled, not controlled: the notes give each arm's completion and thinking
  totals and how many turns ran into the thinking budget. A few runaway turns shift cache
  pressure, batching and wall time, so compare repeated runs before attributing them to a build.
- **Where cache hits came from** and **cache pressure** (eviction, degradation, spill and
  host/device transfer counters summed from the serve's throughput records) explain the cache
  rows.
