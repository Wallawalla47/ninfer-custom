# NInfer — local fork

> **AI disclaimer:** Everything added to this fork, including this README, was written with AI
> (mostly Qwen3.8-27B running on NInfer, plus a few other AI systems). It is likely to be
> neither complete nor entirely accurate. This is hobby development.

This is a personal fork of [Neroued/ninfer](https://github.com/Neroued/ninfer), kept on the
`master` branch. It follows upstream closely and adds changes on top. The sections below explain
what is different, grouped by topic, with credit given where a change came from someone else. The
upstream README follows, copied unchanged, under the "Upstream README" heading.

**The short version.** Compared with upstream, this fork:

- reuses cached prompt prefixes far more often in long multi-turn agent sessions, which cuts
  average time-to-first-token by about three quarters on that kind of workload (by about half
  with the original cache);
- has an optional faster prefill kernel for long prompts (`--fast-prefill-kernel`);
- uses a hybrid prefix cache by default that shares KV between conversations by content and keeps
  it in GPU memory, host RAM and optionally on disk (outlined just below); upstream's checkpoint
  catalog, with this fork's fixes, remains available with `--use-original-prefix-caching`;
- lets ngram copy drafting run with more than one concurrent request;
- decodes about 2-2.5 % faster per speculative round, with the same output, by overlapping each
  decode kernel's launch and weight loading with the kernel before it;
- accepts more tool-call formats and API options used by agent clients such as Claude Code, Qwen
  Code, Codex and Zed;
- builds and runs natively on Windows;
- recovers from out-of-memory and planner errors instead of stopping the whole engine.

## The hybrid prefix cache at a glance

The fork's default prefix cache replaces upstream's checkpoint catalog (still available with
`--use-original-prefix-caching`) with a cache designed around how Qwen3.5-family models work.
Most of their layers are linear-attention (GDN) layers, whose recurrent state cannot be rebuilt
from the KV cache. So resuming a prompt needs two things: the KV
of every earlier token, and a saved state at the exact token where the new prompt continues. The
cache keeps those two things apart and stores each as cheaply as it can. Add `--host-cache-mib N`
for host RAM (default 8192) and, optionally, `--prefix-cache-file PATH` to keep the cache across
restarts; everything else is sized automatically.

**What it stores**

- **KV blocks keyed by content.** KV is kept in 64-token blocks in a radix tree. A block is
  identified by its tokens (and any image it contains) plus the block before it, so the same
  prefix is stored once however many conversations use it.
- **Sparse state snapshots.** A snapshot is the model's recurrent state at one token position,
  anchored on the block path that leads to it. Snapshots are taken only at planned points:
  - *exact* points split the prefill: the end of the system prompt and tools, client cache
    breakpoints, and the start of the assistant reply;
  - *flexible* points cost nothing because they fall on prefill chunk boundaries: the end of the
    prompt, and a few points spread back through long history;
  - an *endpoint* snapshot at the end of each answer, which the next turn resumes from when the
    client echoes the conversation back exactly.

**Where it keeps them**

| Tier | Holds | Evicts |
|---|---|---|
| GPU | free VRAM after the model becomes block cache, plus a few snapshot slots | least recently used |
| Host RAM | one pinned pool (`--host-cache-mib`) shared by blocks and snapshots | by prefill time saved per byte, dead KV first |
| Disk (optional) | the host tier, saved on shutdown and reloaded at startup | replaced on each save |

A block leaving the GPU is copied to host RAM first when it is worth keeping, so GPU eviction
usually only drops a copy.

**How a request uses it**

1. Admission walks the tree to the longest cached block path and picks the deepest usable
   snapshot on it. A cost model chooses between restoring from host RAM and prefilling.
2. The request reserves its GPU pages and a state slot. Anything held only in host RAM is copied
   back on a separate stream in layer order. The request starts at once, and each layer waits only
   for its own data.
3. Prefill runs from the snapshot, taking new snapshots at the planned points. Its new blocks join
   the tree as they are committed, so a request that arrives meanwhile can reuse them.
4. Requests that arrive together with the same new prefix wait for the first one's snapshot
   instead of all prefilling it.
5. When the request ends, its blocks are written through to host RAM and its endpoint snapshot is
   published. Pins keep everything a running request uses out of eviction's reach.

**Where the code is**

- `src/runtime/prefix_cache/`: the block tree and eviction (`prefix_index`), the snapshot
  planner (`tap_planner`) and the cost model.
- `src/models/qwen3_5/program/prefix/`: the model side. It covers GPU and host copies
  (`hybrid_cache`), admission, snapshots and finish (`hybrid_program`), the host memory layout
  and the cache file.
- `src/runtime/engine/context_cache/hybrid_resource_manager.h`: the Engine's admission contract.

Features and measurements are described under
[Hybrid prefix cache](#hybrid-prefix-cache-the-default), and the full design
in the [hybrid prefix cache spec](docs/maintainer/hybrid-prefix-cache-spec.md).

## Performance: this fork vs upstream

### Three-way agentic A/B (September 2026)

The closed-loop agentic suite in [`bench/agentic_ab/`](bench/agentic_ab/README.md) replays three
coding-agent sessions plus eleven subagents: 130 requests with fan-outs, a concurrent subagent
pair, compaction, retries, an abort and a solo wrap-up, prompts of 25K-135K tokens and thinking
on. Each arm's own answers are fed back as an agent client does, and the three main sessions take
their turns in lock-step rounds, so every build meets the same order of session turns whatever
its speed. It ran on three workload seeds, each replaying different observations. All three arms
served the official Qwen3.8-27B NVFP4 artifact on an RTX 5090 with the production launch flags
(DFlash2 + ngram drafting, `--max-concurrency 2`, int8 KV, 52 GB host cache) and production
sampling (temperature 1.0, top_p 0.95, top_k 20) at `--max-context 160000`, the largest context
the upstream build starts with.

- **Upstream + Windows port:** upstream at the commit this fork merged, plus only the Windows port
  (build `96da12bb`), given the same host RAM split as the original cache. It has no ngram
  drafting.
- **master:** this fork at `e36f7ee0` with the original prefix cache (then the default, now
  `--use-original-prefix-caching`) and `--fast-prefill-kernel`.
- **master + hybrid cache:** the same build with the hybrid prefix cache (then selected with
  `--use-alt-prefix-caching`, now the default).

Each cell is the mean over the three seeds with the lowest and highest seed in brackets; the
changes are computed per seed against that seed's upstream run.

| Metric | Upstream + Windows port | master | master + hybrid cache |
|---|---|---|---|
| Average time to first token (s) | 15.7 (12.4-19.0) | 7.0 (6.6-7.7), −54 % | 3.3 (3.0-3.5), −78 % |
| Median time to first token (s) | 10.0 (5.9-14.4) | 2.2 (1.8-2.8), −74 % | 0.8 (0.6-1.0), −90 % |
| 90th-percentile time to first token (s) | 37.4 (31.9-47.3) | 21.3 (16.4-24.4), −40 % | 8.8 (7.0-9.8), −75 % |
| Average TTFT, continuing-session turns (s) | 16.4 (12.7-20.1) | 6.9 (6.0-7.7), −57 % | 3.2 (2.9-3.5), −80 % |
| Average TTFT, new long prompts (s) | 13.2 (12.8-13.6) | 11.9 (10.4-14.3), −10 % | 8.2 (8.1-8.4), −37 % |
| Prompt tokens served from cache | 67.2 % (65.3-69.2) | 75.1 % (74.4-76.3) | 90.2 % (90.1-90.3) |
| Prompt tokens prefilled | 1.93M (1.73-2.07) | 1.46M (1.35-1.53), −24 % | 0.56M (0.54-0.57), −71 % |
| Main-session turns that re-prefilled the whole prompt (of 75) | 15 (10-19) | 11 (10-12) | 1 (1-1) |
| Subagent turns that re-prefilled the whole prompt (of 37) | 25.7 (24-29) | 2 (1-3) | 0 |
| Prefill tok/s, requests with no cache hit in any arm | 4,969 (4,765-5,130) | 6,050 (5,890-6,165), +22 % | 7,464 (7,363-7,526), +50 % |
| Output tok/s, one request decoding | 185 (175-193) | 189 (186-191), +3 % | 209 (188-222), +13 % |
| Decode rounds/s, one request decoding (engine speed) | 54.1 (53.5-54.6) | 54.0 (53.2-55.2), −0.2 % | 54.8 (54.4-55.3), +1.2 % |
| Tokens per round, one request decoding (acceptance) | 3.41 (3.26-3.56) | 3.50 (3.45-3.56) | 3.82 (3.44-4.08) |
| Output tok/s, two requests decoding (combined) | 315 (305-324) | 295 (290-303), −6 % | 314 (304-320), 0 % |
| Decode rounds/s, two requests decoding (engine speed) | 51.4 (51.3-51.4) | 45.9 (45.0-46.6), −11 % | 47.8 (46.8-48.5), −7 % |
| Output tok/s, all decoding at the run's own batching | 197 (186-203) | 234 (225-248), +19 % | 253 (247-257), +28 % |
| Decode rounds that ran two requests | 9.8 % | 38.3 % | 40.5 % |
| Workload wall time (min) | 21.6 (19.9-23.1) | 18.0 (16.1-19.8), −17 % | 13.6 (12.6-15.1), −37 % |

How to read it:

- Every arm completed every request on every seed. Against upstream, every seed agrees on the
  direction of the average, median and continuing-session TTFT, cache hits, prefilled tokens and
  whole-run output. master's TTFT on new long prompts and its count of main-session re-prefills
  moved either way between seeds. The combined report (`bench/agentic_ab/analyze.py --aggregate`)
  has every range.
- TTFT includes queueing: up to seven requests are in flight on two lanes. The average queue wait
  was 12.8 s / 4.7 s / 2.6 s. Without it, TTFT averaged 2.93 s / 2.38 s / 0.72 s.
- The cache rows now repeat closely: across seeds, master served 74.4-76.3 % of prompt tokens from
  cache and the hybrid cache 90.1-90.3 %. Before the sessions ran in lock-step, two runs of
  one seed on one master build served 61.6 % and 72.6 %, because a faster or slower turn changed
  which session's prefix was evicted.
- Output tok/s is decode tokens per second of the engine's own decode time, so prefill and idle
  time do not dilute it. It splits into decode rounds/s, the engine's speed, and tokens per round,
  the speculative acceptance, which moves with what the model happened to write. With one request
  decoding, the three arms run the same number of rounds per second within about 1 % on average
  and 2 % on any seed; the differences in output tok/s come from acceptance. The fork's ngram drafting supplied 8-11 % of
  its output; upstream has none.
- With two requests decoding, master's rounds were about 11 % slower than upstream's and its
  combined output 6 % lower; the hybrid-cache arm, on the same build, was 7 % slower per
  round and even on output. Upstream decoded two requests together for only 25-65 s per seed, so
  its two-request rows rest on little data.
- The fork's whole-run output rate is higher because it decodes both lanes together in about 40 %
  of its rounds, upstream in about 10 %.
- Sampled output differs between arms and seeds (152K-195K completion tokens per run, about 78 %
  thinking); the seed ranges include that variation.

### Earlier A/B: original cache vs upstream (September 2026)

Both builds served the same model on the same GPU and replayed the same agent-style workload:

- **Model and GPU:** the official NInfer Qwen3.8-27B NVFP4 artifact
  (`qwen3_8_27b_nvfp4-official.ninfer`) on an NVIDIA GeForce RTX 5090.
- **Context:** `--max-context 180000` for both, the largest context upstream can start with on
  this card (see the note below the launch parameters).
- **Workload:** 14 requests modelled on a real request log. They are multi-turn tool-agent
  sessions sharing an ~82,000-token prefix, with a median prompt of about 121K tokens, including
  two pairs of concurrent requests (one pair competing for the same cached prefix).
  `--max-concurrency 2`, seed 42.

| Metric | Upstream + Windows port | This fork | Change |
|---|---|---|---|
| Average time to first token (s, lower is better) | 28.1 | 14.6 | −48.1 % |
| Median time to first token (s, lower is better) | 25.9 | 11.3 | −56.6 % |
| Prompt tokens served from cache | 19.8 % | 66.3 % | +46.5 points |
| Cached tokens (of 1,649,215 prompt tokens) | 326,780 | 1,093,256 | +234.6 % |
| Prefill tok/s on requests with no cache hit (median) | 4,159 | 4,220 | +1.5 % |
| Output tok/s | 173 | 181 | +4.6 % |

How to read it:

- Time to first token is averaged over all 14 requests.
- "No cache hit" requests are those that had to prefill the whole prompt (upstream 11, fork 2).
- Output tok/s is total completion tokens divided by total decode time. Thinking was on for both.
  Upstream produced more tokens (12,639 against 6,543), but per-request decode rates are similar.
- Where the cache hits came from (requests / cached tokens):

  | Reuse path | Upstream | This fork |
  |---|---|---|
  | Long anchor inside a private conversation | 0 / 0 | 9 / 766,476 |
  | Replay of the previous response | 3 / 326,780 | 3 / 326,780 |
  | No reuse (full prefill) | 11 / 0 | 2 / 0 |

Almost all of the gain comes from the prefix-cache changes described under
[Prefix caching and KV memory](#prefix-caching-and-kv-memory). A workload with more copying would
also gain output speed from ngram drafting (`--ngram-draft-tokens` / `--ngram-min-match`).

This run predates `--host-cache-mib` and `--fast-prefill-kernel`, so the fork used the older
separate cache flags shown below. The launch lists are kept as the record of what was measured;
those flags configure the original prefix cache and now also need `--use-original-prefix-caching`.

Launch parameters, this fork:

```text
--host 127.0.0.1 --port 8080 --max-context 180000 --max-concurrency 2 --spec dflash2
--draft-tokens 7 --lm-head-draft --ngram-draft-tokens 15 --ngram-min-match 12
--kv-dtype int8 --preserve-thinking --host-kv-mib 24000 --pending-timeout-ms 900000
--prefill-chunk 2048 --kv-capacity auto --kv-headroom-mib 0 --log-colours on
--host-state-slots 64 --max-private-continuations 32
--max-long-anchors-per-continuation 8 --max-shared-prefixes 32
--ngram-archive-mib 2048 --ngram-session-mib 256 --ngram-native-sessions
--cuda-graph-allowance-mib 500 --default-thinking-budget 16384
--thinking-budget-message "Considering the limited time available to the user, I must stop
thinking now. Time to act:"
```

Launch parameters, upstream + Windows port (the same list without the fork-only flags
`--ngram-draft-tokens`, `--ngram-min-match`, `--kv-headroom-mib`, `--log-colours`,
`--ngram-archive-mib`, `--ngram-session-mib`, `--ngram-native-sessions`,
`--cuda-graph-allowance-mib` and `--thinking-budget-message`):

```text
--host 127.0.0.1 --port 8080 --max-context 180000 --max-concurrency 2 --spec dflash2
--draft-tokens 7 --lm-head-draft --kv-dtype int8 --preserve-thinking
--host-kv-mib 24000 --pending-timeout-ms 900000 --prefill-chunk 2048 --kv-capacity auto
--host-state-slots 64 --max-private-continuations 32
--max-long-anchors-per-continuation 8 --max-shared-prefixes 32
--default-thinking-budget 16384
```

Why 180000: upstream always keeps 1 GiB of GPU memory spare when sizing the KV cache
automatically, and has no flag to lower it (the fork's `--kv-headroom-mib 0`). On a 32 GiB card
that stops upstream from starting at the 220000 context used in production, so both builds ran at
180000 to keep the comparison fair.

The complete test rig (workload generator, runner, watchdog and control-build driver) is in
[`bench/ab/`](bench/ab/README.md), so you can repeat the test with your own builds and flags.

## What this fork changes

Each topic below lists everything that affects it, whether written here or taken from elsewhere.
"Upstream PR" means an open pull request on `Neroued/ninfer` that this fork has merged before
upstream has.

### Prefix caching and KV memory

Most of the changes below are to upstream's checkpoint catalog, which the fork now runs only with
`--use-original-prefix-caching`; the default is the
[hybrid prefix cache](#hybrid-prefix-cache-the-default).

NInfer can skip prefilling a prompt prefix it has already processed, but only from a *complete
checkpoint*: the saved model state plus the KV cache at that exact point in the prompt. The rules
for when a checkpoint can be reused are upstream's. This fork changes how GPU and host memory is
shared out among checkpoints, which ones are kept, and which ones are thrown away when memory runs
short. On long multi-turn agent sessions, upstream ends up re-prefilling most requests; this fork
reuses most of them. Details are in [paged KV context store](docs/maintainer/paged-kv-cache.md),
[resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
and [HTTP serving](docs/serving.md).

- **GPU KV memory grows with the answer instead of being reserved up front.** Upstream reserves
  room for the full prompt plus the full `max_tokens` output when a request starts, so a client
  asking for `max_tokens: 64000` holds all of that for the whole request and squeezes out the
  cache. This fork reserves the prompt plus a 4,096-token output window, then extends it as the
  answer grows.
  - If the GPU pool is full of cached prefixes when the answer needs more room, the engine frees
    the least recently used idle cache entries (only ones actually holding GPU pages), just enough
    for the answer's next step, and carries on. A running answer always wins over cached data. Before this, answers could stop at about
    4,000 tokens, often mid-reasoning; in one real Qwen Code session this happened on 8 of 53
    requests. The console shows
    `[engine] Device KV lease of lane N extended: released K retained cache owner(s)`.
  - Only if nothing can be freed does the answer end early, with `finish_reason: "length"` and a
    one-off `[engine] warning`, rather than failing.
  - The request log and the `server_start` memory report show the reserved amount separately
    from actual use.
- **Memory pressure moves cache to host RAM before deleting it.** When a prefix loses its GPU
  copy, it is moved to host memory if there is room, for both private conversations and shared
  prefixes. Upstream cleared both copies at once.
- **When something must be evicted, the oldest goes first, and only as much as needed.**
  - Private conversations and shared prefixes share one "least recently used" order. A
    conversation that has just finished counts as recently used.
  - Eviction gives up the oldest entries first, then keeps any it turns out not to need. For
    example, running out of private-conversation slots no longer deletes shared prefixes.
  - If moving to host RAM is not possible, it tries again with the kept entries left in place.
    Clearing everything is only a last resort to keep the engine moving.
  - Other requests waiting to start are limited in the same way and never trigger a full clear.
  - If a fallback plan cannot be carried out, the request waits and plans again instead of every
    request failing.
- **A checkpoint loses its value only when its own conversation has moved past it.** Upstream PR
  #300 (by [pkochubey](https://github.com/pkochubey), for upstream issue
  [#178](https://github.com/Neroued/ninfer/issues/178)) stopped counting checkpoints the incoming
  request cannot use. This fork narrows that: checkpoints from other conversations and shared
  prefixes keep their value, because other requests can still use them, so they no longer look
  free to evict.
- **One host RAM setting: `--host-cache-mib`.** Upstream sizes host cache from two separate
  allocations plus several catalog limits, so total pinned RAM is their sum. Here one ceiling
  covers everything: the engine sizes the saved-state pool for the checkpoints it will really
  create, spends any spare room on more long anchors per conversation, gives host KV the rest,
  and refuses to start rather than over-commit.
- **Long anchors are placed automatically.**
  - Checkpoints are placed at message boundaries without the client marking them
    (`--max-long-anchors-per-continuation`), so editing an older message resumes from a nearby
    anchor instead of from the start.
  - Anchors are spaced further apart the further back they are (`--long-anchor-spacing`, default
    1024 tokens), so short tool-loop turns do not each use one up and older history stays covered.
  - When the set is full, the anchor whose loss costs the least coverage is replaced, rather than
    the deepest one.
- **Work already done is kept.** A request that is aborted still saves the part of the prompt it
  had prefilled, so a retry carries on from there.
- **More time to find a good cache plan when admitting a request.** Picking which cached entries to
  keep for a new request is a time-limited search.
  - The search budget scales with how expensive the request is, from 5 ms up to 250 ms. A flat
    5 ms covered only about 10 options, so good plans were missed (upstream issue
    [#229](https://github.com/Neroued/ninfer/issues/229); approach suggested there by Gene0Liu).
  - The overall planning allowance for each admission is always 250 ms. Upstream cuts it to 50 ms
    while another request is running, which made the second request of a concurrent pair miss an
    81K-token reusable anchor and re-prefill from scratch. The running request pauses for at most
    that 250 ms, once per admission.
  - With `--fast-prefill-kernel`, a different effective chunk size can lead this search down a
    different path, so cache decisions can differ from a run without the flag.
- **The shared-prefix list no longer fills up for good.** Once every `--max-shared-prefixes` slot
  was in use, new shared prefixes were dropped and shared reuse stopped until restart (upstream
  issue [#251](https://github.com/Neroued/ninfer/issues/251)). The least recently used automatic
  entry is now replaced, only once the new prefix is actually being saved (approach suggested
  there by albertov).

Measured on a replay of the traffic that prompted this work (26 multi-turn tool-agent requests,
`--host-cache-mib 40000`, Qwen3.8-27B NVFP4 + DFlash2, `--max-concurrency 2`, RTX 5090):

- 90.75 % of prompt tokens came from cache overall (98.08 % once warm), against 1.44 % in the
  production logs the replay was built from.
- Fully evicted private conversations fell from 179 to 2–3, and full clears from 55 to 0–1.

That replay predates some later eviction fixes and lasts only 3.7 minutes. All 18 real-engine
prefix scenarios in `ninfer_qwen3_5_prefix_real_test` pass on this fork, including
`shared-saturation-reclaim`, `shared-replacement`, `private-checkpoint-pressure` and the four
`review-*` scenarios, which all fail without these changes.

### Hybrid prefix cache (the default)

The fork's default prefix cache, replacing the checkpoint catalog above, which
`--use-original-prefix-caching` selects instead. Design and status:
[hybrid prefix cache](docs/maintainer/hybrid-prefix-cache-spec.md).

- **KV is cached per 64-token block, keyed by content.** Identical prompt blocks are stored once,
  whichever conversation produced them, so a shared system prompt costs its GPU pages once at any
  concurrency.
- **Saved model state is sparse.** The recurrent state is saved only at useful points: where the
  next turn resumes (the start of the assistant reply), the end of tools and system prompt,
  explicit client cache breakpoints, the end of each answer, and a few points spread back through
  long history. Most of these cost no extra prefill work.
- **It configures itself.** The only setting is `--host-cache-mib N` (default 8192; `0` keeps
  the cache on the GPU only). Free VRAM becomes GPU cache (`--kv-capacity`
  defaults to `auto`), and the host budget is one pinned pool that KV blocks and saved states share,
  split by how much prefill time each entry saves. Everything else is derived from
  `--max-concurrency` and `--prefill-chunk`.
- **Restores from host RAM overlap the request's own prefill.** A request resuming from host RAM
  starts at once: each model layer waits only for its own restored data. The request computes early
  layers while later ones are still being copied, so a long restored context costs little more than
  its new tokens.
- **Parallel requests with a new shared prefix prefill it once.** When requests share a prefix
  that is not cached yet, such as subagents started together with the same system prompt, later
  requests wait for the first one's saved state at the point where the prompts diverge. They
  resume from it instead of prefilling the prefix again. Four requests with a new 13.9K-token
  system prompt: mean time to first token 1.48 s instead of 3.52 s.
- **The cache can survive a restart.** With `--prefix-cache-file PATH` (for example
  `--prefix-cache-file "e:\NInfer-Deploy-V3\file.cache"`), the host RAM part of the cache is read
  from that file at startup if it exists, and written back when the server shuts down with Ctrl+C
  or Ctrl+Break, or when its console window is closed. Long conversations and system prompts then
  resume instead of re-prefilling. Without the flag nothing is saved. A file from a different
  model, KV format or `ninfer-serve` build is ignored and replaced at shutdown.
  - Windows ends a closing console's process about 5 seconds after the close. A save that has not
    finished by then is abandoned and the previous file is kept. At about 3 GB/s that covers a few
    GiB of cache; for larger caches, stop the server with Ctrl+C, which has no time limit.

### Faster prefill: `--fast-prefill-kernel`

An opt-in flag (off by default) on `ninfer-serve`, `ninfer-perplexity` and `ninfer_bench` that
speeds up prefill with `--kv-dtype int8`, especially for long prompts. Without it, nothing
changes. With it:

- **A faster attention kernel for prompts.** Every prefill step wider than 16 tokens (wider than
  64 once the context is long, see below) uses a new INT8 prompt-attention kernel
  (`src/ops/softmax_attention/dense/causal_cache/prompt_i8_fast.cuh`), written in the style of
  FlashAttention-2:
  - each warp keeps its 16 query rows, scores and output in registers for the whole pass over
    the keys;
  - the KV cache stays INT8 in memory, is double-buffered, and V is decoded in registers;
  - the probability × V product runs on FP16 Tensor Cores and is added into FP32 once per
    64-key tile, with an exact power-of-two rescale for V scales large enough to overflow FP16;
  - a small cost model picks 128-row or 64-row blocks, and the longest blocks are launched first.

  At 131K context the kernel runs at about 313 TFLOP/s instead of 193 on an RTX 5090. Other KV
  formats keep their existing kernels.
- **Prefill chunks sized to fill the GPU evenly.** The effective `--prefill-chunk` is rounded
  down to a whole number of GPU "waves" (896 tokens for this 24-head model on 170 SMs), so `4096`
  runs as `3584` and `4480` stays as it is. This saves about 7 % of attention time at long
  context.

Measured against build `f351298e` (flag off) on Qwen3.8-27B NVIDIA NVFP4, `--kv-dtype int8`,
RTX 5090:

| Measurement | Flag off | Flag on | Change |
|---|---|---|---|
| New 16K prompt (`ninfer_bench`, prefill tok/s) | 10,059 | 10,438 | +3.8 % |
| New 64K prompt | 6,803 | 7,801 | +14.7 % |
| New 128K prompt | 4,717 | 5,891 | +24.9 % |
| `bench/ab` workload, attention context under 60K (prefill tok/s) | 4,860 | 5,766 | +18.6 % |
| `bench/ab` workload, 60–100K | 3,338 | 4,201 | +25.8 % |
| `bench/ab` workload, over 100K | 2,956 | 3,762 | +27.3 % |
| Perplexity, `--quick`, 64K context / 32K stride | 4.1679 | 4.1713 | +0.08 % |

- The workload rows pool 52 requests that got exactly the same cache reuse in both runs, over four
  runs, with both runs using the same effective chunk size. Total prefill time fell 20 %, every
  request got faster (1.17–1.31×), decode speed was unchanged, and average time to first token
  fell 19 %.
- Accuracy is unaffected: BF16 KV scores 4.1695 perplexity, so the fast kernel is as close to full
  precision as the default one, and closer in three of the four test domains.
- A variant sharing one block per KV head ("PackGQA") was also tried and dropped: it was never
  faster and up to 1.4× slower on short chunks.

### Other speed changes

- **Faster speculative decode rounds.** Output is unchanged token for token; only the time per round
  changes.
  - Kernels in the decode CUDA Graph launch as programmatic dependents of the kernel before them
    (PDL). Weight-streaming kernels (the NVFP4/FP8 projections, the drafter's Q8/Q4 projections,
    the GDN record step) load their first weight tiles while the previous kernel is still running,
    and let the next kernel launch only after their own main loop, so the two never compete for
    memory bandwidth. A 28-byte memset in the proposal head's top-k, which cost about 70 us of idle
    GPU per round on Windows, is now a kernel.
  - After each round the engine no longer waits for the recurrent-state fold before preparing the
    next round, except when a finishing request still needs its input buffer.
  - Measured on Qwen3.8-27B NVIDIA NVFP4, DFlash2 K=7, INT8 KV, RTX 5090:

    | Measurement | Before | After | Change |
    |---|---:|---:|---:|
    | `ninfer_bench` greedy decode, 16K context (ms per round) | 16.79 | 16.39 | -2.4 % |
    | `ninfer_bench` greedy decode, 16K context (decode tok/s) | 391.0 | 400.6 | +2.5 % |
    | `ninfer_bench` greedy decode, 60K context (ms per round) | 18.00 | 17.61 | -2.2 % |
    | `ninfer_bench` greedy decode, 60K context (decode tok/s) | 346.9 | 354.6 | +2.2 % |
    | Served coding prompts, production thinking sampling, same seeds (output tok/s) | 208.1 | 213.7 | +2.7 % |

    The `ninfer_bench` rows are 9 repetitions per arm, interleaved, with identical round and
    acceptance counts in both builds (standard deviation at most 0.01 ms).

    Both builds produced byte-identical greedy text through `ninfer-serve` with DFlash2 + ngram,
    MTP and no speculation, and the same 141,575 sampled tokens in the served run. On the
    closed-loop agentic A/B (`bench/agentic_ab`) the time per single-request decode round fell by
    about 2 %; its headline output rates move by 5-10 % between runs with sampled content alone,
    so they cannot resolve a change this size.
  - Tried and not kept (details in [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md)): a fused NVFP4 RMSNorm +
    SwiGLU + down MLP (0.3 % faster alone, 0.7 % slower together with PDL), and restricting or
    sharpening the DFlash2 proposal distribution (no gain in acceptance).
- **Ngram copy drafting with more than one concurrent request.** Ngram drafting proposes the next
  tokens by copying matching text from earlier in the context, alongside MTP/DFlash/DFlash2. The
  single-request version is the original work of [remesis](https://github.com/remesis) in the
  [remesis/ninfer](https://github.com/remesis/ninfer) fork (upstream issue
  [#234](https://github.com/Neroued/ninfer/issues/234)). This fork extends it to
  `--max-concurrency` above 1: each request gets its own drafting state, the shared history archive
  is enabled, and startup checks the model's concurrency limit. See [ngram copy
  proposals](docs/ngram.md).
- **Several requests can prefill at the same time.** By David Oelfke in the
  [gzenz/ninfer](https://github.com/gzenz/ninfer) fork (commit `576e72ea`). Upstream lets only one request prefill at a time and holds other
  admissions until it finishes. Here new requests can be admitted to free lanes while others are
  prefilling, so one request's prefill overlaps other requests' prefill and decode. Each step still
  advances one prefilling request.
- **Short prefill steps over long contexts use split-KV attention.** A prefill step of 17–64 new
  tokens (a user message or a chat template's closing tokens) against a long context used the
  prompt-attention kernel. That kernel runs one thread block per query head and row block, so it
  left most of the GPU idle: 32 new tokens against 180K cached tokens took 9.5 ms per attention
  layer. Such steps now use the chunked split-KV kernels that speculative verification already
  used, once the context holds at least 64 cached tokens per new token (80 for 16-head models). The
  same step then takes 1.06 ms, 4–12× faster from 16K to 180K tokens, for every KV format. With
  ~90K cached tokens, a short follow-up question's time to first token fell from 204 ms to 91 ms
  (together with fewer prefill splits in the hybrid prefix cache).
- **Kernel tuning from upstream PRs:** partial last tile in the fused SwiGLU TMA route (#264) and
  the sigmoid gate folded into the causal reduce step (#268), both by Michael Dementii; the text
  `rmsnorm_rope` route (#273, Michael Dementii); tuned Q6 34,816×5120 dispatch (#284, by
  [bingchengcc](https://github.com/bingchengcc)); and Q5 linear K-split sized to the token count
  (#292, by [giveen](https://github.com/giveen)).
- **Kernel changes adapted from [llmq](https://github.com/IST-DASLab/llmq)** (IST-DASLab, Erik
  Schultheis), from upstream PRs by [DuncanBetts](https://github.com/DuncanBetts):
  - From 1,024 prompt tokens up, the NVFP4 attention-input projection runs its RMSNorm and
    activation quantisation in one kernel, skipping one launch and one round trip of the
    normalised activations through memory (#305). This applies to the NVFP4 artifacts on this page. The
    output is byte-identical to the separate steps, and the PR measured the norm, quantise and GEMM
    stage 4–17 µs faster per layer at 1,024–4,096 tokens.
  - Target log-probabilities (perplexity / CausalScoring) find the maximum and the sum in a single
    pass over the logits instead of two (#307): about 1.6× faster for that kernel. The results
    match to within the existing test tolerance.

### Tool calls and reasoning output

- **More tool-call formats are understood.** Upstream PR #300 (by
  [pkochubey](https://github.com/pkochubey), for upstream issue
  [#276](https://github.com/Neroued/ninfer/issues/276)) accepts the XML forms emitted by Claude
  Code and other agent tools: `<function name="…">`, `<invoke>`, `<function_calls>`, and the short
  `<param>` / `</parameter>` tags, including while streaming.
- **Repeated tool-call parameters keep the last value** (as JSON does) instead of turning the
  whole call into plain text. From upstream PR #299 by [adubkov](https://github.com/adubkov); the
  repair is counted as `duplicate_parameters_repaired`, and a short snippet is logged if a call
  still falls back to text. PR #300's rejection of conflicting duplicates was replaced with this
  rule.
- **Quoting `</think>` no longer ends the reasoning early.** The engine used to end the reasoning
  at the first `</think>` the model wrote, so a model thinking *about* chat templates had the rest
  of its reasoning published as the answer, which clients such as Qwen Code reject as leaked
  thinking tags. From upstream PR #309 by Fedor Suchkov, adapted here:
  - `</think>` only ends the reasoning when a line break (or the end of the turn) follows it,
    which is how the model really writes it. The PR also accepted a space, which still leaked in
    a live test on text such as "the `</think>` tag".
  - If a quoted `<tool_call>` does not parse, the parser tries later `<tool_call>` markers. It
    does not restart at other marker forms, which would misread a truncated call.
- **`--tolerant-tool-calls` (opt-in) rescues slightly broken tool calls.** Designed by David Oelfke in the
  [gzenz/ninfer](https://github.com/gzenz/ninfer) fork (commits `b2267e06`, `0ce6e3f6`, `0f3c9f55`,
  `38709834`, `44f2c9c3`) and ported onto this fork's parser. In tolerant mode:
  - a good call followed by junk or a broken second call keeps the good call (logged as
    `truncated_tail`);
  - a final call cut off by the output limit is kept with its partial value, if at least one
    parameter is complete;
  - a missing `>` after the function name is repaired;
  - calls to tools the client did not declare are still returned as tool calls.

  Without the flag, the strict parser is used as before.

### API and client compatibility

- **llama.cpp-style model details on `/v1/models`** (`n_vocab`, `n_ctx`, `n_ctx_train`, `n_embd`,
  `n_params`, `size`, `ftype`). From upstream PR #162 by
  [Hector Ramon Jimenez (hecrj)](https://github.com/hecrj), rewritten for the current source
  layout.
- **`ignore_eos` on chat completions.** Upstream PR #197 by [Thireus](https://github.com/Thireus).
- **GitHub Copilot and other agent-host requests are accepted** instead of refused before
  generation. From upstream PR #316 by [paq85](https://github.com/paq85) (Damian Sromek), with
  only its serving commits taken:
  - on chat completions, `custom` tools are served to the model as a function with one string
    `input`, under the caller's own tool name. `tool_choice` `required`, named or `custom`,
    `allowed_tools` in `required` mode, `strict: true` and `parallel_tool_calls: false` are
    accepted but only advisory, because NInfer cannot force a call or constrain the arguments.
    `reasoning_effort` `default` / `auto` use the server's own setting;
  - tool names may be up to 256 bytes on every protocol (was 64, or 128 on Messages), because VS
    Code wraps MCP tools under longer names such as `activate_fallback_mcp_<server>_<tool>`;
  - a rejected tool name is reported with its value, byte length and exact location in the
    request;
  - `--usage-chunk-choice` (opt-in) gives the streamed usage chunk a blank choice, for clients that
    reject the standard empty `choices` array.

  The PR's tool-call parser rewrite is not merged: it would hide a failed tool call rather than
  return it as text, which overlaps with this fork's opt-in `--tolerant-tool-calls` and its
  handling of quoted `<tool_call>` text. Its prefix-cache test change is also left out, because
  that test here no longer depends on the chat template.
- **Responses API options used by Codex and Zed Agent:** `reasoning.summary` and
  `include: ["reasoning.encrypted_content"]` are accepted. Upstream PR #295 by
  [Macasacker](https://github.com/Macasacker), based on an earlier PR by
  [Sha1rholder](https://github.com/Sha1rholder).
- **`response_format` of `json_object` or `json_schema` is accepted** instead of refused, so
  clients that always send one (such as Hermes-style clients) work. The format is not enforced,
  as NInfer has no constrained decoding; `docs/serving.md` says so. Upstream PR #300 by
  [pkochubey](https://github.com/pkochubey).
- **A cut-off tool call is reported as cut off.** When the output or context limit truncates a
  tool call, the finish reason is `length` (chat completions) or `max_tokens` /
  `model_context_window_exceeded` (Messages), not `tool_calls` / `tool_use`, so a client does not
  run a call with incomplete arguments. The partial call is still streamed. Upstream PR #300 by
  [pkochubey](https://github.com/pkochubey).
- **Continuing a partial assistant reply.** A request that ends with an assistant message is
  treated as "continue this reply" on chat completions as well as Messages, and that message may
  contain reasoning or tool calls (upstream PR #300 by [pkochubey](https://github.com/pkochubey)).
  It only works with thinking off: with thinking on, the template would put the continued text
  inside an open reasoning block, so the request is refused with `invalid_prompt`. When neither
  the request nor the server says whether thinking is on, it now counts as on (matching the chat
  template), so such a request is refused rather than rendered as a broken continuation that the
  model ends after a few tokens.

### Stability

- **Out-of-memory no longer stops the engine.** By David Oelfke in the gzenz/ninfer fork (commit
  `3f3272d6`). If reserving GPU memory fails, only the affected request fails, with a retryable
  `Overloaded` error. If memory runs out mid-run, the active requests fail, the engine resets and
  carries on with waiting requests still queued. After 8 failed recoveries in a row it fails the
  queue instead. A fatal crash logs a `WORKER CRASH` message.
- **Recovery really leaves the engine empty.** When an internal check fails mid-request (for
  example a cache accounting error while several agents run at once), recovery used to discard
  the requests but could leave some cached GPU KV pages or saved states with no owner. The engine
  then looked full while idle, rejected every new request, and after 8 retries stopped serving
  for good (every later request got HTTP 503). Now, if anything is still held after cleanup, the
  engine rebuilds its cache stores from empty and logs
  `[engine] recovery rebuilt the context stores: ...` with what had been left behind. A request
  that still cannot start on an idle engine fails on its own instead of taking the engine down.
  Cache accounting errors also now name the step that failed and the values involved, so the
  cause can be traced from the console.
- **Cache planning cannot race with itself.** By [Gideon Zenz (gzenz)](https://github.com/gzenz) in the
  gzenz/ninfer fork (commit `c53e025c`). The step that checks and then commits an eviction plan could be disturbed by a
  concurrent move to host RAM, which threw an error that stopped the whole engine. It is now
  claimed atomically; if the claim or the commit fails, the request re-prefills instead of
  failing. Saving a shared prefix uses the same claim and simply skips saving if it fails.
- **No shared-memory overflow at very long contexts.** By David Oelfke in the gzenz/ninfer fork
  (commit `7a876cf7`). Split-KV decode attention stages at most 64 KV pages per split, so at large
  (YaRN-extended) contexts too few splits overflowed shared memory. The split count now has a
  minimum based on the context length (and a maximum of 256).

### Models, conversion and vision

- **GGUF files as conversion sources** (any ggml quantisation). Upstream PR #282 by
  [giveen](https://github.com/giveen).
- **Quasar NVFP4 conversion fixes:** DFlash2 head declarations and an indexed proposal head
  (`--proposal`), so rebuilt artifacts support `--lm-head-draft`.
- **Q8 MTP** with mixed-format row-split projection, and a **general BF16 GEMM fallback** for
  shapes without a dedicated kernel (full-precision vocabulary heads, BF16 vision projections).
- **`--rope-yarn-factor`** for YaRN context extension (factor 1–4, up to the 1M visible-key
  limit).
- **`--vision-residency overlay`** keeps the vision tower in pinned host RAM and streams it to
  the GPU when needed, leaving more GPU memory for KV. Based on the original work by
  [Valeriy Selitskiy (iamwavecut)](https://github.com/iamwavecut) for the previous-generation
  engine, rewritten for this one and fixed to work with every artifact.

### Windows

- **Native Windows build and run** with MSVC and CUDA: static CUDA runtime, FFmpeg/curl from vcpkg
  with their DLLs copied next to the executables, and workarounds for MSVC limits (non-RDC NVFP4
  kernels, TMA descriptors stored into device memory by a staging kernel, Windows file-mapping
  rules).
- **PNG images in the vision path.** The prebuilt vcpkg FFmpeg has no PNG decoder, so a built-in
  PNG decoder (`NINFER_MEDIA_NATIVE_PNG`) was added.
- **Converter recipe paths with drive letters.** In `--recipe FILE[:function]`, a colon inside
  the path (such as `E:`) is treated as part of the path.

### Options and console

- **`--log-colours`** colours the console statistics (throughput, cache reuse, memory).
- **Session statistics panel.** On an interactive terminal the serve console pins a panel beneath
  the scrolling log with session and last-ten averages of TTFT, cache hit rate, non-cached prefill
  and decode speed, model-drafter (MTP/DFlash) acceptance and n-gram acceptance
  (`--log-stats-panel off` removes it).
- **Grouped help screens.** `--help` groups options into sections (Context, KV Cache,
  Speculative Decoding, Vision, Sampling, Networking & Resources, …) and covers flags that were
  previously undocumented.
- **`--kv-headroom-mib`** sets how much GPU memory automatic KV sizing leaves spare (upstream
  always leaves 1 GiB).
- **Measured CUDA Graph allowance.** The automatic allowance is 64 MiB plus 4 MiB per decode-graph
  executable, sized from the graph memory measured on an RTX 5090 (DFlash2 with n-gram drafting at
  `--max-concurrency 2` reserves 160 MiB and uses about 62 MiB, where upstream's per-profile
  estimate reserves 1,920 MiB). Startup reports the memory the graphs actually used and warns
  when it exceeds the allowance; **`--cuda-graph-allowance-mib`** replaces the automatic value.
- **`--thinking-budget-message`** sets the message inserted when a request reaches its thinking
  budget.
- **`--chat-template`** loads the chat template from a file instead of the artifact.

### Kept in sync with upstream

Upstream `master` is merged regularly, bringing in its ongoing kernel, build and engine work. Once
upstream adopts a change listed above, it is removed from this README.

## Model artifacts

**[Qwen3.8-27B-Quasar-NinferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-Quasar-NinferV3)**
on Hugging Face — a single-file `.ninfer` engine artifact of
[QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)
(the QAT-trained NVFP4 checkpoint), built with `tools/convert/quasar_nvfp4.py` for an
RTX 5090 (sm_120a). The QUASAR NVFP4 weights are imported bit-exact (no requantisation
round-trip); the DFlash2 draft model is grafted verbatim from the official NInfer
artifact, and an indexed 131,072-row proposal head gathered from the QUASAR output head
enables `--lm-head-draft`. The HF page carries the full creation outline and the
conversion report.

Configuration used for running it (single 32 GB GPU — stop any other resident model
first):

```bat
ninfer-serve.exe "E:\NInfer-Deploy-V3-output\qwen3_8_27b_nvfp4-quasar-proposal.ninfer" --host 127.0.0.1 --port 8080 --max-context 240000 --max-concurrency 2 --spec dflash2 --draft-tokens 7 --lm-head-draft --ngram-draft-tokens 15 --ngram-min-match 12 --kv-dtype int8 --preserve-thinking --host-cache-mib 40000 --pending-timeout-ms 900000 --prefill-chunk 2048 --kv-capacity auto --kv-headroom-mib 0 --log-colours on --ngram-archive-mib 2048 --ngram-session-mib 256 --ngram-native-sessions --cuda-graph-allowance-mib 500 --request-log-jsonl log.json --default-thinking-budget 32000 --thinking-budget-message "I'm done thinking. Time to act:"
```

This is the `LaunchQwen3.8-27B-quasar-dflash2-ngram.bat` configuration with the retention tier
in its single-knob form: `--host-cache-mib` is the one pinned host pool that the default hybrid
prefix cache's KV blocks and state snapshots share.

**[Qwen3.8-27B-NVIDIA-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-NInferV3)**
on Hugging Face — a single-file `.ninfer` engine artifact of
[nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
(the Model Optimizer mixed NVFP4/FP8 checkpoint), built with the
`qwen3_8_27b_nvfp4_nvidia` recipe in `tools/convert/official_recipes.py` for an
RTX 5090 (sm_120a). The NVFP4 MLP and FP8 attention/GDN projections are
imported bit-exact (no requantisation round-trip); only the output head is
re-quantised (NVFP4 → row-scale FP8, because the engine registers the
vocabulary projection only for Q8/Q6/FP8, and FP8 was benchmarked faster than
Q8 at the decode/verify token range). The DFlash2 draft model is grafted
verbatim, and an indexed 131,072-row proposal head gathered from the NVIDIA
output head enables `--lm-head-draft`. The HF page carries the full creation
outline and the conversion report.

Configuration used for running it (single 32 GB GPU — stop any other resident model
first):

```bat
ninfer-serve.exe "E:\NInfer-Deploy-V3\qwen3_8_27b_nvfp4-nvidia.ninfer" --host 127.0.0.1 --port 8080 --max-context 240000 --max-concurrency 2 --spec dflash2 --draft-tokens 7 --lm-head-draft --ngram-draft-tokens 15 --ngram-min-match 12 --kv-dtype int8 --preserve-thinking --host-cache-mib 52000 --pending-timeout-ms 900000 --prefill-chunk 4096 --kv-capacity auto --kv-headroom-mib 0 --log-colours on --ngram-archive-mib 2048 --ngram-session-mib 256 --ngram-native-sessions --cuda-graph-allowance-mib 500 --request-log-jsonl log.json --default-thinking-budget 16384 --thinking-budget-message "Considering the limited time available to the user, I must stop thinking now. Time to act:" --tolerant-tool-calls
```

This is the current `LaunchQwen3.8-27B-nvidia-dflash2-ngram.bat` launch configuration, with the
retention tier in its single-knob form: the default hybrid prefix cache shares the 52,000 MiB
budget between KV blocks and state snapshots by eviction value. With `--use-original-prefix-caching`
the same budget resolves, at `--max-concurrency 2`, to 139 Host StateImages of 195,897,344 B — 31
long anchors per continuation — with the remaining ≈26,000 MiB given to Host KV; the `server_start`
memory ledger reports the resolved split.

## Thanks

A big thank you to all the contributors to upstream NInfer —
[Neroued](https://github.com/Neroued),
[Michael Dementii](https://github.com/MichaelDementii),
[Minnnn](https://github.com/Minnnn),
[Thireus](https://github.com/Thireus),
[remesis](https://github.com/remesis),
[Valeriy Selitskiy (iamwavecut)](https://github.com/iamwavecut),
[Hector Ramon Jimenez (hecrj)](https://github.com/hecrj),
[giveen](https://github.com/giveen),
[bingchengcc](https://github.com/bingchengcc),
[Macasacker](https://github.com/Macasacker),
[Sha1rholder](https://github.com/Sha1rholder),
[adubkov](https://github.com/adubkov), and everyone else whose pull
requests, reviews and commits made this fork possible — and a particular thank you to
**[Neroued](https://github.com/Neroued)** for creating NInfer, maintaining upstream so
well, and for the work this branch builds on.

---

## Upstream README (direct copy)

Everything below is a copy of the upstream
[NInfer README](https://github.com/Neroued/ninfer/blob/master/README.md) as of the latest upstream
sync (`bace20dc` on `origin/master`), unchanged except for one added link to the fork's
[ngram copy proposals](docs/ngram.md) guide.

# NInfer

> Selected checkpoints. Maximum single-GPU inference performance.

NInfer is a from-scratch C++/CUDA inference engine for Qwen3.5 Dense and MoE architectures on a
single NVIDIA GeForce RTX 5090. It runs text, image, and video prompts through a local CLI or
OpenAI-/Anthropic-compatible HTTP APIs. The runtime is deliberately specialized: one GPU, one
resident model, and a startup-fixed capacity of one to eight active requests.

Five official artifacts are available. The quick-start commands use Qwen3.8-27B NVFP4.

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

Each v3 `.ninfer` artifact carries model configuration, encoded weights, logical bindings and
frontend resources. Runtime execution uses those facts with the implemented model and Op
capabilities. You can also [convert your own weights](docs/weight-conversion.md), reuse an official
recipe or choose another supported mixture of formats.

The current engine requires v3 artifacts. Existing official v2 downloads can be
[upgraded locally](docs/weight-conversion.md#upgrade-an-existing-v2-artifact) without downloading
the weights again.

## Quick start

NInfer requires 64-bit Linux, an NVIDIA GeForce RTX 5090, a CUDA toolkit supporting `sm_120a`,
CMake 3.28 or newer, a C++20 host compiler, Ninja, `pkg-config`, FFmpeg development libraries
(`libavformat`, `libavcodec`, `libavutil`, and `libswscale`), and `libcurl >= 7.85`.
CUDA 13.1 is the validated development toolkit; CMake does not impose a CUDA version floor.
The build rejects CUDA architectures other than `sm_120a`.

Build the product binaries:

```bash
git clone https://github.com/Neroued/ninfer.git
cd ninfer

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Tests and benchmarks are excluded from the default build. `cmake --preset release` configures
the same product build; `cmake --preset dev` also enables tests and benchmarks and finds a
Python 3 interpreter. Both presets use `build/` and explicitly reset the build options.
Machine-specific compiler and Python paths belong in the ignored `CMakeUserPresets.json`.
See [build organization and configuration](docs/maintainer/build-system.md) for details.

There is no install target or packaged binary distribution; run NInfer from its source build tree.
Python tools run independently of CMake; the standalone HBM probe has its own
[build command](tools/README.md#standalone-hbm-probe).

Download the artifact used by this example with the Hugging Face CLI:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

Start a long-running text/agent server with two active-request lanes and explicit Device/Host
checkpoint capacity:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

Each request has a 240,000-token logical ceiling. A shared 240,000-token Device KV pool serves
admitted requests; two requests run concurrently when their combined reservations fit. The cache
tiers provide two Device checkpoint slots, eight pinned Host State slots, and 8 GiB of pinned Host
KV beyond the two active StateImages.

Send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

Run a one-shot CLI request with a 32,768-token allocation:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Answer content is written to stdout. Human-readable startup/runtime diagnostics and the CLI-owned
reasoning, timing, throughput, memory, and speculative-decoding report are written to stderr;
reasoning and the result report remain unprefixed product output. On a terminal, weight
materialization uses one transient progress line followed by a compact Engine-ready summary.
Redirected stderr receives persistent readable progress without terminal control sequences. Use
`--log-level debug` for complete startup detail. Option and local input errors remain direct command
diagnostics. Use `--messages FILE` and `--vision` for structured image/video input; see the
[CLI guide](docs/cli.md) and [committed examples](examples/cli/).

## Resource-aware long-context reuse

A reusable prefix checkpoint contains KV and the complete continuation state for its exact prompt
frontier. A Device-resident checkpoint resumes directly. Under pressure, the planner weighs Device
retention, pinned Host State/KV, and eviction by immediate restore work and later reuse cost. Active
requests retain their completion reservations.

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Performance

Published measurements use an RTX 5090. The [performance index](docs/performance.md) links to
per-model run records and the [measurement rules](docs/performance/methodology.md). The tables
below are excerpts from those detailed results.

### Concurrent MTP3 decode

Saturated decode used INT8 group-64 KV, CUDA Graphs, MTP3, and one 8,192-token generation per active
request. Throughput uses aggregate committed decode tokens from complete intervals whose actual
decode batch equaled the configured concurrency. Acceptance covers the complete request wave;
these rates are steady decode (tok/s).

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept | C8 / C1 |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% | 2.88× |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% | 5.67× |
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#decode-saturation) `groupwise-int` | 642.5 / 68.6% | 907.2 / 66.3% | 1,213.5 / 69.6% | 1,380.7 / 68.0% | 2.15× |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `nvfp4` | 143.8 / 48.9% | 267.6 / 48.1% | 461.1 / 45.8% | 766.6 / 46.0% | 5.33× |

### Single-request serving

The serial serving corpus used INT8 group-64 KV, CUDA Graphs, a 1,024-token prefill chunk, and five
fixed seeds after warm-up. The table keeps one short-prefill, one extreme-prefill, and one
structured-output MTP3 point for each published profile; the full context and scenario matrices are
linked from each model below.

| Model profile | 7,680-token prefill | 260,096-token prefill | Structured MTP3 decode |
|---|---:|---:|---:|
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) `groupwise-int` | 17,705.4 tok/s | 5,247.0 tok/s | 779.6 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `groupwise-int` | 3,218.1 tok/s | 1,614.8 tok/s | 193.0 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `nvfp4` | 11,191.5 tok/s | 2,510.6 tok/s | 252.2 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `groupwise-int` | 3,274.7 tok/s | 1,609.7 tok/s | 224.4 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `nvfp4` | 8,340.4 tok/s | 2,203.1 tok/s | 219.8 tok/s |

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. Text evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens to
fit the RTX 5090 after weights. Each score is one sample per problem; model cards contain the
correct/total counts and evaluation notes.

## Startup notes

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` independently selects Vision residency. Qwen3.6-35B-A3B DFlash can be combined with
Vision; it accelerates generated-text decode after multimodal prefill, not Vision encode itself.

## Docker

Build the runtime image on a host with the NVIDIA Container Toolkit:

```bash
docker build --tag ninfer:local .
```

Mount the downloaded model and run the same example server profile:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

## Capabilities and limits

The official artifacts provide the following capabilities, with optional components enabled at startup:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16, INT8, FP8, NVFP4, and K8V4 KV storage;
- offline causal-perplexity scoring;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

The 35B-A3B target additionally supports DFlash with draft windows from one to fifteen for Text and
image/video Vision prompts. Qwen3.8-27B artifacts with the DFlash2 companion weights support
`--spec dflash2 --draft-tokens 7` for the same Text/Vision Engine path, with draft counts 1..15
and either full or optimized proposal heads.

The product boundary remains intentionally small:

- one RTX 5090 and one resident model per Engine;
- a startup-fixed capacity of one to eight active requests with bounded FIFO ingress;
- no request preemption, priority/QoS, active-request swapping, weight offload, multi-GPU, or
  distributed serving;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- model architectures and format/shape combinations use explicitly implemented native paths;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

`--max-context` is each sequence's logical limit. `--kv-capacity` sizes the shared Main Text KV pool
used by active requests and retained prefixes; `auto` resolves the largest legal capacity at
startup from the memory remaining after weights while keeping 1 GiB of sizing headroom. Explicit
capacities remain fixed for the process lifetime.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Ngram copy proposals](docs/ngram.md)
- [Performance](docs/performance.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Weight conversion and custom recipes](docs/weight-conversion.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

Run the relevant `--help` for the exact current option contract.

## Support

NInfer is a personal project that I develop out of interest. If you find it useful and would like
to support its continued development, you can [support the project on Ko-fi](https://ko-fi.com/neroued).

Support is entirely voluntary. It is not a purchase or investment and does not come with financial
returns, promised services or features, or a role in project decisions. The project's direction,
priorities, technical choices, and release schedule remain independently determined by the
maintainer.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
