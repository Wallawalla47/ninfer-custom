# NInfer — custom fork

> **AI disclaimer:** Everything added to this fork, including most of this README, was written with
> AI (mostly Claude Opus 5.5, Qwen3.8-27B running on NInfer, plus a few other AI systems I’ve been
> testing). It is likely to be neither complete nor entirely accurate. This is hobby development.

This is a personal fork of [Neroued/ninfer](https://github.com/Neroued/ninfer). It follows upstream
closely and adds changes on top. The sections below explain what is different, grouped by topic,
with credit given as best my AI agents can where a change came from someone else. The upstream
README follows, copied unchanged, under the "Upstream README" heading. A huge thank you to Neroued
for creating NInfer!

**The short version.** Compared with upstream, this fork:

1. includes changes allowing it to be built and run on Windows
2. includes a new alternative prefix caching system designed and implemented by Claude Opus 5.5 as
   the default. You also simply set an amount of system RAM to be used for prefix caching with
   `--host-cache-mib N`. This option seems to work fantastically well and seems much more
   effective than the prefix caching currently in upstream NInfer. It can also be combined with
   `--prefix-cache-file PATH` to load/store the prefix cache to a file on start/close. You do need
   to close with Ctrl+C rather than closing the cmd window, as Windows does not necessarily allow
   enough time post-window close to dump a large prefix cache to a file
3. in case you want to stick with the original upstream prefix caching system, this is retained
   with a raft of fixes and improvements (as I was working on this prior to going with a new
   design – I found the upstream system to be too complex and fragile) – it is gated behind the
   launch parameter `--use-original-prefix-caching`
4. adds an option to use a faster prefill kernel when using int8 (Hadamard rotated) for KV cache
   (which has a slight penalty to perplexity) by using launch parameter `--fast-prefill-kernel`
5. adds ngram-mod copy drafting (based on an implementation by
   [remesis](https://github.com/remesis)) to significantly increase the speed of copy-heavy
   workloads
6. improves decode speed with speculative decoding about 2-2.5% faster per speculative round, with
   the same output, by overlapping each decode kernel's launch and weight loading with the kernel
   before it
7. adds the ability to offload the vision encoder to system RAM (by specifying
   `--vision-offload on`) based on the work of Valeriy Selitskiy
   ([iamwavecut](https://github.com/iamwavecut))
8. enables the use of YaRN context extension for scaling context up to 1m tokens (by specifying
   `--rope-yarn-factor F`, where F is a number from 1 to 4)
9. fixes the CUDA graph allowance which, depending on the speculative decoding method used,
   sometimes took up much more VRAM than would ever be required
10. allows the user to shrink the default 1024MiB VRAM headroom left available after KV cache when
    using `--kv-capacity auto`, by specifying a custom headroom value with `--vram-headroom-mib N`,
    where N is the number of MiB to leave available
11. allows the user to specify a custom thinking budget message (by specifying
    `--default-thinking-budget N` and `--thinking-budget-message S`, where N is the budget of
    thinking tokens and S is the thinking budget message specified in double quotes “”)
12. includes various improvements (mostly sourced from others credited below) to fix some Qwen tool
    calling issues and leaking thinking tokens etc. Use the launch parameter
    `--tolerant-tool-calls` to fix some broken tool calls
13. accepts more tool-call formats and API options used by agent clients such as Claude Code, Qwen
    Code, Codex and Zed (again mostly based on the work of others credited below)
14. makes improvements to the console logging including an option to turn on colourful logging
    which allows for easier visual tracking of particular figures as the log progresses
    (`--log-colours on`) and some average statistics shown at the bottom of the console view
    (which can be turned off with `--log-stats-panel off`)
15. has a help screen organised by category
16. contains various other fixes and improvements (most of which are outlined below), including
    merging in some PRs on the upstream repo.

I recommend using this with the NVIDIA NVFP4 artifact I’ve uploaded here, which runs a bit faster
than the original artifact based on the Unsloth quant and takes up less VRAM:
<https://huggingface.co/wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-NInferV3>

## Quick start (Windows)

Prerequisites: Visual Studio 2026 (MSVC), the CUDA 13 toolkit, and FFmpeg + curl from vcpkg
(`x64-windows`, at `C:\vcpkg`); adjust the paths at the top of `build_native.bat` for your machine.

```bat
build_native.bat configure
build_native.bat build
```

The server is `build-windows\apps\Release\ninfer-serve.exe`, with the FFmpeg, curl and zlib DLLs
copied next to it. The launch I use on a single 32 GB RTX 5090 (stop any other resident model
first):

```bat
ninfer-serve.exe qwen3_8_27b_nvfp4-nvidia.ninfer --host 127.0.0.1 --port 8080 --max-context 240000 --max-concurrency 2 --spec dflash2 --draft-tokens 7 --lm-head-draft --ngram-draft-tokens 15 --ngram-min-match 12 --kv-dtype int8 --fast-prefill-kernel --preserve-thinking --host-cache-mib 52000 --pending-timeout-ms 900000 --prefill-chunk 4096 --kv-capacity auto --vram-headroom-mib 0 --log-colours on --ngram-archive-mib 2048 --ngram-session-mib 256 --ngram-native-sessions --request-log-jsonl log.json --default-thinking-budget 16384 --thinking-budget-message "Considering the limited time available to the user, I must stop thinking now. Time to act:" --tolerant-tool-calls
```

Add `--prefix-cache-file PATH` to keep the prefix cache across restarts (stop the server with
Ctrl+C). `ninfer-serve.exe --help` lists every option by category.

## Performance: this fork vs upstream

Both benchmarks below compare this fork with **upstream + Windows port**: upstream at the commit
this fork last merged (`bace20dc`) plus only the Windows port (commit `96da12bb` on the branch
`ab/upstream-windows-port-bace20dc`). Everything ran on an RTX 5090 under Windows with the
official Qwen3.8-27B NVFP4 artifact (`qwen3_8_27b_nvfp4-official.ninfer`).

### Agentic coding workload (September 2026)

The closed-loop suite in [`bench/agentic_ab/`](bench/agentic_ab/README.md) replays three
coding-agent sessions plus eleven subagents: 130 requests with fan-outs, a concurrent subagent
pair, compaction, retries, an abort and a solo wrap-up, with prompts of 25K-135K tokens and
thinking on. Each arm's own answers are fed back as an agent client does, and the three main
sessions take their turns in lock-step rounds, so every build meets the same order of session
turns whatever its speed. Every arm completed every request on each of three workload seeds (42,
43, 44), which replay different observations.

Settings:

- **Fork arms** (build `e36f7ee0`): the production launch flags, identical in both arms except
  for the prefix cache. The hybrid-cache arm was then selected with `--use-alt-prefix-caching`;
  today it is the default and the original-cache arm needs `--use-original-prefix-caching`. The
  run also passed `--cuda-graph-allowance-mib 500`, an option since removed now that the
  allowance is measured, and `--vram-headroom-mib` was then named `--kv-headroom-mib`.

  ```text
  --max-context 160000 --max-concurrency 2 --spec dflash2 --draft-tokens 7 --lm-head-draft
  --ngram-draft-tokens 15 --ngram-min-match 12 --kv-dtype int8 --fast-prefill-kernel
  --preserve-thinking --host-cache-mib 52000 --pending-timeout-ms 900000 --prefill-chunk 4096
  --kv-capacity auto --vram-headroom-mib 0 --ngram-archive-mib 2048 --ngram-session-mib 256
  --ngram-native-sessions --default-thinking-budget 16384
  --thinking-budget-message "Considering the limited time available to the user, I must stop
  thinking now. Time to act:" --tolerant-tool-calls
  ```

- **Upstream arm:** the same flags minus those upstream does not have, with the host RAM split
  the fork's original cache resolves from the same 52,000 MiB passed as explicit flags:

  ```text
  --max-context 160000 --max-concurrency 2 --spec dflash2 --draft-tokens 7 --lm-head-draft
  --kv-dtype int8 --preserve-thinking --pending-timeout-ms 900000 --prefill-chunk 4096
  --kv-capacity auto --default-thinking-budget 16384 --host-state-slots 115 --host-kv-mib 30515
  --max-private-continuations 4 --max-long-anchors-per-continuation 25 --max-shared-prefixes 7
  ```

- **Context:** 160,000 tokens, the largest context the upstream build starts with under these
  flags. **Sampling:** temperature 1.0, top_p 0.95, top_k 20, `max_tokens: 64000` on agent turns.

Each cell is the mean over the three seeds, with the lowest and highest seed in brackets; changes
are computed per seed against that seed's upstream run.

| Metric | Upstream + Windows port | Fork, original cache | Fork, hybrid cache (default) |
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
| Workload wall time (min) | 21.6 (19.9-23.1) | 18.0 (16.1-19.8), −17 % | 13.6 (12.6-15.1), −37 % |

- Time to first token includes queueing: up to seven requests are in flight on two lanes, and
  the average queue wait was 12.8 s / 4.7 s / 2.6 s.
- Output tok/s counts decode tokens per second of the engine's own decode time. It splits into
  decode rounds/s (engine speed) and tokens per round (speculative acceptance, which moves with
  what the model happened to write). The fork's ngram drafting supplied 8-11 % of its output;
  upstream has none.
- The two-request decode rows predate the concurrent decode fix (`824e5976`), and upstream
  decoded two requests together for only 25-65 s per seed. The benchmark below measures
  concurrent decode directly, after the fix.

### Concurrent decode (September 2026)

The decode-saturation suite of `tools/bench/run_serve_concurrency.py` starts a fresh server at
each `--max-concurrency` C and decodes C requests at once, each up to 8,192 tokens, with DFlash2
K=7 and `--lm-head-draft` (no ngram drafting), stochastic sampling, `--max-context 32768
--kv-capacity auto`. The fork (with `824e5976`) and upstream alternated point by point in two
passes, C=1 to 8 and back.

| C | Upstream tok/s | Fork tok/s | Fork time per decode round vs upstream (pass 1, pass 2) |
|---|---|---|---|
| 1 | 169.6 | 173.6 | −3.4 %, −1.6 % |
| 2 | 322.9 | 323.5 | −1.6 %, −1.6 % |
| 3 | 433.5 | 440.8 | −0.9 %, −1.0 % |
| 4 | 557.8 | 552.0 | −0.9 %, −0.8 % |
| 5 | 653.2 | 659.3 | −1.2 %, −0.7 % |
| 6 | 753.7 | 754.4 | −0.7 %, −0.7 % |
| 7 | 835.9 | 846.6 | −0.9 %, −0.8 % |
| 8 | 926.0 | 940.6 | −1.0 %, −0.7 % |

Tok/s is the mean of both passes. The fork's decode rounds are faster at every concurrency; tok/s
also moves with speculative acceptance on the sampled text, which is why C=4 is lower despite
faster rounds.

### Running the benchmarks

Build this fork, then the upstream control from the branch `ab/upstream-windows-port-bace20dc`
([details](bench/agentic_ab/README.md#running-it)); stop any other server on the port first:

```bat
build_native.bat configure
build_native.bat build
git worktree add C:\ab\control\src ab/upstream-windows-port-bace20dc
set AB_CONTROL_SRC=C:\ab\control\src
set AB_CONTROL_BUILD=C:\ab\control\build
bench\agentic_ab\build_control.bat configure
bench\agentic_ab\build_control.bat build
```

Agentic workload (about 2.7 hours for three arms on three seeds). The runner reads the model path
and launch flags from `AB_LAUNCH_BAT`, adds `--fast-prefill-kernel` to the fork arms, calibrates
the largest context the control starts with, and writes `report.md` under
`profiles\bench\agentic_ab\`. `treatment` is the fork with its default hybrid cache, `alt` the fork
with `--use-original-prefix-caching`, and `control` upstream:

```bat
set AB_CONTROL_EXE=C:\ab\control\build\apps\Release\ninfer-serve.exe
set AB_LAUNCH_BAT=<a launch .bat with the fork flags above>
py -3.11 bench\agentic_ab\runner.py --arms treatment,alt,control --seeds 42,43,44
```

Concurrent decode, one call per build (repeat `--concurrency` to sweep, or alternate single-point
calls between the builds as above):

```bat
py -3.11 tools\bench\run_serve_concurrency.py --serve build-windows\apps\Release\ninfer-serve.exe ^
  --artifact q38=qwen3_8_27b_nvfp4-official.ninfer --mode dflash2_7 --suite decode-saturation ^
  --max-context 32768 --kv-capacity auto --concurrency 1 --concurrency 8 --output profiles\bench\cc-fork
```

Upstream uses the same command with the control checkout's own copy of the script and its
`ninfer-serve.exe`, because its server writes an older request-log schema. On Windows that copy
needs `wait_for_final_throughput` from commit `21bca1c0`, which reads the final statistics
interval that Windows otherwise loses when the server is stopped.

## What this fork changes

Each topic lists everything that affects it, whether written here or taken from elsewhere.
"Upstream PR" means an open pull request on `Neroued/ninfer` that this fork merged before upstream
did.

### Hybrid prefix cache (the default)

Designed around how Qwen3.5-family models work: most of their layers are linear-attention (GDN)
layers, whose recurrent state cannot be rebuilt from the KV cache, so resuming a prompt needs the
KV of every earlier token plus a saved state at the exact token where the new prompt continues.
The cache keeps the two apart and stores each as cheaply as it can
([design](docs/maintainer/hybrid-prefix-cache-spec.md)).

- **KV is cached per 64-token block, keyed by content** (its tokens, any image in it, and the
  block before it), in a radix tree. A shared system prompt is stored once and costs its GPU pages
  once at any concurrency.
- **Saved model state is sparse.** Snapshots are taken only at useful points: the end of the
  system prompt and tools, client cache breakpoints, the start of the assistant reply, the end of
  each answer, and a few points spread back through long history. Most cost no extra prefill work
  because they fall on prefill chunk boundaries.
- **Three tiers.** Free VRAM after the model becomes GPU block cache (`--kv-capacity` defaults to
  `auto`); `--host-cache-mib` (default 8192, `0` = GPU only) is one pinned host RAM pool that
  blocks and snapshots share, split by how much prefill time each entry saves; and
  `--prefix-cache-file PATH` saves the host tier on shutdown and reloads it at startup. A file
  from a different model, KV format or `ninfer-serve` build is ignored and replaced. Windows ends
  a closing console window's process about 5 seconds after the close, so stop large caches with
  Ctrl+C.
- **Restores overlap the request's own work.** Host RAM copies run on a separate stream in layer
  order and each layer waits only for its own data, so a long restored context costs little more
  than its new tokens.
- **Parallel requests with a new shared prefix prefill it once.** Later requests wait for the
  first one's snapshot where the prompts diverge. Four requests with a new 13.9K-token system
  prompt: mean time to first token 1.48 s instead of 3.52 s.
- Everything except `--host-cache-mib` is derived from `--max-concurrency` and `--prefill-chunk`;
  `--device-snapshot-slots`, `--cache-taps-per-request`, `--cache-tap-ladder` and
  `--cache-tap-min-gap` are optional overrides.
  Code: `src/runtime/prefix_cache/` (block tree, eviction, snapshot planner, cost model) and
  `src/models/qwen3_5/program/prefix/` (GPU and host copies, admission, cache file).

### Original prefix cache: `--use-original-prefix-caching`

Upstream's checkpoint catalog (a saved state plus the KV at that exact point), kept with this
fork's fixes. It is configured with `--host-cache-mib` or upstream's separate capacity flags,
which require `--use-original-prefix-caching`. Details:
[resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md).

- **GPU KV grows with the answer instead of being reserved up front.** A request reserves its
  prompt plus a 4,096-token output window and extends it as the answer grows, so `max_tokens:
  64000` no longer squeezes out the cache. When the pool is full, the least recently used idle
  entries are freed, just enough for the next step; only if nothing can be freed does the answer
  end early with `finish_reason: "length"`.
- **Memory pressure moves cache to host RAM before deleting it**, for private conversations and
  shared prefixes alike.
- **Eviction takes the oldest entries first, and only as many as needed**, in one least recently
  used order across private conversations and shared prefixes; clearing everything is a last
  resort, and a plan that cannot be carried out makes the request wait and re-plan.
- **A checkpoint loses its value only when its own conversation has moved past it.** Builds on
  upstream PR #300 by [pkochubey](https://github.com/pkochubey) (upstream issue
  [#178](https://github.com/Neroued/ninfer/issues/178)).
- **One host RAM setting, `--host-cache-mib`**, sizes the saved-state pool for the checkpoints
  the engine will really create, spends spare room on more long anchors, gives host KV the rest,
  and refuses to start rather than over-commit.
- **Long anchors are placed automatically** at message boundaries, spaced further apart further
  back (`--long-anchor-spacing`), and the one whose loss costs least coverage is replaced first.
- **Aborted requests keep their prefilled prefix**, so a retry carries on from there.
- **More time to find a cache plan:** the admission search budget scales from 5 ms to 250 ms with
  the request's cost (upstream issue [#229](https://github.com/Neroued/ninfer/issues/229),
  approach suggested there by Gene0Liu), and stays 250 ms while another request runs.
- **The shared-prefix list no longer fills up for good**: the least recently used automatic entry
  is replaced (upstream issue [#251](https://github.com/Neroued/ninfer/issues/251), approach
  suggested there by albertov).

### Faster prefill: `--fast-prefill-kernel`

Opt-in on `ninfer-serve`, `ninfer-perplexity` and `ninfer_bench`, for `--kv-dtype int8`:

- A FlashAttention-2 style INT8 prompt-attention kernel
  (`src/ops/softmax_attention/dense/causal_cache/prompt_i8_fast.cuh`): each warp keeps its query
  rows, scores and output in registers, the KV stays INT8 and double-buffered, and P×V runs on FP16
  Tensor Cores. At 131K context it runs at about 313 TFLOP/s instead of 193.
- The effective `--prefill-chunk` is rounded down to whole GPU waves (896 tokens for this model on
  170 SMs), so `4096` runs as `3584`.

Against the flag off (Qwen3.8-27B NVIDIA NVFP4, int8 KV): new-prompt prefill +3.8 % at 16K,
+14.7 % at 64K and +24.9 % at 128K; perplexity 4.1713 against 4.1679 (+0.08 %; BF16 KV scores
4.1695).

### Decode speed

- **Overlapped decode kernels.** Kernels in the decode CUDA Graph launch as programmatic
  dependents of the kernel before them (PDL): weight-streaming kernels load their first weight
  tiles while the previous kernel runs, and release the next kernel only after their own main
  loop. Kernels that fill more than half the GPU release it only when they finish, so a following
  one-wave kernel no longer squeezes onto the few free SMs (`824e5976`). A 28-byte memset that cost
  about 70 µs per round on Windows is now a kernel, and the engine no longer waits for the
  recurrent-state fold before preparing the next round. Output is unchanged token for token;
  greedy decode is 2.2-2.5 % faster per round (DFlash2 K=7, 16K and 60K context). Ideas tried and
  dropped are in [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md).
- **Ngram copy drafting with more than one concurrent request.** Ngram drafting proposes the next
  tokens by copying matching text from earlier in the context, alongside MTP/DFlash/DFlash2. The
  single-request version is the original work of [remesis](https://github.com/remesis) in the
  [remesis/ninfer](https://github.com/remesis/ninfer) fork (upstream issue
  [#234](https://github.com/Neroued/ninfer/issues/234)); this fork extends it to
  `--max-concurrency` above 1. See [ngram copy proposals](docs/ngram.md).
- **Several requests can prefill at the same time**, overlapping one request's prefill with other
  requests' prefill and decode. By David Oelfke in the [gzenz/ninfer](https://github.com/gzenz/ninfer)
  fork (commit `576e72ea`).
- **Short prefill steps over long contexts use split-KV attention**: 32 new tokens against 180K
  cached tokens take 1.06 ms per attention layer instead of 9.5 ms.
- **Kernel tuning from upstream PRs:** the fused SwiGLU TMA partial tile (#264), sigmoid gate in
  the causal reduce (#268) and text `rmsnorm_rope` route (#273), by Michael Dementii; tuned Q6
  34,816×5120 dispatch (#284, [bingchengcc](https://github.com/bingchengcc)); Q5 linear K-split
  sized to the token count (#292, [giveen](https://github.com/giveen)); and, adapted from
  [llmq](https://github.com/IST-DASLab/llmq) (IST-DASLab, Erik Schultheis) by
  [DuncanBetts](https://github.com/DuncanBetts), a fused NVFP4 RMSNorm + quantise for the attention
  input projection (#305) and a single-pass target log-probability kernel (#307).

### Tool calls and reasoning output

- **More tool-call formats**: the XML forms emitted by Claude Code and other agent tools
  (`<function name="…">`, `<invoke>`, `<function_calls>`, short `<param>` tags), also while
  streaming. Upstream PR #300 by [pkochubey](https://github.com/pkochubey) (upstream issue
  [#276](https://github.com/Neroued/ninfer/issues/276)).
- **Repeated tool-call parameters keep the last value** instead of turning the call into plain
  text. Upstream PR #299 by [adubkov](https://github.com/adubkov).
- **Quoting `</think>` no longer ends the reasoning early**: it only ends the reasoning when a line
  break or the end of the turn follows. Adapted from upstream PR #309 by Fedor Suchkov.
- **`--tolerant-tool-calls`** keeps a good call followed by junk, a final call cut off by the output
  limit (if a parameter is complete), repairs a missing `>` after the function name, and returns
  calls to undeclared tools. By David Oelfke in the [gzenz/ninfer](https://github.com/gzenz/ninfer)
  fork, ported onto this fork's parser.

### API and client compatibility

- **llama.cpp-style model details on `/v1/models`**: upstream PR #162 by
  [Hector Ramon Jimenez (hecrj)](https://github.com/hecrj).
- **`ignore_eos` on chat completions**: upstream PR #197 by [Thireus](https://github.com/Thireus).
- **GitHub Copilot and other agent-host requests** (`custom` tools, advisory `tool_choice` /
  `strict` / `parallel_tool_calls`, tool names up to 256 bytes, `--usage-chunk-choice`): the
  serving commits of upstream PR #316 by [paq85](https://github.com/paq85) (Damian Sromek).
- **Responses API options used by Codex and Zed Agent** (`reasoning.summary`,
  `include: ["reasoning.encrypted_content"]`): upstream PR #295 by
  [Macasacker](https://github.com/Macasacker), based on an earlier PR by
  [Sha1rholder](https://github.com/Sha1rholder).
- **`response_format` `json_object` / `json_schema` is accepted** (not enforced), a **tool call
  cut off by the output or context limit is reported as cut off** (`length` / `max_tokens`) rather
  than as a tool call, and a request ending with an assistant message **continues that reply**
  (thinking off only): upstream PR #300 by [pkochubey](https://github.com/pkochubey).

### Stability

- **Out-of-memory no longer stops the engine**: only the affected requests fail, the engine resets
  and carries on with the queue. By David Oelfke in the gzenz/ninfer fork (commit `3f3272d6`).
- **Recovery really leaves the engine empty**: if cleanup leaves cache pages or states with no
  owner, the engine rebuilds its cache stores instead of looking full and refusing every request.
- **No resource-underflow HTTP 500s in the original cache**: releasing a request whose shared
  cached pages became exclusive to another request now credits that request's entitlement.
- **Cache planning cannot race with itself**: by [Gideon Zenz (gzenz)](https://github.com/gzenz)
  in the gzenz/ninfer fork (commit `c53e025c`).
- **No shared-memory overflow at very long (YaRN) contexts** in split-KV decode attention. By
  David Oelfke in the gzenz/ninfer fork (commit `7a876cf7`).

### Models, conversion and vision

- **GGUF files as conversion sources**: upstream PR #282 by [giveen](https://github.com/giveen).
- **NVIDIA ModelOpt NVFP4 and FP8 checkpoints** as conversion sources (their scale layouts),
  with the `qwen3_8_27b_nvfp4_nvidia` recipe storing the output head as FP8.
- **Third-party Qwen checkpoints convert cleanly**: missing or non-standard tokenizer settings
  that the runtime requires are rebuilt during conversion.
- **Quasar NVFP4 conversion** with DFlash2 heads and an indexed proposal head (`--proposal`), so
  rebuilt artifacts support `--lm-head-draft`.
- **A `qwen3_8_27b_q6` recipe**, the Q6 fused gate/up projection shape, and a `grouped_mse`
  scale-search method for groupwise quantisation.
- **Q8 MTP** and a **general BF16 GEMM fallback** for shapes without a dedicated kernel.
- **`--rope-yarn-factor F`** for YaRN context extension (F from 1 to 4, up to 1M tokens of context).
- **`--vision-offload on`** keeps the vision tower in pinned system RAM instead of VRAM and streams
  it to the GPU while an image is encoded (off by default), and **`--vision-max-merged N`** bounds
  the merged vision tokens per image or video (64–32768). Based on the original work by
  [Valeriy Selitskiy (iamwavecut)](https://github.com/iamwavecut), rewritten for this engine.

### Windows

- **Native build and run** with MSVC and CUDA (see [Quick start](#quick-start-windows)): static
  CUDA runtime, FFmpeg/curl from vcpkg, non-RDC NVFP4 kernels, TMA descriptors staged into device
  memory by a kernel, a built-in PNG decoder for the vision path, and drive letters in converter
  recipe paths.
- **Running on another PC** needs an RTX 50-series GPU (the build targets `sm_120a`) and an NVIDIA
  driver of 580 or later (CUDA 13); no CUDA toolkit is needed. Copy the DLLs next to
  `ninfer-serve.exe` and install the Visual C++ redistributable if it is missing.

### Options and console

- **`--log-colours on`** colours the console statistics, and a **session statistics panel**
  beneath the log shows session and last-ten averages of TTFT, cache hit rate, prefill and decode
  speed and drafter acceptance (`--log-stats-panel off` removes it).
- **Grouped `--help`** by category on `ninfer-serve` and the `ninfer` CLI, covering flags that
  were previously undocumented, with separate sections for the two prefix caching systems. The CLI
  statistics are coloured too.
- **`--vram-headroom-mib N`** sets how much GPU memory `--kv-capacity auto` leaves spare after
  sizing the KV pool (upstream always leaves 1 GiB).
- **Engine messages are ordinary log records** (`engine | ...`) that scroll above the statistics
  panel. Routine ones, such as a Device KV lease that grew by releasing retained cache, are `debug`
  and appear only with `--log-level debug`; warnings and errors always appear.
- **Measured CUDA Graph allowance**: the KV sizing reserves 64 MiB plus 4 MiB per decode-graph
  executable, measured on an RTX 5090 across every speculative mode and concurrency (DFlash2 with
  ngram drafting at `--max-concurrency 2` reserves 160 MiB and uses about 62 MiB, where upstream's
  estimate reserves 1,920 MiB). Startup reports the memory the graphs used and warns if it ever
  exceeds the allowance.
- **`--thinking-budget-message S`** sets the message inserted when a request reaches its
  `--default-thinking-budget N`, and **`--chat-template`** loads the chat template from a file.

### Kept in sync with upstream

Upstream `master` is merged regularly. Once upstream adopts a change listed above, it is removed
from this README.

## Model artifacts

- **[Qwen3.8-27B-NVIDIA-NVFP4-NInferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-NInferV3)**
  (recommended): [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4), the
  Model Optimizer mixed NVFP4/FP8 checkpoint, converted with the `qwen3_8_27b_nvfp4_nvidia` recipe
  in `tools/convert/official_recipes.py`. The weights are imported bit-exact except the output
  head (NVFP4 → row-scale FP8), with the DFlash2 draft model and an indexed 131,072-row proposal
  head for `--lm-head-draft`.
- **[Qwen3.8-27B-Quasar-NinferV3](https://huggingface.co/Wallawalla47/Qwen3.8-27B-Quasar-NinferV3)**:
  [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4),
  the QAT-trained NVFP4 checkpoint, converted with `tools/convert/quasar_nvfp4.py`, with the same
  DFlash2 draft model and proposal head.

Both are single-file `.ninfer` artifacts for an RTX 5090 (`sm_120a`); each Hugging Face page has
the creation outline and conversion report. The [Quick start](#quick-start-windows) launch works
for either.

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
