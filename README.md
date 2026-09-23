# NInfer — local fork

> **AI disclaimer:** Everything added to this fork — the local features and this README
> included — has been written with AI (mostly Qwen3.8-27B running on NInfer, with a few
> other AI systems as well), so it is likely to be neither complete nor entirely
> accurate. This is simply hobby development.

This repository is a personal fork of [Neroued/ninfer](https://github.com/Neroued/ninfer),
maintained on the branch `master`. It tracks upstream while adding
local features on top. This section summarises, in high terms, everything merged in from
other sources and everything built locally; the upstream README follows, unmodified, below
the "Upstream README" heading.

## Inference performance: this fork vs upstream (A/B benchmark)

Both arms serve the same official NInfer Qwen3.8-27B NVFP4 artifact
(`qwen3_8_27b_nvfp4-official.ninfer`) on a NVIDIA GeForce RTX 5090, run at the same
max-context (180000 — the largest context at which the upstream arm starts on this card;
see below) and replay the same synthesized agentic workload: 14 requests shaped from
the production request log (multi-turn tool-agent sessions, ~82,000-token shared prefix,
median prompt ≈ 121K tokens, two concurrent-request pairs including a shared-prefix
cache-contention pair, max-concurrency 2).

| Metric | Upstream + Windows port | This NInfer-custom fork | Δ |
|---|---|---|---|
| Avg TTFT (s, lower better) | 28.1 | 14.6 | -48.1% |
| TTFT median (s, lower better) | 25.9 | 11.3 | -56.6% |
| Prefix-cache hit rate (hit/prompt tokens) | 19.8% | 66.3% | +46.5 pp |
| Total cache-hit tokens (of 1,649,215 prompt) | 326,780 | 1,093,256 | +234.6% |
| Prefill tok/s, cold (root) requests, median | 4,159 | 4,220 | +1.5% |
| Output tok/s | 173 | 181 | +4.6% |

- Avg/median TTFT over all 14 completed requests per arm.
- Cold (root) prefill = requests that did a full uncached prefill (n: upstream 11, fork 2).
- Output tok/s = total completion tokens / total decode wall time (completion tokens:
  upstream 12,639, fork 6,543; thinking on for both arms; per-request decode rates are
  comparable — see the per-request tables in the full report).
- Cache hits by reuse path (requests / hit tokens):

  | Path | Upstream | This NInfer-custom fork |
  |---|---|---|
  | private_long_anchor | 0 / 0 | 9 / 766,476 |
  | private_response_replay | 3 / 326,780 | 3 / 326,780 |
  | root | 11 / 0 | 2 / 0 |

**Workload note.** This was tested on what is designed to be a fairly representative
agentic workload (the synthetic sessions described above). If the workload includes
more copying work, the ngram-based drafting implementation
(`--ngram-draft-tokens` / `--ngram-min-match`) would boost output tok/s further.

**Configuration and launch parameters.** Model: the official NInfer Qwen3.8-27B NVFP4
artifact `qwen3_8_27b_nvfp4-official.ninfer`; GPU: NVIDIA GeForce RTX 5090; max-context
180000 for both arms; max-concurrency 2; `--kv-dtype int8`. This run predates the retention
changes summarised under **Prefix caching** below, so the fork-arm figures describe that
earlier tree — its launch list is the pre-budget component form of the cache flags, kept here
as the record of what was measured.

Launch parameters, this fork (all flags):

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

Launch parameters, upstream + Windows port (same list minus the fork-only flags it does
not support, which are dropped: `--ngram-draft-tokens`, `--ngram-min-match`,
`--kv-headroom-mib`, `--log-colours`, `--ngram-archive-mib`, `--ngram-session-mib`,
`--ngram-native-sessions`, `--cuda-graph-allowance-mib`, `--thinking-budget-message`):

```text
--host 127.0.0.1 --port 8080 --max-context 180000 --max-concurrency 2 --spec dflash2
--draft-tokens 7 --lm-head-draft --kv-dtype int8 --preserve-thinking
--host-kv-mib 24000 --pending-timeout-ms 900000 --prefill-chunk 2048 --kv-capacity auto
--host-state-slots 64 --max-private-continuations 32
--max-long-anchors-per-continuation 8 --max-shared-prefixes 32
--default-thinking-budget 16384
```

The upstream serve bakes an automatic 1 GiB KV headroom into `--kv-capacity auto` that
has no flag to lower (the fork's `--kv-headroom-mib 0`), so it cannot start at the
production max-context 220000 on this 32 GiB card; both arms therefore run at the same
calibrated 180000 so the arms stay comparable. Thinking is on for both arms
(`--default-thinking-budget 16384` + `--preserve-thinking`; both upstream-supported).

Workload: 14 requests, seed 42; groups replayed in order, requests within a group sent
concurrently. The full A/B rig — workload generator, runner, watchdog and control build
driver — lives in [`bench/ab/`](bench/ab/README.md) so the test can be replicated with
your own builds and flags.

## Merged from upstream and other sources

**Upstream pull requests** (`Neroued/ninfer`):

- **PR #162** — llama.cpp-compatible model metadata on `/v1/models` (`n_vocab`, `n_ctx`,
  `n_ctx_train`, `n_embd`, `n_params`, `size`, `ftype`), by
  [Hector Ramon Jimenez (hecrj)](https://github.com/hecrj). Re-implemented against the v3
  `src/models` layout, which replaced the `src/targets` layout the PR was written for.
- **PR #197** — honour `ignore_eos` on chat completions, by
  [Thireus](https://github.com/Thireus).
- **PR #264** — take a partial last M tile in the fused SwiGLU TMA route, by Michael
  Dementii.
- **PR #268** — fold the sigmoid gate into the causal reduce epilogue, by Michael
  Dementii.
- **PR #273** — register the text profile of `rmsnorm_rope` and route to it, by Michael
  Dementii.
- **PR #281** — correct the Q5 parent-shape explanations and route table, by
  [Minnnn](https://github.com/Minnnn).
- **PR #282** — read GGUF (any ggml quantisation level) as a conversion source, by
  [giveen](https://github.com/giveen).
- **PR #284** — tune the Q6 34,816×5120 dispatch and report the curve, by
  [bingchengcc](https://github.com/bingchengcc).
- **PR #292** — ladder the Q5 linear K-split capacity to the token count, by
  [giveen](https://github.com/giveen).
- **PR #295** — accept `reasoning.summary` and `include: ["reasoning.encrypted_content"]`
  in OpenAI Responses API requests, unblocking harnesses such as Codex and Zed Agent, by
  [Macasacker](https://github.com/Macasacker), rebasing
  [Sha1rholder](https://github.com/Sha1rholder)'s original PR #148. The PR's new test
  assertions were adapted locally to compare JSON objects order-independently, because
  `RequestJson` is an `ordered_json` whose object equality is insertion-order sensitive.
- **PR #299** — keep the last value on a duplicate tool-call parameter (JSON object
  semantics) instead of falling back to text, with the repair counted as
  `duplicate_parameters_repaired` in the parse diagnostics and a short markup snippet
  logged when a fallback does occur, by [adubkov](https://github.com/adubkov).
- **PR #300 (tool-call parsing)** — recognise the XML tool-call forms emitted by Claude Code and
  other agent harnesses: `<function name="…">` attribute names, `<invoke>`, `<function_calls>`
  containers, and the `<param>`/`</parameter>` short aliases, with attribute-token-boundary name
  lookup, matching opening/closing tag pairs, and a streaming decoder that sees every variant
  rather than only `<tool_call>`. Resolves upstream issue
  [#276](https://github.com/Neroued/ninfer/issues/276), by
  [pkochubey](https://github.com/pkochubey). The PR's conflicting-duplicate rejection was adapted
  to the last-value-wins rule from PR #299 above.
- **PR #300 (`response_format`)** — accept `{"type":"json_object"}` and `{"type":"json_schema"}` on
  chat completions instead of refusing everything but `{"type":"text"}`, so harnesses that always
  send a response format (Hermes-style clients among them) are not rejected, by
  [pkochubey](https://github.com/pkochubey). The type is not enforced: NInfer still has no
  constrained decoding, and `docs/serving.md` states that rather than claiming the schema is
  honoured.
- **PR #300 (truncation stop reasons)** — report a truncated answer as truncated rather than as a
  completed tool call: `length` on chat completions and `max_tokens` /
  `model_context_window_exceeded` on Messages now win over `tool_calls` / `tool_use` when the
  output limit or context capacity cut the call short, by
  [pkochubey](https://github.com/pkochubey). A client that trusts the terminal reason would
  otherwise act on a call whose arguments may be incomplete. The partial call is still streamed.
- **PR #300 (assistant prefill)** — accept a trailing assistant message as an assistant-prefill
  continuation on chat completions as well as Messages, and allow that final turn to carry
  reasoning content or tool calls, by [pkochubey](https://github.com/pkochubey). This lets an
  agent client hand back an output-limited partial assistant turn and have the Engine continue it
  in place. Thinking-enabled prefill stays refused, unlike the PR: the template places the
  continued content inside an ambiguous reasoning opener, which `test_assistant_continuation`
  pins, so only the PR's reasoning/tool-call relaxation is taken here.
- **PR #300 (unreachable checkpoints)** — exclude a checkpoint the incoming request cannot reach
  from the portfolio recovery-loss accounting, so the planner stops crediting an owner for a
  prefix hit that request could never take, by [pkochubey](https://github.com/pkochubey).
  Resolves upstream issue [#178](https://github.com/Neroued/ninfer/issues/178).

Recurring merges from upstream `master` additionally bring in ongoing kernel and build
work: NVFP4/Q8/sparse-MoE dispatch tuning, whole-tile W4A4 TMA scale routing, the real
Jinja chat-template interpreter (`third_party/llama-jinja`), state-cache benchmark
fixtures, and the per-component CMake reorganisation.

**Other sources:**

- **Ngram copy drafting (the original C=1 implementation)** — from the
  [remesis/ninfer](https://github.com/remesis/ninfer) fork, by
  [remesis](https://github.com/remesis): exact-checked CPU copy proposals alongside
  MTP/DFlash/DFlash2 with optional bounded session retention (upstream issue
  [Neroued/ninfer#234](https://github.com/Neroued/ninfer/issues/234)), cherry-picked from
  remesis's branch. The C=1 implementation is remesis's original work; this fork's
  contribution is the extension to C>1 concurrency (below).
- **Vision overlay residency (RAM offload of the vision tower)** — original work by
  [Valeriy Selitskiy (iamwavecut)](https://github.com/iamwavecut), from the
  previous-generation tree's PR #73 ("content-addressed KV host cache + vision overlay
  residency"): the overlay port, 35B support, and the residency flags. Re-implemented for
  this V3 engine as `--vision-residency overlay` (local re-implementation, below).
- **Tolerant tool-call recovery (`--tolerant-tool-calls`)** — original design and
  implementation by [David Oelfke (gzenz)](https://github.com/gzenz) in the
  [gzenz/ninfer](https://github.com/gzenz/ninfer) fork (September 2026), commits
  `b2267e06` ("add --tolerant-tool-calls to recover complete Qwen calls with malformed
  wrapper or suffix output"), `0ce6e3f6` ("recover a single truncated final call in tolerant
  mode when closing tags are cut off at region end"), `0f3c9f55` ("recover function name
  in tolerant mode when the closing '>' is omitted before a parameter tag"), `38709834`
  ("keep a value cut by the output budget, and keep a truncated final call only when at
  least one parameter is complete — otherwise the region stays text, with the operational
  record guarded to match"), and `44f2c9c3` ("keep complete calls whose name is not in the
  declared tools"). gzenz's version
  sits on a divergent canonical parser, so the recovery logic was **ported** onto this fork's
  multi-marker parser (PR #300's marker recognition and PR #299's last-value-wins duplicate
  handling are preserved): in tolerant mode, a complete call followed by a trailing suffix or
  a malformed second call keeps the good call with the tail discarded (`truncated_tail`
  diagnostic, logged at Info severity), a single final call cut at the region end by the
  output budget is kept with its partial value, a missing closing bracket after the function
  name is recovered by an identifier-run scan, and undeclared tool names stay structured. The
  strict parser — the default — is unchanged.
- **Materialization seal-window atomicity** — original fix by
  [Gideon Zenz (gzenz)](https://github.com/gzenz) in the
  [gzenz/ninfer](https://github.com/gzenz/ninfer) fork (September 2026), commit
  `c53e025c` ("fix(planner): make materialization seal atomic via a seal-window claim"):
  the materialization assess-to-seal window raced on a victim's continuation slot
  generation — a concurrent demote bumped it, so the seal's revalidation saw stale
  policy state and the planner's throw propagated to the Engine worker's last-resort
  handler, failing the whole instance. An atomic claim (compare-and-swap with bounded
  backoff) now serializes the window, and a lost claim or failed seal falls back to the
  root-maximal eviction target so the request re-prefills instead of failing admission.
  The commit's unrelated search-budget tuning and debug instrumentation were not taken;
  the tolerant-parser tweak it bundled is already covered by the multi-marker recovery
  above.
- **Split-KV page-limit floor (small-t split safety)** — original fix by
  [Gideon Zenz (gzenz)](https://github.com/gzenz) in the
  [gzenz/ninfer](https://github.com/gzenz/ninfer) fork (September 2026), commit
  `7a876cf7` ("fix(attn): floor split-KV split count by page limit to prevent shared-memory
  OOB at large context windows"): for the `SmallTSplitScale == 1` geometry, each split
  stages at most 64 physical-page IDs into `__shared__ physical_pages_s[64]`, so the
  number of splits must keep keys-per-split within one page table; the old code clamped
  splits to `SmallTMaximumSplits`, which under-provisioned at (YaRN-extended) large
  windows and overflowed shared memory. The split count is now floored to
  `div_up(window, 3968)` (62 pages plus a 2-page rounding margin) and capped at 256
  (the split reducer), applied consistently in the host-side `causal_small_t_split_upper_bound`
  and the device-side split selection.
- **Engine OOM recovery** — original work by
  [Gideon Zenz (gzenz)](https://github.com/gzenz) in the
  [gzenz/ninfer](https://github.com/gzenz/ninfer) fork (September 2026), commit
  `3f3272d6` ("Improve engine OOM recovery and qwen3_6 prefill stream safety"): a
  `std::bad_alloc` (typically device-KV reservation failure) no longer crashes the
  Engine worker. The materialization reserve in admission is guarded and fails only the
  affected request with a retryable `Overloaded` error; the worker loop catches OOM,
  errors the active/materializing requests through a per-step-guarded
  `force_complete_error`, resets the scheduler and program state while leaving pending
  requests in the FIFO, and continues with a bounded admission backoff — after
  `kOomMaxRecoveries` (8) consecutive failed recoveries it fails all pending instead.
  Recoverable `logic_error`s take the same path, and the fatal crash handler now logs a
  `WORKER CRASH` diagnostic. The commit's qwen3_6 prefill stream-sync hunk was not
  taken (that target does not exist in this fork).
- **Concurrent staged-prefill lanes** — original work by
  [Gideon Zenz (gzenz)](https://github.com/gzenz) in the
  [gzenz/ninfer](https://github.com/gzenz/ninfer) fork (September 2026), commit
  `576e72ea` ("Support concurrent staged-prefill lanes and apply YARN scaling to DFlash
  target RoPE positions"): staged prefill ownership changed from a single optional lane
  to a per-lane bitmask in the Scheduler, so several requests may prefill
  simultaneously. A staged prefill holds no resource transaction (materialization
  already committed the lane's full reservation), so admission is no longer gated
  behind a prefill owner — waiting requests can be admitted to free lanes while others
  are prefilling, and every prefill unit boundary re-arms the admission check, letting
  one request's prefill overlap the prefill and decode of the rest. Each worker
  boundary still advances exactly one staged lane (lowest index first, skipping
  capture-offering lanes). The commit's bundled qwen3_6 YARN RoPE-position scaling was
  not taken (that target does not exist in this fork; this fork applies YaRN through
  the RoPE op instead).

## Local changes

**Platform**

- **Native Windows build and run** — MSVC + CUDA on Windows: static CUDA runtime,
  vcpkg-resolved FFmpeg/curl with runtime DLL staging, non-RDC NVFP4 kernels, `ws2_32` and
  `UTF8PROC_STATIC`, TMA descriptors staged through pinned buffers (around MSVC's
  `__grid_constant__` limitation), Windows mapped-file and `ReadOnlyFile` rules in the
  artifact Reader, and re-application of the Windows build after upstream's per-component
  CMake reorganisation.
- **PNG image support in the vision path** — the prebuilt Windows vcpkg FFmpeg tree ships
  without the PNG decoder, so a native PNG decode path (`NINFER_MEDIA_NATIVE_PNG`) was
  added; the vision path accepts `.png` images on these builds.
- **Converter recipe references on Windows** — the `--recipe FILE[:function]` reference
  treats a colon inside the path (such as a Windows drive letter) as part of the path; only
  a colon followed by a bare function name selects the entry function.

**Operability and UX**

- **Colourful console logging** — colourised operational and CLI statistics output for
  visually tracking throughput, cache-reuse and memory statistics (`--log-colours`).
- **Categorised help screens** — the flat CLI and serve option dumps were replaced with
  named sections (Context, KV Cache, Speculative Decoding, Vision, Sampling, Networking &
  Resources, …), making the flags easier to read and documenting previously undocumented
  flags.
- **`--kv-headroom-mib`** — manually specify the headroom used by the automatic KV
  capacity sizing (upstream keeps a fixed 1 GiB; this makes it an operator choice).
- **`--cuda-graph-allowance-mib`** — manually specify the CUDA Graph memory allowance
  instead of the engine's automatic value.
- **`--thinking-budget-message`** — manually specify the message appended when a request
  hits its thinking budget, replacing the built-in end-of-thinking control message.
- **llama.cpp-compatible `/v1/models` metadata** — the local re-implementation of
  upstream PR #162 (above).
- **`--chat-template`** — load the artifact chat template from a file.

**Prefix caching**

The reuse model itself is upstream's: a prefix hit requires a *complete* checkpoint — a StateImage
plus the Main (and speculative-backend) KV for that exact prompt frontier — so cache capacity is
accounted in checkpoints and KV page groups, not in tokens. What is local is how that checkpoint set
is leased, sized, retained and reported under Device pressure. Together these are the changes that
stopped long multi-turn agentic sessions on this box re-prefilling almost every request; the
derivation and the invariants live in
[paged KV context store](docs/maintainer/paged-kv-cache.md) and
[HTTP serving](docs/serving.md).

- **Bounded, elastic Device KV lease** — upstream reserves each request's full
  prompt-plus-effective-output KV at admission and holds it to completion, so a client that sends
  `max_tokens: 64000` for generations averaging a few hundred tokens parks its whole output budget
  as reserved KV, and a single long-prompt request can reserve the entire pool away from the prefix
  cache. The fork leases a bounded window (the prompt plus `max(prefill-chunk, 4096)` output tokens)
  and extends it at a decode-round boundary as generation approaches the window, up to the same
  output ceiling. Every round is covered by the reservation already held, so an extension is a pure
  pool reservation — no copy, no graph change, no measurable decode or TTFT cost. When the pool
  cannot grant the next ladder rung (the margin, then two page groups, then one), the lease settles
  and the request ends at `finish_reason: length` instead of failing admission or throwing
  mid-step.
- **Demote instead of destroy** — a private conversation prefix that loses Device KV is demoted to
  Host (device pages freed, host copy kept, so the active context always fits) whenever the Host
  tier can take it, for every private owner rather than a configured subset. Under upstream's
  escape hatch the pressure fallback cleared both tiers at once, which turned one over-capacity
  request into a wiped cache and a self-sustaining zero-hit steady state.
- **Recency-ordered eviction ladder** — when Host cannot absorb the pressure either, the fallback
  ranks every prefix — private conversations and shared prefixes in one order — by its latest hit
  or publication and sacrifices the oldest rung by rung: rung k fully evicts the k oldest and keeps
  the rest (demoting private and shared prefixes alike wherever Host can take them), and the
  incremental materialization search may only fully evict inside that LRU tail — so the cache never
  trades a more recent prefix for an older one's Device KV. The clear-all target remains only as
  the guaranteed liveness backstop. The fork's `--preserved-recent-prefixes` escape hatch (a
  configured set of pinned owners, defaulting to none) is removed; the ladder covers every owner
  instead.
- **One Host RAM budget — `--host-cache-mib`** — upstream sizes the retention tier with two
  independent allocations (`--host-state-slots`, `--host-kv-mib`) plus three catalog limits, so the
  RAM actually pinned is their sum and one StateImage costs a full
  `memory.host_state_image_bytes` (~187 MiB on Qwen3.8-27B NVFP4 with DFlash2) regardless of the
  prefix depth it holds. The fork derives the whole tier from a single pinned-RAM ceiling: it counts
  the checkpoint inventory the capture path creates, `(2 + long anchors) × private continuations +
  shared prefixes` images, caps State at half the budget, spends the remaining state headroom on
  **more long anchors per continuation** — bounded by the anchor count whose re-prefill gap still
  outweighs one image — re-sizes the pool for the grown count, gives Host KV the remainder, and
  refuses to start rather than overcommit. The conversation count follows `--max-concurrency`
  (default `2×`) and the shared-prefix catalog `max(concurrency, 7)`, so the leftover budget is
  divided across those owners as extra anchors: a low-concurrency server buys depth per conversation,
  a high-concurrency server buys breadth. The component flags still work standalone and are rejected
  alongside a budget.
- **Salvaged prefills and automatic anchoring** — prefilled context is salvaged when a request is
  aborted, so a retry resumes from the salvaged frontier instead of from root; the engine anchors
  the last N message boundaries of a conversation automatically
  (`--max-long-anchors-per-continuation`); and a lane publishes with its staged-prefill bookkeeping
  cleared.
- **Cost-scaled materialization search budget** — addresses the crux of
  [Neroued/ninfer#229](https://github.com/Neroued/ninfer/issues/229): the flat 5 ms
  under-pressure materialization-search budget is not sufficient — it only manages to
  get through searching about 10 targets before it times out, so valuable completion
  plans are missed. Implements the solution suggested by Gene0Liu: scale the
  materialization search budget with the number of targets, subject to a 250 ms cap
  (which is almost always hit) — an expensive incumbent earns up to the full 250 ms
  search, a cheap one keeps the 5 ms floor.
- **Automatic shared-prefix catalog reclaim** — looks to resolve
  [Neroued/ninfer#251](https://github.com/Neroued/ninfer/issues/251): the shared
  stable-prefix catalog saturated and did not evict its least-recently-used entries —
  with no eviction path for automatic-evidence traffic, once every
  `--max-shared-prefixes` slot was resident, later automatic candidates were dropped and
  their shared-prefix reuse froze until an engine restart. Implements the solution
  suggested by albertov: reclaim the least-recently-used eligible automatic entry when a
  candidate finds no vacant slot, and count reclaimable slots as publication slack so the
  materialization selection stops discarding automatic candidates at saturation.
- **Lease and resolved-capacity observability** — the `--request-log-jsonl` occupancy record reports
  the unmaterialised part of the Device KV lease separately from `allocated + reserved` pages, so a
  leased-but-unwritten pool is visible as a lease rather than indistinguishable from real KV, and
  the `server_start` memory ledger reports the Host tier's *resolved* slot counts with both unit
  costs and the derived budget split, not the requested ones.

Measured net effect on a replay of the traffic that motivated this — 26 multi-turn tool-agent
requests, `--host-cache-mib 40000`, Qwen3.8-27B NVFP4 + DFlash2, `--max-concurrency 2`, RTX 5090:
token-level prefix reuse 90.75 % overall and 98.08 % once warm, against 1.44 % across the
chronic production window the replay was built from; private owners fully evicted dropped from 179
to 2–3 and clear-all fallbacks from 55 to 0–1, with no mid-conversation root fallback anywhere. The
replay runs 3.7 minutes, so it demonstrates the mechanism rather than hours-long saturation
behaviour.

**Features**

- **Ngram copy drafting above one concurrent request** — the local contribution here is
  the C>1 extension of remesis's C=1 implementation: it makes ngram copy drafting work
  with more than one concurrent request across MTP/DFlash/DFlash2, with a per-lane
  decode frame, the cross-request retention archive enabled at concurrency above one,
  and startup validation of the GDN width/concurrency limit.
- **Vision offload re-implementation** — the local V3 re-implementation of the overlay
  residency above: `--vision-residency overlay` streams the vision tower from pinned host
  RAM instead of holding it on the GPU (per-object overlay staging), freeing device
  memory for KV, including the fix that made the overlay path work for every artifact.
- **`--rope-yarn-factor`** — startup-fixed YaRN context extension (factor 1–4, capped at
  the 1M visible-keys limit).
- **Q8 MTP with mixed-format row-split projection**, and a **runtime-shape bf16 GEMM
  fallback** for shapes without a specialised kernel (full-precision vocab heads, bf16
  vision-tower projections).
- **Quasar NVFP4 conversion fixes** — dflash2 head-use declarations and an indexed
  proposal head (`--proposal`) so rebuilt artifacts support `--lm-head-draft`.

**Numerics**

- **Accurate `silu` in the NVFP4 fused SwiGLU TMA epilogue** — reverts the approximate
  activation upstream took in PR #250 (commit `05507ab0`), restoring `silu` at the four
  call sites in `nvfp4_linear_swiglu_w4a4_tma.cuh` and deleting the now-unused
  `silu_approx` helper. Follows
  [Neroued/ninfer#285](https://github.com/Neroued/ninfer/issues/285), where
  [bingchengcc](https://github.com/bingchengcc) measured the trade at model level rather
  than op level: over 47,917 scored tokens the approximate form costs +0.009323 absolute
  (+0.61 % relative) corpus perplexity (1.540180 against 1.530857), while the prefill wall
  clock it buys back is only about 1.4 % (TTFT 1.832 s against 1.858 s on a 16,817-token
  prompt) — far short of the ~10.8 % the Op benchmark reported — with decode unchanged.
  Quality was judged the better side of that trade; every other SwiGLU epilogue in the
  tree already computes the accurate form.

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
in its single-knob form: the `.bat`'s `--host-kv-mib` / `--host-state-slots` /
`--max-long-anchors-per-continuation` / catalog flags are replaced by the one `--host-cache-mib`
ceiling, which rejects them alongside it.

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
retention tier in its single-knob form. At `--max-concurrency 2` this artifact's 52,000 MiB budget
resolves to 139 Host StateImages of 195,897,344 B — 31 long anchors per continuation — with the
remaining ≈26,000 MiB given to Host KV; the resolved split is what the `server_start` memory ledger
reports.

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

Everything below is a **direct, unmodified copy of the upstream
[NInfer README](https://github.com/Neroued/ninfer/blob/master/README.md)**, as of the
latest upstream sync (`9e163eee` on `origin/master`).

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
