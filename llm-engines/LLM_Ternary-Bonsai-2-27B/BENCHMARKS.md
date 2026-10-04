# Ternary-Bonsai-2-27B (GGUF) — benchmarks (2026-10-04)

Setup: PrismML llama.cpp fork (`prism` branch @2459f68, 2026-10-02) built from
source for sm_121 (no linux-cuda-aarch64 release exists — see Dockerfile /
Dockerfile.runtime; stock llama.cpp produces garbage on these quants).
`-ngl 99`, `--flash-attn on`, KV q8_0, ctx 65536 over 4 slots, `-n 16384`
(reasoning model — small budgets return empty replies). Same harness and
6,167-token prompt as the other engines, via the :18083 bridge.

## Quant A/B

| | PTQ1_0 (1.72 bpw, default) | PQ2_0 (2.13 bpw) |
|---|---|---|
| GGUF size | 5.95 GB | 7.21 GB |
| Engine VRAM | 8.6 GB | 10.1 GB |
| Single-stream decode | **31.3 tok/s** (ITL 31.9 ms) | 26.6 tok/s (ITL 37.5 ms) |
| TTFT @6k prompt | 7.2 s | 6.9 s |
| Conc-10 aggregate | 36–53 output tok/s | 53 output tok/s |
| Conc-10 tool verdicts | 9 perfect + 1 partial | 9–10 perfect |
| Conc-10 success | 10/10 | 10/10 |

**PTQ1_0 is the default**: smaller, faster single-stream (the fork's latest
commit is a fused FWHT quantizer for this layout), same quality tier. PQ2_0
buy: nothing measurable here. (Card claims ~130 tok/s on RTX 5090 for PQ2_0;
GB10 aarch64 CUDA kernels are newer/slower for this format.)

## vs the vLLM engines on this box

| engine | VRAM | single tok/s | conc-10 agg | tool verdicts |
|---|---|---|---|---|
| Ornith-1.5-35B-A3B-NVFP4 | 30.3 GB | 72.7 | 138 | 7 perfect / 8 acc |
| Ternary-Bonsai-2-27B PTQ1_0 | 8.6 GB | 31.3 | 36–53 | **9 perfect / 10 acc** |
| Qwen3.8-27B-NVFP4 (prior bench) | 36 GiB | — | — | — |

Bonsai is the VRAM/speed/quality trade: 3.5× less VRAM than Ornith, ~43% of
its single-stream speed, best tool-calling accuracy of the three, but llama.cpp
TTFT under queued concurrency is poor (prefill doesn't batch like vLLM's
chunked prefill) — TTFT avg 35–55 s at conc-10 vs 12–16 s on Ornith.

## Gotchas hit during bring-up

- Fork's arg parser rejects `--flag=value` for --model ("invalid argument")
  — compose uses space-separated args throughout.
- llama-server needs libllama*/libmtmd*/libgomp1 beside it (Dockerfile.runtime).
- Build needs the CUDA driver stub exposed as libcuda.so AND libcuda.so.1
  (soname) or the final link fails on cuMem* symbols.
- Build is CPU-heavy: cap it (`--cpus 8`) — an uncapped compile pushed this
  board past 90 °C once already.

## TTFT tuning (2026-10-04, follow-up)

| change | single TTFT @6k | tool TTFT | conc-10 TTFT avg | conc-10 agg |
|---|---|---|---|---|
| baseline (20 threads, ub 512) | 7.24 s | 8.50 s | 35–55 s | 36–53 tok/s |
| threads 8 + ub 2048 + batch 4096 | **0.88 s** (warm same-prompt) | **3.24 s** | ~57 s | ~35 tok/s |
| + 8 slots / 98304 ctx | — | — | 87 s (WORSE) | 23 tok/s |

- `--cache-reuse` (KV prefix reuse) is **rejected** by llama-server for this
  model: the hybrid linear-attention layers hold recurrent state, not reusable
  KV — shared-prefix caching is architecturally unavailable. Same-prompt TTFT
  still improves via plain slot KV carry-over.
- 8 slots lose: concurrent prefills compete for compute instead of queueing.
  4 slots × 16384 ctx is the sweet spot on GB10.
- Remaining conc-10 TTFT (~55 s) is compute-bound ternary prefill on
  slow-ish CUDA kernels — the structural fix would be serving this quant
  through vLLM once it supports PTQ1_0/PQ2_0, or keeping Bonsai as the
  low-VRAM/single-stream route and using Ornith for concurrent traffic.
