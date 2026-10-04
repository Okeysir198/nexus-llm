# LFM2.5-8B-A1B (Liquid flagship) — benchmarks (2026-10-04)

Setup: stock `vllm/vllm-openai:v0.25.0-aarch64` (has `Lfm2MoeForCausalLM` and
the `lfm2` tool parser registered — no patches needed), BF16 weights ~16.6 GB,
8 GiB KV pin → 698,516 KV tokens, max-model-len **128000** (native limit —
131072 hard-crashes vLLM's pydantic validation, don't raise), 24 seqs,
prefix caching + chunked prefill. Same harness / 6k-token prompt as the other
engines, via the :18083 bridge.

## Results

| metric | value |
|---|---|
| Engine VRAM | 25.6 GB (BF16 + 8 GiB KV pin) |
| Single-stream decode | 54.3 tok/s (ITL 18.4 ms) |
| TTFT @6k prompt (warm) | 0.35–0.71 s — best of the fleet |
| Conc-10 aggregate | **134–145 output tok/s** (≈ Ornith's 138) |
| Conc-10 tool verdicts | 8 perfect + 1 partial / 9 acc (better than Ornith's 7–8) |
| Conc-10 success | 10/10 |

It's a thinking model (`<think>` blocks) — reasoning tokens count against
client max_tokens; the harness disables thinking via chat_template kwargs.

## Fleet comparison (all measured 2026-10-04, same harness)

| engine | VRAM | single tok/s | conc-10 agg | tool verdicts | TTFT @6k |
|---|---|---|---|---|---|
| Ornith-1.5-35B-A3B-NVFP4 | 30.3 GB | 72.7 | 138 | 7 perfect / 8 acc | 0.7–1.7 s |
| **LFM2.5-8B-A1B** | **25.6 GB** | 54.3 | 134–145 | **8 perfect / 9 acc** | **0.35–0.71 s** |
| Ternary-Bonsai-2-27B PTQ1_0 | 8.6 GB | 31.3 | 36–53 | 9 perfect / 10 acc | 0.9–7 s |
| Qwen3.8-27B-NVFP4 | 33.7 GB | 9.8 | 51 | 9–10 perfect | 0.7–3.4 s |

## Verdict

LFM2.5 delivers Ornith-class concurrency (134 vs 138 tok/s) and BETTER tool
verdicts at 4.7 GB less VRAM, with the best warm TTFT of the fleet. The
"low-VRAM" claim is only partly true in BF16 (25.6 GB); a GGUF build (~5 GB
via llama.cpp) would be the true low-memory mode. The DSpark drafter (claimed
~2.5x) remains untested — invariant 7 warns DSpark drafters lost on GB10 for
Bonsai; only try with a before/after benchmark.

Recommended team fit: co-default with Ornith (LFM for high-concurrency
routine work + TTFT-sensitive chat, Ornith for complex orchestration and
code), Bonsai stays the ultra-low-VRAM specialist, Qwen3.8 retired from
routing.

## Optimization pass (2026-10-04, follow-up): FP8 + 4 GiB KV pin

Target: max speed, lowest VRAM, healthy at 5 concurrent streams.

| config | VRAM | single tok/s | ITL | conc-10 agg | conc TTFT avg |
|---|---|---|---|---|---|
| BF16, 8 GiB pin (first boot) | 25.6 GB | 54.3 | 18.4 ms | 134–145 | 25–43 s |
| **FP8 weights + 4 GiB pin (final)** | **13.7 GB** | **74.4** | **13.4 ms** | **219–237** | ~25 s |

- FP8 dynamic (`--quantization fp8`) is FASTER than BF16 on GB10 (+37%
  single, +50–63% concurrent) — FP8 GEMMs are native on Blackwell/GB10 and
  halve decode bandwidth. KV 349,258 tokens = 5 concurrent 64k streams with
  headroom (2.7 full 128k windows).
- Tool verdicts after FP8: 8 perfect + 1 partial / 9 acc — one notch of
  perfect→partial shift vs BF16, all still acceptable.
- First request after boot ~22 s TTFT (one-time FP8/graph warmup); warm
  TTFT 250 ms.
- Final config: `--quantization fp8`, `--kv-cache-memory-bytes 4294967296`,
  max-model-len 128000. To revert to BF16: set `LLM_LFM_QUANT=off`-style
  comment-out of the --quantization line and 8 GiB pin.

## Optimization round 2 (2026-10-04): further levers tested

| lever | result |
|---|---|
| FP8 KV cache (`--kv-cache-dtype fp8_e4m3`) | ❌ REJECTED — breaks lfm2 tool calls (10/10 wrong or empty tool_calls; speed unchanged). Do not re-enable without re-testing tools. |
| VLLM_USE_OINK_OPS=1 (Blackwell RMSNorm) | ❌ BROKE lfm2 tool calls (empty tool_calls). Disabled. |
| DSpark speculative decoding (official LiquidAI drafter, spec=1/3) | ❌ NOT SERVABLE — arch `Lfm2DSparkDraftModel` unregistered in every available vLLM build (v0.25.0-aarch64, eugr dev21553, cu134-nightly 2026-10-04). llama.cpp-only today (DSpark-GGUF). Claimed 2.5–3.2x stays unverified on this box. |
| `--reasoning-parser qwen3` | ✓ KEPT — official recipe; separates <think> from content. NOTE: does NOT work in streaming mode for this model (think streams as content); harmless. |
| chat_template_kwargs.enable_thinking=false | ⚠⚠ CRITICAL: LFM2.5's template MISHANDLES it — with it the model answers in prose and NEVER emits tool_calls (looked like "parser mismatch" in tests/shared.py). Removed from the harness for LFM. |

**Final config**: FP8 weights + 4 GiB BF16 KV pin + reasoning-parser qwen3 +
lfm2 tool parser + prefix caching — 13.7 GB, 74.4 tok/s single, 219–237 tok/s
conc-10 aggregate. Further optimization is blocked on upstream (Lfm2DSpark
vLLM registration; FP8-KV+lfm2-parser compatibility).
